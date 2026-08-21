"""lewm: a LeWorldModel (LeWM / JEPA) LEARNED latent world model, ARC-adapted.

This is Layer 3 of the ARC world-engine: a genuinely learned latent dynamics
model in the spirit of LeWM (arXiv:2603.19312), the "two-loss" JEPA world
model. It is deliberately NOT a 224px ViT -- ARC-AGI-3 grids are small
(<= 64x64) with 16 discrete colours and a discrete action set
(ACTION1..ACTION7), so the whole architecture is sized for that regime.

Faithful to the paper's core recipe:

  * TWO-TERM LOSS, nothing else:
      L = MSE(predictor(z_t, a_t), encoder(grid_{t+1}))          # prediction
        + lambda * SIGReg(Z)                                     # anti-collapse
    with lambda = 0.1.
  * NO EMA, NO stop-gradient, NO pretrained encoder. The prediction target is
    the encoder's OWN output for the next frame; the ONLY thing standing
    between that and trivial representation collapse (encode everything to a
    constant -> MSE 0) is SIGReg.
  * SIGReg (Sketched-Isotropic-Gaussian Regularizer): project the batch of
    latents onto M random unit-norm directions and, by the Cramer-Wold
    theorem, push every 1D marginal toward N(0,1) via the differentiable
    Epps-Pulley characteristic-function statistic. Matching all 1D marginals
    == matching the full joint N(0, I), which provably prevents ALL collapse
    modes (not just the marginal-variance / covariance modes a VICReg-style
    penalty covers).

ARC adaptation specifics:

  * encoder(grid) -> z: pad/crop the variable-size grid to a fixed GxG
    (default 32x32), embed each cell colour (nn.Embedding 16 -> 8), a small
    CNN, then an MLP head -> latent z (dim ~96).
  * predictor(z, action_onehot) -> z_hat_next: a small MLP with the discrete
    action one-hot concatenated to the latent.
  * CEM planner over DISCRETE actions: maintain a per-step categorical
    distribution over ACTION1..ACTION7, sample N horizon-H sequences, roll
    them out IN LATENT SPACE through the predictor, score by terminal
    ||z_H - z_goal||^2, refit the per-step categoricals to the top-K elites,
    iterate; return the best elite's action sequence.
  * surprise(grid, a, next_grid) = ||predictor(enc(grid), a) - enc(next)||^2,
    the model's own prediction error -- a clean stuck/anomaly signal.

State is per-episode by default (a fresh model + fresh replay buffer each
game), mirroring the per-episode lifecycle of prism_micro / action_memo.

Optional-heavy-dependency discipline (mirrors how prism_micro.py guards
numpy): the torch import is wrapped in try/except; if torch is unavailable the
module still imports cleanly and every method degrades gracefully with
self.ok == False -- it NEVER raises into the caller.

Default OFF. The whole thing is gated behind ARC_WORLD_ENGINE=1 (checked at
the caller via lewm.enabled(), mirroring the grid_topology / prism gate
convention); baseline behaviour is byte-identical when the flag is unset.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, NamedTuple, Optional, Sequence

logger = logging.getLogger(__name__)

# -- optional heavy dependency guard (mirrors prism_micro's numpy guard) -------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except Exception:  # pragma: no cover - torch is an optional dependency here
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_OK = False

# numpy is a hard dependency of arcengine, but guard it too for parity/safety.
try:
    import numpy as np
    _NUMPY_OK = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_OK = False

_ACTION_ORDER = [f"ACTION{i}" for i in range(1, 8)]  # ACTION1..ACTION7

# "ACTION6(19,16)" | "ACTION6 (19, 16)" | "ACTION3"
_ACTION_RE = re.compile(r"^ACTION([1-9][0-9]*)\s*(?:\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\))?$")


class ParsedAction(NamedTuple):
    idx: int    # 0-based predictor index in [0, A); -1 when reset
    x: int      # -1 when the action carries no coordinates
    y: int      # -1 when the action carries no coordinates
    reset: bool


class _Tr(NamedTuple):
    """One buffered transition. Fields 0..5 keep their original order and meaning, so
    every existing positional access (b[0], b[2], b[4], ...) still reads correctly."""
    ids_t: Any          # [G,G] uint8
    a: int              # 0-based action index -- now guaranteed a REAL action
    ids_tp1: Any        # [G,G] uint8
    game: Optional[str]
    cond_t: Any         # [D_c] f16 | None
    cond_tp1: Any       # [D_c] f16 | None
    ax: int = -1        # click x, -1 = action carries no coordinates
    ay: int = -1        # click y
    ep: int = -1        # episode id (per game) -- lets the rollout loss chain steps
    t: int = -1         # step index within the episode
    chg: bool = True    # did this transition change >=1 cell (precomputed)
    effmask: Any = None  # [G,G] uint8 | None -- LLM-denoised per-cell EFFECT mask (Stage B).
                         #   APPENDED so every positional read (b[0..5], .ax/.ay/.chg) is
                         #   unchanged: ax=6 ay=7 ep=8 t=9 chg=10 effmask=11. len(b)>11 gates it.
    # -- Part A: reward/termination for the VALUE head (same safe-append discipline) --------
    # reward=12 done=13 legal=14. All optional with neutral defaults, so an old checkpoint's
    # buffer (schema without these) reads identically and the value head just sees r=0/done=0.
    reward: float = 0.0  # extrinsic reward at this transition (Δlevels_completed - death penalty)
    done: bool = False   # terminal (GAME_OVER or WIN) -- stops the n-step return walk
    legal: Any = None    # available_actions mask/list at grid_t, or None


def parse_action(action: Any, n_actions: int = 7) -> Optional[ParsedAction]:
    """THE action parser. Returns None for anything unrecognised -- NEVER a silent 0.

    The silent-0 fallback this replaces is the single most damaging bug this model has
    had. The solver formats a click as "ACTION6(19,16)"; the old code did
    int("6(19,16)"), caught the ValueError, and returned 0 -- which is ALSO the index of
    a real ACTION1. So index 0 became a garbage bucket holding real ACTION1s, every
    click (coordinates discarded), every RESET, and every first-turn None. It was 82% of
    the replay buffer, its conditional mean was "nothing changes", and identity therefore
    became the Bayes-optimal prediction. No loss function can survive that, and three
    attempts to fix it in the loss all failed before the cause was found here.

    Accepts:
      "ACTION6(19,16)" / "ACTION6 (19, 16)" -> idx 5, x=19, y=16
      "ACTION3" / "action3"                 -> idx 2, x=y=-1
      "RESET"                               -> reset sentinel (idx -1)
      ("ACTION6", 19, 16) / ("ACTION3",)    -> as above  (the solver's _last_action_tuple)
      {"name"|"id": 6, "x": 19, "y": 16}    -> idx 5, coords
      int n in [1, n_actions]               -> idx n-1   (1-BASED ARC id, matching
                                               GameAction.value and the ARC API)
      int 0                                 -> reset sentinel
      None / "None" / "" / anything else    -> None

    An id outside [1, n_actions] returns None rather than being CLAMPED -- clamping is
    how out-of-range garbage got into index 0 in the first place.
    """
    x = y = -1

    if isinstance(action, dict):
        raw = action.get("name", action.get("id"))
        if action.get("x") is not None and action.get("y") is not None:
            try:
                x, y = int(action["x"]), int(action["y"])
            except Exception:
                x = y = -1
        action = raw
    elif isinstance(action, (tuple, list)) and action:
        if len(action) >= 3:
            try:
                x, y = int(action[1]), int(action[2])
            except Exception:
                x = y = -1
        action = action[0]

    if action is None:
        return None

    n: Optional[int] = None
    if isinstance(action, str):
        s = action.strip().upper()
        if s in ("", "NONE"):
            return None
        if s == "RESET":
            return ParsedAction(-1, -1, -1, True)
        m = _ACTION_RE.match(s)
        if m:
            n = int(m.group(1))
            if m.group(2) is not None:
                x, y = int(m.group(2)), int(m.group(3))
        else:
            try:
                n = int(s)            # bare numeric string -> 1-based id
            except ValueError:
                return None
    else:
        try:
            n = int(action)           # int / np-int / 0-dim tensor -> 1-based id
        except Exception:
            return None

    if n == 0:
        return ParsedAction(-1, -1, -1, True)     # RESET
    if n is None or not (1 <= n <= n_actions):
        return None
    return ParsedAction(n - 1, x, y, False)


def enabled() -> bool:
    """The default-OFF gate. Callers must check this before using the model,
    so baseline behaviour is byte-identical when ARC_WORLD_ENGINE is unset."""
    return os.environ.get("ARC_WORLD_ENGINE", "") not in ("", "0", "false", "False")


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def _pick_device(requested: Optional[str] = None):
    """Resolve the torch device to run on. Precedence:

      1. an explicit `requested` argument ("cpu" | "cuda" | "auto"), else
      2. the ARC_WM_DEVICE env override ("auto" | "cpu" | "cuda"; default "auto").

    "auto" (the default) picks cuda iff torch.cuda.is_available(); "cpu" pins
    CPU even when a GPU is present; "cuda" asks for GPU but degrades to CPU
    (with a warning) when no CUDA device is actually available -- never raises.
    Returns None when torch is unavailable. Keeps CPU behaviour byte-identical
    to the old code path when no CUDA device exists."""
    if not _TORCH_OK:
        return None
    choice = (requested if requested is not None
              else os.environ.get("ARC_WM_DEVICE", "auto"))
    choice = str(choice).strip().lower()
    if choice == "cpu":
        return torch.device("cpu")
    try:
        has_cuda = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - defensive; cuda probe should not throw
        has_cuda = False
    if choice == "cuda":
        if has_cuda:
            return torch.device("cuda")
        logger.warning("lewm: ARC_WM_DEVICE=cuda but no CUDA device available; using cpu")
        return torch.device("cpu")
    # "auto" (or anything unrecognised): cuda iff available, else cpu.
    return torch.device("cuda" if has_cuda else "cpu")


# -- nn.Module definitions (only when torch is present) ------------------------
if _TORCH_OK:

    class _Encoder(nn.Module):
        """grid (GxG ints in [0, C)) -> latent z (dim = latent).

        Colour embedding -> small CNN -> MLP head. No batchnorm/dropout, so
        train/eval mode is irrelevant (keeps train_step / plan symmetric)."""

        def __init__(self, grid: int, colours: int, colour_emb: int, latent: int) -> None:
            super().__init__()
            self.grid = grid
            self.emb = nn.Embedding(colours, colour_emb)
            self.conv = nn.Sequential(
                nn.Conv2d(colour_emb, 16, kernel_size=3, stride=2, padding=1),  # G -> G/2
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # G/2 -> G/4
                nn.ReLU(),
            )
            feat = (grid // 4) * (grid // 4) * 32
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat, 128),
                nn.ReLU(),
                nn.Linear(128, latent),
            )

        def forward(self, ids):  # ids: [B, G, G] long
            e = self.emb(ids)                 # [B, G, G, Cemb]
            e = e.permute(0, 3, 1, 2).contiguous()  # [B, Cemb, G, G]
            h = self.conv(e)                  # [B, 32, G/4, G/4]
            return self.head(h)               # [B, latent]

    class _CondFiLM(nn.Module):
        """FiLM / AdaLN modulation of the latent by an auxiliary conditioning
        vector: z' = z * (1 + gamma(cond)) + beta(cond).

        gamma/beta are produced by a single zero-initialised Linear, so an
        untrained FiLM is the IDENTITY -- enabling the aux channel (D_c>0) on a
        model loaded from a grid-only checkpoint perturbs nothing until the head
        learns to use it (a byte-identical warm start). This is the surgical
        multimodal fusion seam: every non-grid sense enters here as `cond`."""

        def __init__(self, cond_dim: int, latent: int) -> None:
            super().__init__()
            self.to_gb = nn.Linear(cond_dim, 2 * latent)
            nn.init.zeros_(self.to_gb.weight)
            nn.init.zeros_(self.to_gb.bias)

        def forward(self, z, cond):            # z: [B, D], cond: [B, Dc]
            gamma, beta = self.to_gb(cond).chunk(2, dim=-1)
            return z * (1.0 + gamma) + beta

    class _Predictor(nn.Module):
        """(z, action_one_hot[, ctx]) -> z_hat_next. Residual MLP with FiLM action
        conditioning.

        The action MULTIPLICATIVELY modulates EVERY latent dimension (FiLM:
        z' = z*(1+gamma(a)) + beta(a)) BEFORE the MLP, instead of being a tiny
        7-of-263 concatenated one-hot the network learns to ignore. gamma/beta come
        from a ZERO-initialised Linear, so an untrained FiLM is the IDENTITY —
        early training behaves like the old action-agnostic predictor, then the
        head learns per-action modulation. This is what lets the model predict
        DIFFERENT effects for different actions (the old concat couldn't). An
        optional top-down ctx vector still concatenates for goal conditioning."""

        def __init__(self, latent: int, n_actions: int, ctx_dim: int = 0) -> None:
            super().__init__()
            self.ctx_dim = int(ctx_dim)
            self.act_film = nn.Linear(n_actions, 2 * latent)   # action -> (gamma, beta)
            # SMALL-RANDOM (not zero) init: zero-init sits exactly at the symmetric
            # saddle where every action produces the SAME modulation and the
            # differentiation gradient VANISHES — training can never break out. A
            # small perturbation seeds distinct per-action modulation the losses
            # can then amplify. Kept small so it's near-identity at start.
            nn.init.normal_(self.act_film.weight, std=0.05)
            nn.init.zeros_(self.act_film.bias)
            self.net = nn.Sequential(
                nn.Linear(latent + self.ctx_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, latent),
            )

        def forward(self, z, a_onehot, ctx=None):   # z:[B,D] a_onehot:[B,A] ctx:[B,ctx_dim]
            gamma, beta = self.act_film(a_onehot).chunk(2, dim=-1)
            zc = z * (1.0 + gamma) + beta          # ACTION strongly modulates the latent
            parts = [zc]
            if self.ctx_dim > 0:
                if ctx is None:
                    ctx = z.new_zeros((z.shape[0], self.ctx_dim))
                parts.append(ctx)
            delta = self.net(torch.cat(parts, dim=-1))
            return z + delta                  # residual: predict the change

    class _Decoder(nn.Module):
        """latent z (dim=latent) -> per-cell colour LOGITS [B, C, G, G].

        The encoder collapses ALL spatial layout into a flat `latent`-d code, so
        this is a genuine generative upsampler, not a cheap conv: project z to a
        tiny 4x4 seed, then ConvTranspose x N up to GxG. It renders the world
        model's imagination -- impressionistic by construction, since it is
        decoding a global bottleneck. Trained decoder-only against a DETACHED z
        (see train_step), so it NEVER perturbs the JEPA encoder/predictor."""

        def __init__(self, latent: int, colours: int, grid: int, seed_ch: int = 128) -> None:
            super().__init__()
            self.seed = 4
            self.seed_ch = seed_ch
            self.fc = nn.Linear(latent, seed_ch * self.seed * self.seed)
            n_up, g = 0, self.seed
            while g < grid:                     # 4 -> ... -> G (one stage per 2x)
                g *= 2
                n_up += 1
            layers, ch = [], seed_ch
            for i in range(max(1, n_up)):
                out = colours if i == n_up - 1 else max(colours, ch // 2)
                layers.append(nn.ConvTranspose2d(ch, out, kernel_size=4, stride=2, padding=1))
                if i < n_up - 1:
                    layers.append(nn.ReLU())
                ch = out
            self.net = nn.Sequential(*layers)

        def forward(self, z):                   # z: [B, latent]
            h = self.fc(z).view(-1, self.seed_ch, self.seed, self.seed)  # [B, seed_ch, 4, 4]
            return self.net(h)                  # [B, C, G, G] logits

    # ----- SPATIAL-LATENT variants (ARC_WM_SPATIAL=1) -------------------------
    # The flat modules above collapse the whole board into one global vector,
    # which (a) can't reconstruct complex boards (a 256-float MSE bottleneck
    # can't carry 4096 cells x 4 bits) and (b) re-renders the ENTIRE board on any
    # latent move (board-wide over-swing). The spatial variants keep the latent
    # as a [Dch, Gh, Gw] FEATURE MAP (Gh=Gw=G/4): every token owns a 4x4 patch,
    # so reconstruction is faithful, a conv predictor with bounded receptive
    # field changes only LOCAL regions, and decoding preserves 2D correspondence.
    # z stays a FLAT vector on the wire (length Dch*Gh*Gw) via reshape, so the
    # /encode /decode /dream contract and the client are unchanged.

    class _SpatialEncoder(nn.Module):
        """grid (GxG) -> spatial latent map [Dch, G/4, G/4], FLATTENED to [B, L].

        No normalization layers: the flat model trained stably without any on this
        exact (free-threaded python + CUDA torch) stack, whereas GroupNorm's CUDA
        kernel crashed it (double-free). Convs only; SIGReg controls latent scale."""

        def __init__(self, grid: int, colours: int, colour_emb: int, dch: int) -> None:
            super().__init__()
            self.grid = grid
            self.gh = grid // 4
            self.dch = dch
            self.emb = nn.Embedding(colours, colour_emb)
            self.conv = nn.Sequential(
                nn.Conv2d(colour_emb, 32, 3, stride=2, padding=1), nn.ReLU(),  # G -> G/2
                nn.Conv2d(32, dch, 3, stride=2, padding=1), nn.ReLU(),         # G/2 -> G/4
            )

        def forward(self, ids):                 # ids: [B, G, G] long
            e = self.emb(ids).permute(0, 3, 1, 2).contiguous()   # [B, Cemb, G, G]
            m = self.conv(e)                                     # [B, Dch, G/4, G/4]
            return m.flatten(1)                                  # [B, L]

    class _SpatialResBlock(nn.Module):
        """Conv residual block (no norm — see _SpatialEncoder). Residual keeps it
        stable at this depth."""

        def __init__(self, ch: int, dropout: float = 0.0) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
            # Channel dropout on the residual branch (LeWM's predictor-regularisation
            # lever). Dropout2d zeros whole feature maps, the right granularity for a conv
            # predictor; no-op at eval() and when p==0.
            self.drop = nn.Dropout2d(dropout) if dropout > 0 else None

        def forward(self, x):
            h = F.relu(self.c1(x))
            h = self.c2(h)
            if self.drop is not None:
                h = self.drop(h)
            return F.relu(x + h)

    class _SpatialPredictor(nn.Module):
        """(z_flat, action_one_hot) -> z_hat_flat. Per-CHANNEL FiLM action
        conditioning broadcast over the map + residual conv blocks with a bounded
        receptive field, so an action's predicted effect is LOCAL by construction
        (kills board-wide over-swing). z' = z + conv_stack(FiLM(z, a))."""

        def __init__(self, dch: int, gh: int, n_actions: int, ctx_dim: int = 0,
                     n_blocks: int = 3, coord_planes: int = 0, dropout: float = 0.0) -> None:
            super().__init__()
            self.dch = dch
            self.gh = gh
            self.ctx_dim = int(ctx_dim)
            self.coord_planes = int(coord_planes)
            self.act_film = nn.Linear(n_actions, 2 * dch)   # action -> per-channel (gamma, beta)
            nn.init.normal_(self.act_film.weight, std=0.05)  # break the symmetric saddle
            nn.init.zeros_(self.act_film.bias)
            # GAME conditioning: an ACTION means different things in different games, so
            # its (gamma, beta) must be game-relative. Added to the action's modulation.
            self.game_film = None
            if self.ctx_dim > 0:
                self.game_film = nn.Linear(self.ctx_dim, 2 * dch)
                nn.init.normal_(self.game_film.weight, std=0.02)
                nn.init.zeros_(self.game_film.bias)
            # CLICK conditioning: coordinate planes concatenated after the FiLM. The stem
            # is IDENTITY-INITIALISED on the passthrough channels and small-random on the
            # coord planes, and has NO activation after it -- so with all-zero coord planes
            # (i.e. a coord-free action, or an old checkpoint that has no stem at all)
            # stem(cat([h, 0])) == h EXACTLY. A freshly-added stem therefore leaves an
            # existing trained predictor bit-identical on day 0. A ReLU here would clip
            # negative latent values and silently destroy that warm start.
            self.stem = None
            if self.coord_planes > 0:
                self.stem = nn.Conv2d(dch + self.coord_planes, dch, 3, padding=1)
                with torch.no_grad():
                    w = torch.zeros(dch, dch + self.coord_planes, 3, 3)
                    for c in range(dch):
                        w[c, c, 1, 1] = 1.0                  # centre-tap identity
                    w[:, dch:].normal_(0, 0.02)              # coord planes: small random
                    self.stem.weight.copy_(w)
                    self.stem.bias.zero_()
            self.blocks = nn.ModuleList([_SpatialResBlock(dch, dropout) for _ in range(n_blocks)])
            self.head = nn.Conv2d(dch, dch, 3, padding=1)
            # SMALL-RANDOM (not zero): a zero head makes z'=z for EVERY action
            # (identity) AND zeroes the gradient into the conv blocks, so the action
            # path can never learn to differentiate — the same symmetric saddle that
            # pinned the flat model's FiLM (act_contrast stuck at the margin). A small
            # init keeps the residual near-identity while letting per-action
            # modulation propagate and grow.
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)

        def forward(self, z, a_onehot, ctx=None, coord=None):
            # z:[B,L]  a_onehot:[B,A]  ctx:[B,ctx_dim] game emb  coord:[B,P,Gh,Gw]
            B = z.shape[0]
            zm = z.view(B, self.dch, self.gh, self.gh)          # [B, Dch, Gh, Gw]
            mod = self.act_film(a_onehot)                       # [B, 2*Dch]
            if self.game_film is not None and ctx is not None:
                mod = mod + self.game_film(ctx)                 # action's meaning is game-relative
            gamma, beta = mod.chunk(2, dim=-1)                  # each [B, Dch]
            gamma = gamma.view(B, self.dch, 1, 1)
            beta = beta.view(B, self.dch, 1, 1)
            h = zm * (1.0 + gamma) + beta                       # action modulates every token
            if self.stem is not None:
                if coord is None:
                    coord = zm.new_zeros(B, self.coord_planes, self.gh, self.gh)
                h = self.stem(torch.cat([h, coord], dim=1))     # NO activation -- see __init__
            for blk in self.blocks:
                h = blk(h)
            delta = self.head(h)                                # [B, Dch, Gh, Gw]
            return (zm + delta).flatten(1)                      # residual, back to [B, L]

    class _GradScale(torch.autograd.Function):
        """Identity forward; scales the gradient on the way back. Lets the inverse-
        dynamics loss shape the ENCODER by a tunable amount (0.0 = not at all), which is
        the safety valve protecting the banked 0.94 reconstruction."""

        @staticmethod
        def forward(ctx, x, scale: float):
            ctx.scale = float(scale)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return g * ctx.scale, None

    class _InverseDynamics(nn.Module):
        """IDM(Δz) -> (action_logits [B,A], click_logits [B, Gh*Gw]).

        THE anti-collapse mechanism. Forward prediction alone admits a trivial solution:
        map everything to a constant and predict that constant, and the forward loss is
        zero (Sensorimotor World Models, 2606.20104). Asking instead "which action does
        this displacement encode?" has a UNIQUE supervised answer, so a predictor cannot
        satisfy it by moving in a direction the decoder ignores -- which is exactly how
        every repulsion-based attempt was defeated here.

        Reads the DISPLACEMENT Δz = z_next - z_t, never the concatenation [z_t; z_next]:
        a concat lets the head shortcut the action from action-correlated absolute state
        at the endpoint without the model representing the TRANSITION at all
        (Delta-JEPA's "action-correlated endpoint shortcuts"; it ablates concat as worse
        by 4-12.6 points). A displacement carries only what moved.

        Mean AND max pooling for the action head: a click's signature is a single small
        local blob, which mean-pooling over 16x16 would dilute ~256x. Max-pooling is what
        lets it survive.
        """

        def __init__(self, dch: int, gh: int, n_actions: int, hidden: int = 64) -> None:
            super().__init__()
            self.dch = dch
            self.gh = gh
            self.body = nn.Sequential(
                nn.Conv2d(dch, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
            self.act_head = nn.Linear(2 * hidden, n_actions)   # [mean-pool ; max-pool]
            self.click_head = nn.Conv2d(hidden, 1, 1)          # -> [B,1,Gh,Gw]

        def forward(self, dz):                       # dz: [B, L] flat displacement
            B = dz.shape[0]
            h = self.body(dz.view(B, self.dch, self.gh, self.gh))
            pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
            return self.act_head(pooled), self.click_head(h).flatten(1)

    class _ValueHead(nn.Module):
        """V(z) -> scalar value of the state (the discounted future EXTRINSIC return).

        Mirrors _InverseDynamics' cheap conv-on-the-token-map design (not a Linear(D,...)):
        reads the [Dch, Gh, Gw] latent, a small conv trunk, mean+max pool, MLP -> 1 scalar.
        Trained against precomputed n-step returns on a DETACHED, frozen latent, so it never
        reshapes the encoder. Intrinsic reward (novelty) stays in the planner; this head is
        the TASK value only (avoid GAME_OVER, seek level-ups) -- the clean RL split of
        extrinsic value + intrinsic exploration bonus."""

        def __init__(self, dch: int, gh: int, hidden: int = 64) -> None:
            super().__init__()
            self.dch = dch
            self.gh = gh
            self.body = nn.Sequential(
                nn.Conv2d(dch, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, z):                        # z: [B, L] flat latent
            bsz = z.shape[0]
            h = self.body(z.view(bsz, self.dch, self.gh, self.gh))
            pooled = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
            return self.head(pooled).squeeze(-1)     # [B]

    class _FsAdapter(nn.Module):
        """A trainable view of the FROZEN latent, feeding ONLY the predictor.

        Measured, not guessed. Training the ENCODER on filesystem purposes (SupCon and
        push-only repel alike) reliably improved the forward model -- predictive coverage
        14.7 -> 18.3, from tying the blind baselines to beating them by 41% -- but just as
        reliably DEGRADED the map: curious coverage 18.3 -> 13.7 and target-reach 170 -> 196.
        The frozen ARC encoder's geometry is precisely what makes novelty/frontier work, and
        any purpose-shaped objective flattens the within-kind variety the landmark count is
        counting. The two competencies want DIFFERENT latents.

        So stop making one latent serve both: keep the trunk frozen (encode() / MapMemory /
        curious keep the good geometry) and give the PREDICTOR its own contrastively-separated
        view through this adapter. Residual 1x1 convs with a ZERO-INIT second layer -> exact
        identity at warm-start, so switching it on is byte-identical until it learns. Output
        stays in the frozen latent's space, so predict()'s result remains directly comparable
        to the map's landmarks (what _predicted_child_novelty needs)."""

        def __init__(self, dch: int) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(dch, dch, 1)
            self.c2 = nn.Conv2d(dch, dch, 1)
            nn.init.zeros_(self.c2.weight)
            nn.init.zeros_(self.c2.bias)        # identity at init

        def forward(self, zmap):                # [B, Dch, Gh, Gw]
            return zmap + self.c2(F.relu(self.c1(zmap)))

    class _SpatialDecoder(nn.Module):
        """spatial latent [B, L] -> per-cell colour logits [B, C, G, G] by
        upsampling the map. Token (i,j) governs its own patch -> faithful,
        spatially-correspondent reconstruction (and a local latent change stays
        a local pixel change)."""

        def __init__(self, dch: int, colours: int, grid: int) -> None:
            super().__init__()
            self.dch = dch
            self.gh = grid // 4
            layers = [
                nn.ConvTranspose2d(dch, 32, 4, stride=2, padding=1), nn.ReLU(),  # G/4 -> G/2
                nn.ConvTranspose2d(32, colours, 4, stride=2, padding=1),         # G/2 -> G
            ]
            self.net = nn.Sequential(*layers)

        def forward(self, z):                   # z: [B, L]
            B = z.shape[0]
            m = z.view(B, self.dch, self.gh, self.gh)
            return self.net(m)                  # [B, C, G, G] logits


class LeWorldModel:
    """LeWM/JEPA learned latent world model, ARC-adapted. See module docstring.

    Lifecycle: per-episode by default -- construct one per game. Every method
    is safe to call when torch is unavailable (self.ok == False): they degrade
    to no-ops / None / empty and never raise."""

    def __init__(
        self,
        grid: Optional[int] = None,
        latent: Optional[int] = None,
        n_actions: Optional[int] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.ok = _TORCH_OK
        # config (env-overridable; args win over env)
        self.G = int(grid if grid is not None else _envi("ARC_WM_GRID", 32))
        self.C = _envi("ARC_WM_COLORS", 16)
        self.Cemb = _envi("ARC_WM_COLOR_EMB", 8)
        self.D = int(latent if latent is not None else _envi("ARC_WM_LATENT", 96))
        self.A = int(n_actions if n_actions is not None else _envi("ARC_WM_ACTIONS", 7))
        # SPATIAL latent: keep a [Dch, G/4, G/4] feature map instead of a flat
        # global vector. When on, the flat wire length D is DERIVED (Dch*Gh*Gw) so
        # /encode still returns a flat list; Gh=Gw=G/4. Default ON (the redesign).
        self.spatial = _envi("ARC_WM_SPATIAL", 1) not in (0,)
        self.Dch = _envi("ARC_WM_DCH", 32)
        self.Gh = self.Gw = self.G // 4
        if self.spatial:
            self.D = self.Dch * self.Gh * self.Gw   # flat wire length of the map
        # feature dim SIGReg operates on: per-token (Dch) when spatial else flat D
        self._sig_feat_dim = self.Dch if self.spatial else self.D
        # Multimodal fusion widths (0 = off -> byte-identical grid-only model).
        # D_c: auxiliary/sensory conditioning fused into the latent via FiLM (the
        #      CNS Sensorium vector: hardware ⊕ world2d ⊕ affect ⊕ internal ⊕ …).
        # ctx_dim: top-down goal/attention conditioning fed to the predictor.
        self.D_c = int(_envi("ARC_WM_COND_DIM", 0))
        # ctx_dim is now the GAME-EMBEDDING width (was plumbed end-to-end but dead at 0).
        self.ctx_dim = int(_envi("ARC_WM_CTX_DIM", 16))
        self.lr = _envf("ARC_WM_LR", 1e-3)
        self.batch = _envi("ARC_WM_BATCH", 32)
        # Bigger cap + reservoir eviction so the buffer holds a REPRESENTATIVE sample
        # of the whole (multi-game) transition stream, not just the most-recent window.
        # FIFO forgets early games -> can't generalize; reservoir keeps uniform coverage.
        self.buffer_cap = _envi("ARC_WM_BUFFER", 50000)
        self.buffer_policy = os.environ.get("ARC_WM_BUFFER_POLICY", "reservoir").strip().lower()
        self.sig_lambda = _envf("ARC_WM_SIGREG_LAMBDA", 0.1)
        self.sig_M = _envi("ARC_WM_SIGREG_M", 256)
        self.sig_T = _envi("ARC_WM_SIGREG_T", 20)
        # SUPERVISED CONTRASTIVE on the encoder latent (Phase 1b -- the cartographer encoder
        # fix). The frozen ARC encoder maps a filesystem parent and child to near-identical
        # latents, so both the latent gate (49% beats-identity) and the task gate (curious only
        # +24% coverage) are encoder-bound. This pulls directories with the SAME LLM READING
        # (cond purpose) together and pushes different readings apart, on POOLED [n,Dch] latents
        # -- a SupCon loss. Being supervised with negatives it CANNOT collapse to a constant, so
        # it REPLACES SIGReg (whose per-token [T, B*Gh*Gw, M] intermediate is the unfreeze-OOM
        # culprit): run it with ARC_WM_FREEZE_ENCODER=0 ARC_WM_SIGREG_LAMBDA=0 to finally let the
        # encoder separate regions. Label = argmax of the cond's purpose dims (0:13, present in
        # both the 16-d and 32-d conds). Default 0 = OFF -> byte-identical to today.
        self.contrastive_lambda = _envf("ARC_WM_CONTRASTIVE_LAMBDA", 0.0)
        self.contrastive_temp = _envf("ARC_WM_CONTRASTIVE_TEMP", 0.2)
        # MODE (measured, not guessed): full SupCon REGRESSED the task gate -- its POSITIVE term
        # pulls every same-purpose directory onto one point, which destroys the WITHIN-kind
        # diversity the map's coverage metric counts (curious landmarks 18.3 -> 14.0), even as
        # its repulsion half genuinely helped the forward model (predictive 14.7 -> 17.7, from
        # tying the baselines to beating them). So keep the PUSH, drop the PULL:
        #   "repel"  (default) -- only push DIFFERENT-purpose dirs apart past a cosine margin.
        #                         Separates the kinds (the flatness fix) while leaving
        #                         same-kind dirs free to stay distinct (landmarks survive).
        #   "supcon"           -- the full pull+push (kept for comparison / ablation).
        self.contrastive_mode = os.environ.get("ARC_WM_CONTRASTIVE_MODE", "repel").strip().lower()
        self.contrastive_margin = _envf("ARC_WM_CONTRASTIVE_MARGIN", 0.0)
        # The adapter (see _FsAdapter): a trainable predictor-only view of the FROZEN latent, so
        # the contrastive objective sharpens the FORWARD MODEL without flattening the map that
        # curiosity/frontier depends on. Run it WITH ARC_WM_FREEZE_ENCODER=1. Default off.
        self.fs_adapter_on = _envi("ARC_WM_FS_ADAPTER", 0) not in (0,)
        self.fs_adapter = None
        # Predict-then-decode supervision: force the PREDICTED next latent to
        # DECODE to the real next grid, weighted toward transitions that actually
        # changed. This breaks the "identity shortcut" (predicting next≈current
        # ignores the action, which minimises the latent MSE when frames barely
        # move) so the predictor learns what each ACTION actually does.
        self.pred_ce_lambda = _envf("ARC_WM_PRED_CE_LAMBDA", 1.0)
        # Per-CELL weight on cells that actually changed. This has to be LARGE, and 4.0 was
        # not: a typical ARC transition moves ~32 of 4096 cells (0.8%), so at weight 5 the
        # moving cells carry only 0.8%*5 = 4% of the cross-entropy mass while the static
        # 99.2% carries the rest -- identity still wins. Equal mass needs 1+w ~= 1/0.008,
        # i.e. w ~= 125. At 100 the changed cells carry ~45% of the loss, which is what it
        # takes to actually pay the predictor to move pixels rather than merely move the
        # latent in some direction the decoder never looks at.
        self.change_weight = _envf("ARC_WM_CHANGE_WEIGHT", 100.0)

        # -- ACTION SENSITIVITY: identification, not repulsion ---------------------
        # The predictor used to collapse to identity: it emitted the SAME next-state for
        # every action. Three REPULSION mechanisms were tried against that and all three
        # were gamed, because repulsion only ever demands "be different", which is
        # satisfiable in any direction nothing else looks at:
        #   1. latent margin 0.5      -> hinge permanently saturated (true inter-action
        #                                distance is 0.0022, so the margin was 225x too big);
        #   2. latent margin rescaled -> contrast drove to 0.0 (satisfied!) while every
        #                                action STILL decoded to an identical grid: the
        #                                predictor separated latents inside the DECODER'S
        #                                NULL SPACE;
        #   3. decoded-space contrast -> gamed via the soft/argmax gap (soft distributions
        #                                differ enough to clear the hinge; the argmax grid
        #                                does not move).
        # The literature converges on the opposite mechanism (Delta-JEPA 2606.31232;
        # Sensorimotor World Models 2606.20104, which beats SIGReg 84% vs 59%;
        # ACID 2607.02403 -- all three published as improvements over LeWM, i.e. over THIS
        # model): INVERSE DYNAMICS. Instead of pushing actions' predictions apart, require
        # that the action be RECOVERABLE from the latent displacement. That is a supervised
        # target with a unique correct answer, so it cannot be satisfied in a null space --
        # there is no direction in which you are *correctly* decoded as action 3.
        self.inv_lambda = _envf("ARC_WM_INV_LAMBDA", 1.0)
        # Recovering WHERE you clicked from the displacement is DEFAULT OFF, because it is
        # mostly not recoverable -- and that is a fact about ARC, not a modelling failure.
        # Measured over 1,432 random clicks that actually changed the board: only 9.1%
        # changed a cell inside the click's own latent cell, and only 11.3% changed
        # anything within 8 board cells of it. In ARC a click SELECTS something and the
        # effect lands elsewhere. So this target is ~91% label noise, and it was polluting
        # the IDM's shared trunk for nothing (click_acc sat at exactly 0.000 throughout).
        # Note this does NOT weaken coord conditioning: telling the predictor where the
        # click WAS (ARC_WM_COORD_PLANES) is valuable and stays on. Asking it to infer the
        # click from the aftermath is what does not work. Turn on only for a game whose
        # clicks are genuinely local.
        self.inv_click_lambda = _envf("ARC_WM_INV_CLICK_LAMBDA", 0.0)
        self.inv_hidden = _envi("ARC_WM_INV_HIDDEN", 64)
        # DETACHED TRUST HEAD (serving mirror): build + restore the IDM so verify()/dream
        # confidence work when the loaded ckpt carries a head, even at inv_lambda=0. The
        # head is trained offline (frozen predictor); the service only builds+restores+
        # reads it. See agents/lewm.py for the training path.
        self.inv_detached = _envi("ARC_WM_INV_DETACHED", 0) not in (0,)
        # Gradient scale from the real-transition IDM branch back into the ENCODER. The
        # encoder currently reconstructs at 0.94 cell-accuracy -- a banked win. This is the
        # safety valve: set 0.0 to confine the IDM to the head + predictor and leave the
        # encoder's geometry untouched.
        self.inv_enc_scale = _envf("ARC_WM_INV_ENC_SCALE", 1.0)
        # FREEZE the encoder (and decoder) and train ONLY the action-conditioned parts.
        # This is V-JEPA 2-AC's actual recipe: pretrain the representation, then freeze it
        # and fit an action-conditioned predictor on top ("the encoder is frozen ... no
        # EMA"). It matters here because our encoder/decoder are ALREADY good (0.94 cell
        # reconstruction, hard-won) while the predictor is the broken half -- and letting
        # the new IDM gradient reshape the encoder measurably degraded reconstruction
        # (recon 0.11 -> 0.32 within 500 steps). Freezing makes the banked win
        # structurally safe instead of merely monitored.
        self.freeze_encoder = _envi("ARC_WM_FREEZE_ENCODER", 0) not in (0,)
        # Rollout (V-JEPA 2-AC): teacher forcing alone never trains the multi-step dreams
        # we actually ship, so errors compound at inference. 0 disables.
        self.roll_lambda = _envf("ARC_WM_ROLLOUT_LAMBDA", 0.5)

        # Predictor dropout (LeWM Tab. 9: p=0.1 was their single biggest predictor lever,
        # +18% SR; 0.0/0.5 both worse). Channel-dropout on each resblock's residual branch.
        # Parameter-free, so adding it never breaks state_dict restore of an old predictor.
        self.pred_dropout = _envf("ARC_WM_PRED_DROPOUT", 0.0)

        # -- click (x,y) conditioning ----------------------------------------------
        # ACTION6 carries a 4096-way coordinate that the model has never once been shown:
        # observe() stored a bare action index, so a click at (3,7) and a click at (60,60)
        # were literally the same input vector. Feed the coordinate in as spatial planes.
        self.coord_planes = _envi("ARC_WM_COORD_PLANES", 2)   # 0 = off (old behaviour)
        self.coord_sigma = _envf("ARC_WM_COORD_SIGMA", 1.0)   # in LATENT cells

        # -- game conditioning ------------------------------------------------------
        # MEASURED, not assumed: a CNN probe recovers the action from (before, after) at
        # 0.895 balanced accuracy WITHIN one game (chance 0.25), but sits AT CHANCE when
        # all games are pooled. ARC action semantics are game-specific -- ACTION3 means
        # something different in every game -- so a predictor with no game input is being
        # asked to fit contradictory dynamics, for which "predict nothing changes" is the
        # correct average. ctx_dim defaults ON now (it was plumbed but dead at 0).
        self.n_game_slots = _envi("ARC_WM_GAME_SLOTS", 64)    # slot 0 = UNKNOWN
        self.game_key = os.environ.get("ARC_WM_GAME_KEY", "prefix").strip().lower()
        self._game_slots: dict = {}

        # -- no-op rebalancing ------------------------------------------------------
        # ~29-42% of transitions change ZERO cells (many ARC clicks land on nothing). At
        # the natural rate, a large slice of the gradient explicitly rewards identity.
        # Do NOT drive this to 0: predicting a genuine no-op IS real dynamics.
        self.noop_frac = _envf("ARC_WM_NOOP_FRAC", 0.15)      # <0 = natural rate
        self.mse_change_weight = _envf("ARC_WM_MSE_CHANGE_WEIGHT", 4.0)   # 0 = plain mean

        # -- forward-predictor fix (default OFF -> byte-identical to today) -------------
        # DIAGNOSIS (measured 2026-07-13): the forward predictor beats the identity baseline
        # ("predict nothing changes") on 0/400 in-distribution transitions -- median latent
        # skill -12.6 -- while a kNN oracle over the SAME frozen encoder gets median +0.20.
        # The representation carries the signal (inv_acc 0.63); the forward map throws it
        # away. Root cause: the on-axis latent-MSE (pulls z_hat -> z_tp1) is a single,
        # renormalized, NON-detached term, out-voted ~2:1 by two weight-1.0 unique-answer
        # losses (pred_ce, the inv cycle branch) that z_hat can satisfy by moving OFF the
        # z_tp1 axis. And z_tp1 is the encoder's own live (non-detached) output, so MSE can
        # be cut by collapsing successive latents instead of learning dynamics. These knobs
        # de-game the target and put the gradient mass back on the on-axis term.
        #   target_detach:     stop-grad the MSE target (z_tp1) so it can't be gamed by the
        #                      encoder. Lighter than FREEZE_ENCODER; use when the encoder
        #                      must still move (kNN ceiling <=0 frozen). See wm_knn_ceiling.
        #   mse_lambda:        explicit weight on the on-axis latent-MSE (was an implicit 1.0
        #                      competing with pred_ce=1.0 + inv=1.0).
        #   identity_relative: pay the MSE only PAST the identity baseline -- stops rewarding
        #                      already-won samples and makes the trained loss == the gate's
        #                      skill numerator. A floor, not a direction change (see gate).
        self.target_detach     = _envi("ARC_WM_TARGET_DETACH", 0) not in (0,)
        self.mse_lambda        = _envf("ARC_WM_MSE_LAMBDA", 1.0)
        self.identity_relative = _envi("ARC_WM_IDENTITY_RELATIVE", 0) not in (0,)
        self.idrel_margin      = _envf("ARC_WM_IDREL_MARGIN", 0.0)
        # -- Stage B: LLM changed-region mask denoises the per-cell pred_ce weight ONLY.
        # No latent target, no FiLM, no D_c -- invisible to the gate's cond-free encode path.
        self.llm_mask          = _envi("ARC_WM_LLM_MASK", 0) not in (0,)

        # -- C-JEPA object-level latent masking (default OFF) -----------------------
        # arXiv 2602.11389: mask a fraction of the grid's OBJECTS (4-connected
        # components of non-background cells) out of the PREDICTOR INPUT ONLY,
        # replacing their latent cells with a learned mask token. The identity
        # anchor is the point: the target z_tp1, the identity baseline d_id,
        # SIGReg and the IDM all keep the INTACT z_t — masking corrupts only
        # what the predictor sees, so it must model object dynamics it cannot
        # copy through. Spatial mode + train_step only; every serving path
        # (/encode /predict /plan /dream) stays mask-free. Value = fraction of
        # a sample's objects to mask (min 1 when >0). Promotion strictly via
        # wm_latent_gate --recordings A/B, never --buffer.
        self.obj_mask_frac = _envf("ARC_WM_OBJECT_MASK", 0.0)

        # -- Part A: VALUE head (default OFF -> byte-identical). V(z) predicts the discounted
        # future EXTRINSIC return (avoid GAME_OVER, seek level-ups). Trained on a DETACHED
        # frozen latent, so it never reshapes the encoder. Intrinsic novelty stays in the
        # planner (plan_curious) -- clean split: extrinsic value + intrinsic exploration.
        self.value_lambda   = _envf("ARC_WM_VALUE_LAMBDA", 0.0)  # weight in train_step (online)
        self.value_head_on  = _envi("ARC_WM_VALUE_HEAD", 0) not in (0,)  # build head, no loss
        self.value_hidden   = _envi("ARC_WM_VALUE_HIDDEN", 64)
        self.gamma          = _envf("ARC_WM_GAMMA", 0.99)         # return discount
        self.nstep          = _envi("ARC_WM_NSTEP", 8)            # n-step return horizon
        self.death_penalty  = _envf("ARC_WM_DEATH_PENALTY", 1.0)  # reward at a GAME_OVER terminal
        self.value_terminal_w = _envf("ARC_WM_VALUE_TERMINAL_W", 0.0)  # plan terminal-V weight

        # -- LLM-REASONING as the predictor's ctx (the "second vector") -----------------
        # The bare grid cannot say WHICH child you descended into (its contents are not a
        # function of the parent view). Feed the CHILD's LLM reasoning (from its NAME, known
        # before opening it) as the predictor ctx, so predict(z_parent, click, child_reasoning)
        # can imagine the child. Requires D_c==ctx_dim (the reasoning cond IS the ctx). Default
        # OFF -> the predictor ctx stays the game embedding (byte-identical).
        self.reason_ctx = _envi("ARC_WM_REASON_CTX", 0) not in (0,)

        # CEM config
        self.cem_samples = _envi("ARC_WM_CEM_SAMPLES", 256)
        self.cem_elites = _envi("ARC_WM_CEM_ELITES", 32)
        self.cem_horizon = _envi("ARC_WM_CEM_HORIZON", 5)
        self.cem_iters = _envi("ARC_WM_CEM_ITERS", 5)

        # Buffer item is a 6-tuple: (ids_t uint8, a_idx, ids_tp1 uint8, game,
        # cond_t [D_c] f16 | None, cond_tp1 [D_c] f16 | None). cond_* are the
        # fused Sensorium vectors at t / t+1; None when the aux channel is off,
        # which keeps the buffer identical to the old grid-only 4-tuple stream.
        self._buf: List[tuple] = []
        self._total_seen = 0         # lifetime transitions observed (drives reservoir prob)
        self._games: dict = {}       # game_id -> count, for diversity readout
        # Transitions REFUSED, by reason. These used to be silently relabelled ACTION1.
        # Surfaced on /health so a mislabelling regression is loud, not invisible.
        self._drops: dict = {"unparseable": 0, "reset": 0, "coerce": 0}
        self._ep_seq: dict = {}      # game -> current episode id (bumped on RESET)
        self._t_seq: dict = {}       # game -> step index within the current episode
        self._succ: dict = {}        # (game, ep, t) -> buffer slot, for the rollout chain
        # cached changed/no-op partition for the stratified sampler (see _sample_batch)
        self._cls_chg: Optional[list] = None
        self._cls_noop: Optional[list] = None
        self._cls_dirty = True
        self._step = 0
        self._init_loss: Optional[float] = None
        # Running mean of encoded latents (EMA) -> the /probe novelty / frontier
        # signal for self-factualization: ||z - z_ema|| = how novel is here.
        self._z_ema = None
        self._z_ema_beta = _envf("ARC_WM_ZEMA_BETA", 0.99)

        if not self.ok:
            logger.warning("lewm: torch unavailable; LeWorldModel disabled (training deferred)")
            self.encoder = None
            self.predictor = None
            self.film = None
            self._opt = None
            self._sig_dirs = None
            self._sig_t = None
            self.device = None
            return

        self.device = _pick_device(device)
        if seed is not None:
            torch.manual_seed(int(seed))
        self._build_modules()
        self._sig_t = torch.linspace(0.01, 2.0, steps=self.sig_T, device=self.device)  # [T]

    def _build_modules(self) -> None:
        """Construct encoder/predictor/decoder (+ FiLM aux head) and their
        optimizers + the fixed SIGReg basis, from the current config. Spatial or
        flat per self.spatial. Reused by __init__ and load()'s rebuild path."""
        self.idm = None
        self.value_head = None
        self.game_emb = None
        self.fs_adapter = None
        if self.spatial:
            self.encoder = _SpatialEncoder(self.G, self.C, self.Cemb, self.Dch).to(self.device)
            if self.fs_adapter_on:
                self.fs_adapter = _FsAdapter(self.Dch).to(self.device)
            self.predictor = _SpatialPredictor(
                self.Dch, self.Gh, self.A, self.ctx_dim,
                coord_planes=self.coord_planes, dropout=self.pred_dropout).to(self.device)
            # INVARIANT: the predictor lives in eval() (dropout off) everywhere EXCEPT inside
            # train_step, which flips it to train() and back. nn defaults to train() and
            # load_state_dict doesn't touch mode, so without this a dropout>0 predictor would
            # drop during the gate/planning. A no-op when pred_dropout==0 (no dropout modules).
            self.predictor.eval()
            if self.obj_mask_frac > 0:
                # C-JEPA mask token, registered ON the predictor BEFORE the
                # params list below is collected, so it rides
                # predictor.parameters() into the optimizer and the module's
                # state_dict. load() is strict=False in both directions, so an
                # old checkpoint loads with the flag on (token stays fresh) and
                # a masked checkpoint loads with it off (extra key ignored).
                self.predictor.obj_mask_token = nn.Parameter(
                    torch.zeros(self.Dch, device=self.device))
            self.decoder = _SpatialDecoder(self.Dch, self.C, self.G).to(self.device)
            if self.inv_lambda > 0 or self.inv_detached:
                self.idm = _InverseDynamics(self.Dch, self.Gh, self.A,
                                            self.inv_hidden).to(self.device)
            # Part A value head -- built whenever a value or reward lambda is on, OR when
            # explicitly requested (so a value-only offline run / the /value endpoint work
            # even with the training lambdas off). Spatial only (mirrors the IDM).
            if self.value_lambda > 0 or self.value_head_on:
                self.value_head = _ValueHead(self.Dch, self.Gh, self.value_hidden).to(self.device)
        else:
            self.encoder = _Encoder(self.G, self.C, self.Cemb, self.D).to(self.device)
            self.predictor = _Predictor(self.D, self.A, self.ctx_dim).to(self.device)
            self.decoder = _Decoder(self.D, self.C, self.G).to(self.device)
        # Game embedding: slot 0 is reserved UNKNOWN, so an unseen game degrades to a
        # neutral vector instead of borrowing another game's dynamics.
        if self.ctx_dim > 0:
            self.game_emb = nn.Embedding(self.n_game_slots, self.ctx_dim).to(self.device)
            nn.init.normal_(self.game_emb.weight, std=0.02)
        # FiLM aux-fusion head only exists when the aux channel is enabled.
        self.film = _CondFiLM(self.D_c, self.D).to(self.device) if self.D_c > 0 else None
        params = list(self.predictor.parameters())
        if not self.freeze_encoder:
            params += list(self.encoder.parameters())
        else:
            for p_ in self.encoder.parameters():
                p_.requires_grad_(False)
            # 2026-07-26: freezing the ENCODER (the V-JEPA 2-AC recipe) used to freeze
            # the DECODER with it. That conflated two different things. The encoder is
            # the representation and must stay fixed. The decoder is only a READOUT --
            # it is trained against z_t.detach() with its OWN optimizer, so it provably
            # cannot perturb the JEPA weights (the comment below has always said so).
            # Freezing it meant the pixel decoder could never learn a NEW game's
            # palette: measured this day, 10 of 12 recorded games reconstruct at
            # 93-99%, while game lf52 sits at 73.4% (vs a 63.1% guess-the-background
            # baseline) and invents 5 colors that are not on the board. That is what
            # makes the live "Aither's imagination" panel render the right silhouette
            # filled with RGB hash. Keep the decoder learning so it adapts.
            # ARC_WM_FREEZE_DECODER=1 restores the old both-frozen behaviour.
            self.freeze_decoder = _envi("ARC_WM_FREEZE_DECODER", 0) not in (0,)
            if self.freeze_decoder:
                for p_ in self.decoder.parameters():
                    p_.requires_grad_(False)
                logger.warning("lewm: encoder AND decoder FROZEN (ARC_WM_FREEZE_DECODER=1) "
                               "-- reconstruction cannot regress, but it also cannot adapt "
                               "to an unseen game's palette.")
            else:
                logger.warning("lewm: encoder FROZEN (V-JEPA 2-AC style); pixel decoder "
                               "still co-training on detached z so the imagination can "
                               "learn new games' palettes without touching JEPA.")
        if self.film is not None:
            params += list(self.film.parameters())
        if self.idm is not None and not self.inv_detached:
            params += list(self.idm.parameters())   # detached head is offline-trained, static at serve
        if self.value_head is not None:
            params += list(self.value_head.parameters())   # trains regardless of encoder-freeze
        if self.game_emb is not None:
            params += list(self.game_emb.parameters())
        if self.fs_adapter is not None:
            params += list(self.fs_adapter.parameters())   # trains even with the trunk frozen
        self._opt = torch.optim.Adam(params, lr=self.lr)
        # Detached trust head trains on its OWN optimizer inside train_step, so no
        # gradient can reach predictor/encoder through the champion loss (anti-collapse
        # by construction). None when the head is absent or jointly trained.
        self._idm_opt = (torch.optim.Adam(self.idm.parameters(), lr=self.lr)
                         if (self.idm is not None and self.inv_detached) else None)
        # Pixel decoder has its OWN optimizer, trained against a detached z so it
        # can never perturb the JEPA weights.
        # Only a DELIBERATE decoder freeze disables it now (see above) -- a frozen
        # encoder alone must not switch off the readout head's training.
        self._dec_opt = (None if getattr(self, "freeze_decoder", False)
                         else torch.optim.Adam(self.decoder.parameters(), lr=self.lr))
        # Fixed random unit-norm projection directions for SIGReg (no learned
        # params). Feature dim is per-token (Dch) when spatial, else flat D.
        dirs = torch.randn(self._sig_feat_dim, self.sig_M, device=self.device)
        self._sig_dirs = F.normalize(dirs, dim=0)              # [feat, M]

    # -- action / grid coercion -------------------------------------------------

    def _parse(self, action: Any) -> Optional[ParsedAction]:
        """Model-side wrapper over the module-level `parse_action` (which is THE one
        parser -- cns_publisher, arc_live_server and wm_eval_harness import it too, so
        the silent-0 bug cannot be reintroduced in a copy)."""
        return parse_action(action, self.A)

    def _coord_map(self, ax, ay):
        """(x, y) click coords -> [B, coord_planes, Gh, Gw] conditioning planes, or None.

        Plane 0: a Gaussian bump at the click's LATENT cell. The predictor is a conv
                 stack with a bounded receptive field, so telling it WHERE the click
                 landed lets an action's effect be local to the click by construction.
        Plane 1: a "this action HAS coordinates" indicator, so a click whose bump falls
                 between latent cells is distinguishable from an action with no click
                 at all (where both planes are exactly zero -- i.e. bit-identical to the
                 old coord-free model).

        Measured on real play: cells changed by a click are 5.2x enriched within 8 board
        cells of the click vs a random click position -- a real but PARTIAL locality. So
        this is conditioning, not a hard constraint: the residual conv path can still
        move distant cells when the game says so.
        """
        if not self.spatial or self.coord_planes <= 0 or ax is None:
            return None
        B = ax.shape[0]
        out = torch.zeros(B, self.coord_planes, self.Gh, self.Gw, device=self.device)
        has = (ax >= 0) & (ay >= 0)
        if not bool(has.any()):
            return out
        # board cell -> continuous latent-cell centre (G/Gh = 4), so a click at (19,16)
        # splits its mass across neighbouring latent cells instead of snapping.
        s = self.G / float(self.Gh)
        cx = (ax.float() + 0.5) / s - 0.5      # [B]
        cy = (ay.float() + 0.5) / s - 0.5
        gy = torch.arange(self.Gh, device=self.device).view(1, -1, 1).float()
        gx = torch.arange(self.Gw, device=self.device).view(1, 1, -1).float()
        d2 = (gy - cy.view(-1, 1, 1)) ** 2 + (gx - cx.view(-1, 1, 1)) ** 2   # [B,Gh,Gw]
        bump = torch.exp(-d2 / (2.0 * self.coord_sigma ** 2))
        out[:, 0] = bump * has.float().view(-1, 1, 1)
        if self.coord_planes > 1:
            out[:, 1] = has.float().view(-1, 1, 1)
        return out

    def _to_ids(self, grid: Any):
        """Coerce an arbitrary ARC grid to a [G, G] long tensor of colour ids
        (top-left pad/crop to fixed size, colours clamped to [0, C))."""
        if not self.ok:
            return None
        arr = np.asarray(grid)
        # ARC-AGI-3 frames are sometimes 3D (stack of layers) or nested; take a 2D slice.
        while arr.ndim > 2:
            arr = arr[-1]
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        arr = np.clip(np.nan_to_num(arr.astype(np.int64), nan=0), 0, self.C - 1)
        out = np.zeros((self.G, self.G), dtype=np.int64)
        h = min(arr.shape[0], self.G)
        w = min(arr.shape[1], self.G)
        out[:h, :w] = arr[:h, :w]
        return torch.from_numpy(out).to(self.device)

    def _onehot(self, idxs):
        return F.one_hot(idxs.long(), num_classes=self.A).float()

    def _game_norm(self, game: Any) -> Optional[str]:
        """ARC ids look like `ls20-9607627b`, where the suffix is a per-instance card id.
        Key on the FAMILY prefix by default, or 64 embedding slots evaporate on card
        churn and the game embedding never generalises across cards of the same game."""
        if game is None:
            return None
        g = str(game)
        return g.split("-")[0] if self.game_key == "prefix" else g

    def _game_slot(self, game: Any, assign: bool = False) -> int:
        """game -> embedding slot. 0 = UNKNOWN (unseen game, or the table is full)."""
        g = self._game_norm(game)
        if g is None:
            return 0
        s = self._game_slots.get(g)
        if s is not None:
            return s
        if not assign or len(self._game_slots) + 1 >= self.n_game_slots:
            return 0
        s = len(self._game_slots) + 1        # slot 0 stays reserved for UNKNOWN
        self._game_slots[g] = s
        return s

    def _game_ctx(self, games: Sequence[Any]):
        """[game, ...] -> [B, ctx_dim] embedding, or None when game conditioning is off."""
        if self.game_emb is None:
            return None
        slots = torch.tensor([self._game_slot(g) for g in games],
                             dtype=torch.long, device=self.device)
        return self.game_emb(slots)

    def _coerce_vec(self, v: Any, dim: int):
        """Coerce an arbitrary aux/ctx vector to a [dim] float tensor on device,
        pad/cropped to `dim`, NaN/Inf-scrubbed. None when disabled / dim<=0 / no
        vector -- so a caller sending a slightly-off width never raises into a turn."""
        if not self.ok or dim <= 0 or v is None:
            return None
        try:
            arr = np.asarray(v, dtype=np.float32).flatten()
        except Exception:
            return None
        out = np.zeros((dim,), dtype=np.float32)
        k = min(arr.shape[0], dim)
        out[:k] = np.nan_to_num(arr[:k], nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(out).to(self.device)

    def _to_cond(self, cond: Any):
        return self._coerce_vec(cond, self.D_c)

    def _to_ctx(self, ctx: Any):
        return self._coerce_vec(ctx, self.ctx_dim)

    def _encode_ids(self, ids, cond=None):
        """encoder(ids) then optional FiLM aux-fusion. ids:[B,G,G] cond:[B,D_c]|None.
        When film is off or cond is None this is exactly encoder(ids)."""
        z = self.encoder(ids)
        if self.film is not None and cond is not None:
            z = self.film(z, cond)
        return z

    def _object_masks(self, ids):
        """Per-sample C-JEPA object masks. ids:[n,G,G] long -> [n,1,Gh,Gw] float
        (1 = latent cell overlapping a masked object).

        Objects are 4-connected components of non-background cells, background
        being each sample's most frequent colour (ARC boards are dominated by
        their background; a fixed colour 0 would mis-segment recoloured games).
        ceil(obj_mask_frac * K) of a sample's K objects are chosen via
        torch.randperm (respects torch.manual_seed, unlike np.random), and
        their cells are MAX-pooled down to the latent grid, so any overlap
        masks the whole latent cell. Pure-python BFS over ~n*G*G cells --
        milliseconds at batch 16, and no scipy dependency in the image."""
        n, gsz = int(ids.shape[0]), int(ids.shape[1])
        f = max(1, gsz // self.Gh)
        out = torch.zeros(n, 1, self.Gh, self.Gw, device=ids.device)
        arr = ids.detach().cpu().numpy()
        for s in range(n):
            g = arr[s]
            vals, counts = np.unique(g, return_counts=True)
            bg = vals[counts.argmax()]
            fg = g != bg
            if not fg.any():
                continue
            labels = np.zeros((gsz, gsz), dtype=np.int32)
            k = 0
            for i0 in range(gsz):
                for j0 in range(gsz):
                    if fg[i0, j0] and labels[i0, j0] == 0:
                        k += 1
                        labels[i0, j0] = k
                        stack = [(i0, j0)]
                        while stack:
                            i, j = stack.pop()
                            for ii, jj in ((i + 1, j), (i - 1, j),
                                           (i, j + 1), (i, j - 1)):
                                if (0 <= ii < gsz and 0 <= jj < gsz
                                        and fg[ii, jj] and labels[ii, jj] == 0):
                                    labels[ii, jj] = k
                                    stack.append((ii, jj))
            n_mask = max(1, min(k, int(np.ceil(self.obj_mask_frac * k))))
            chosen = (1 + torch.randperm(k)[:n_mask]).numpy()
            cell = torch.from_numpy(
                np.isin(labels, chosen).astype(np.float32))          # [G,G]
            pooled = F.max_pool2d(cell.view(1, 1, gsz, gsz), kernel_size=f)
            out[s] = pooled.squeeze(0).to(ids.device)
        return out

    def _apply_object_mask(self, z_flat, ids):
        """Replace masked latent cells of z (flat [n,L]) with the learned mask
        token. Returns a NEW tensor -- the caller's z stays intact, which is the
        C-JEPA identity anchor (d_id / SIGReg / IDM must see the real z_t)."""
        mask = self._object_masks(ids)                                # [n,1,Gh,Gw]
        if not bool(mask.any()):
            return z_flat
        n = z_flat.shape[0]
        zm = z_flat.view(n, self.Dch, self.Gh, self.Gw)
        tok = self.predictor.obj_mask_token.view(1, self.Dch, 1, 1)
        return (zm * (1.0 - mask) + tok * mask).flatten(1)

    # -- public API -------------------------------------------------------------

    def observe(self, grid: Any, action: Any, next_grid: Any, game: Any = None,
                cond: Any = None, next_cond: Any = None, effect_mask: Any = None,
                reward: float = 0.0, done: bool = False, legal: Any = None) -> str:
        """Buffer one (grid_t, action_t, grid_{t+1}) transition. Returns a status string
        ("ok" | "drop:unparseable" | "drop:reset" | "drop:coerce" | "disabled") -- the
        caller may ignore it, but /health counts the drops.

        No-op when disabled. Grids are stored as compact CPU uint8 id-tensors (colours
        0..C-1 fit in a byte -> 8x smaller than long, so a 50k buffer is cheap). Optional
        `game` tags the transition -- it now also KEYS THE GAME EMBEDDING, not just the
        diversity readout. Optional `cond`/`next_cond` are the fused Sensorium aux vectors
        at t / t+1 (stored compact f16); None when the aux channel is off -> identical to
        the old grid-only stream.

        An action that cannot be parsed is DROPPED, never coerced to index 0. That
        coercion is what destroyed this model's action conditioning.

        Eviction (once the buffer is full) is RESERVOIR sampling by default: each
        new transition replaces a uniformly-random slot with probability cap/seen,
        so the retained set stays a representative sample of the ENTIRE stream
        across every game played -- the property FIFO destroys (it keeps only the
        recent window, so a model that rotates games forgets the earlier ones and
        never accumulates the diverse dataset generalization needs)."""
        if not self.ok:
            return "disabled"
        g = None if game is None else str(game)

        # PARSE FIRST, and DROP anything we cannot label. The old code coerced every
        # unparseable action to index 0 -- which is also ACTION1's index -- so clicks
        # (coords discarded), RESETs and first-turn Nones all piled into one bucket that
        # became 82% of the buffer. A dropped transition costs one sample; a mislabelled
        # one poisons the conditional distribution the whole model is fitted to.
        p = self._parse(action)
        if p is None:
            self._drops["unparseable"] += 1
            if self._drops["unparseable"] in (1, 10, 100, 1000):
                logger.warning("lewm.observe: DROPPED unparseable action %r (%d so far) "
                               "-- it is NOT being silently labelled ACTION1",
                               action, self._drops["unparseable"])
            return "drop:unparseable"
        if p.reset:
            # A reset is a board DISCONTINUITY, not a dynamics transition: grid_t is the
            # end of one level and grid_t+1 is a fresh one. Training on it teaches the
            # model that some action teleports the board. It also opens a new episode.
            self._drops["reset"] += 1
            if g is not None:
                self._ep_seq[g] = self._ep_seq.get(g, 0) + 1
                self._t_seq[g] = 0
            return "drop:reset"

        try:
            ids_t = self._to_ids(grid).to(torch.uint8).cpu()
            ids_tp1 = self._to_ids(next_grid).to(torch.uint8).cpu()
        except Exception as e:  # never raise into the caller
            logger.debug("lewm.observe coercion failed: %s", e)
            self._drops["coerce"] += 1
            return "drop:coerce"

        ax = int(min(max(p.x, -1), self.G - 1))
        ay = int(min(max(p.y, -1), self.G - 1))
        if g is not None:
            self._game_slot(g, assign=True)          # register for the game embedding
            ep = self._ep_seq.setdefault(g, 0)
            t = self._t_seq.get(g, 0)
            self._t_seq[g] = t + 1
        else:
            ep = t = -1
        chg = bool((ids_t != ids_tp1).any().item())
        ct = self._to_cond(cond)
        ctp = self._to_cond(next_cond)
        ct = ct.to(torch.float16).cpu() if ct is not None else None
        ctp = ctp.to(torch.float16).cpu() if ctp is not None else None
        em = None
        if effect_mask is not None:
            try:
                em = torch.as_tensor(np.asarray(effect_mask), dtype=torch.uint8).cpu()
                if tuple(em.shape) != (self.G, self.G):
                    em = None                 # wrong shape -> fall back to raw pixel-diff
            except Exception:
                em = None
        try:
            r_val = float(reward)
        except Exception:
            r_val = 0.0
        item = _Tr(ids_t, p.idx, ids_tp1, g, ct, ctp, ax, ay, ep, t, chg, em,
                   r_val, bool(done), legal)
        self._total_seen += 1
        if g is not None:
            self._games[g] = self._games.get(g, 0) + 1
        if len(self._buf) < self.buffer_cap:
            self._buf.append(item)
            self._link(len(self._buf) - 1)
        elif self.buffer_policy == "fifo":
            self._buf.append(item)
            self._buf = self._buf[-self.buffer_cap:]
            self._reindex_succ()          # every slot shifted
        else:
            # reservoir (Algorithm R): keep a uniform sample of all _total_seen items
            j = int(np.random.randint(0, self._total_seen))
            if j < self.buffer_cap:
                self._unlink(j)           # the evicted item's chain key must go with it
                self._buf[j] = item
                self._link(j)
        return "ok"

    def _link(self, i: int) -> None:
        self._cls_dirty = True          # the changed/no-op partition just moved
        b = self._buf[i]
        if len(b) > 10 and b.game is not None and b.ep >= 0:
            self._succ[(b.game, b.ep, b.t)] = i

    def _unlink(self, i: int) -> None:
        b = self._buf[i]
        if len(b) > 10 and b.game is not None and b.ep >= 0:
            self._succ.pop((b.game, b.ep, b.t), None)

    def encode(self, grid: Any, cond: Any = None):
        """grid (+ optional aux `cond`) -> latent z ([D] tensor), under no_grad.
        None when disabled. cond is fused via FiLM when the aux channel is on."""
        if not self.ok:
            return None
        try:
            ids = self._to_ids(grid).unsqueeze(0)  # [1, G, G]
        except Exception:
            return None
        c = self._to_cond(cond)
        cb = c.unsqueeze(0) if c is not None else None
        with torch.no_grad():
            return self._encode_ids(ids, cb).squeeze(0)    # [D]

    def _adapt(self, z):
        """The PREDICTOR's view of the latent: z -> fs_adapter(z) when the adapter is on, else
        z unchanged. Applied on the predictor's INPUT only -- encode()/MapMemory/curiosity keep
        the raw (frozen) latent, whose geometry is what makes the frontier signal work."""
        if self.fs_adapter is None or not self.spatial:
            return z
        bsz = z.shape[0]
        return self.fs_adapter(z.view(bsz, self.Dch, self.Gh, self.Gw)).reshape(bsz, -1)

    def predict(self, z, action: Any, ctx: Any = None, game: Any = None):
        """(z, action[, ctx][, game]) -> z_hat_next, under no_grad. z may be [D] or [N, D].

        `action` may be an index/label (broadcast) or a batch tensor of indices. The COORDS
        ride inside the action value, so callers need no new argument: "ACTION6(19,16)",
        ("ACTION6", 19, 16) and {"id": 6, "x": 19, "y": 16} all condition the predictor on
        WHERE the click landed. A bare "ACTION6" is coord-free (both planes zero).

        `game` selects the game embedding -- an action means different things in different
        games, so without it the predictor is being asked for an average over contradictory
        dynamics. Unknown/None -> the reserved UNKNOWN slot.

        Returns None for an unparseable action (previously: silently predicted ACTION1)."""
        if not self.ok or z is None:
            return None
        single = (z.dim() == 1)
        zb = z.unsqueeze(0) if single else z
        n = zb.shape[0]
        ax = ay = None
        if isinstance(action, torch.Tensor) and action.dim() >= 1 and action.numel() == n:
            idxs = action.long().to(self.device)          # batch of indices (CEM/plan path)
        else:
            p = self._parse(action)
            if p is None or p.reset:
                return None
            idxs = torch.full((n,), p.idx, dtype=torch.long, device=self.device)
            ax = torch.full((n,), p.x, dtype=torch.long, device=self.device)
            ay = torch.full((n,), p.y, dtype=torch.long, device=self.device)
        cb = None
        cvec = self._to_ctx(ctx)
        if cvec is not None:
            cb = cvec.unsqueeze(0).expand(n, -1)
        elif self.game_emb is not None:
            cb = self._game_ctx([game] * n)
        with torch.no_grad():
            out = self.predictor(self._adapt(zb), self._onehot(idxs), cb,
                                 coord=self._coord_map(ax, ay))
        return out.squeeze(0) if single else out

    def decode(self, z) -> Optional[list]:
        """latent z -> a GxG grid of colour ids (the model's rendered mental
        picture of that latent), under no_grad. z may be a python list [D], a
        [D] tensor, or a [N, D] batch. Returns a GxG list of ints (single) or a
        list of such grids (batch). None when disabled."""
        if not self.ok or z is None:
            return None
        try:
            if not isinstance(z, torch.Tensor):
                z = torch.tensor(z, dtype=torch.float32, device=self.device)
            z = z.to(self.device).float()
            single = (z.dim() == 1)
            zb = z.unsqueeze(0) if single else z
            with torch.no_grad():
                ids = self.decoder(zb).argmax(dim=1)   # [N, G, G]
            grids = ids.detach().cpu().tolist()
            return grids[0] if single else grids
        except Exception as e:
            logger.debug("lewm.decode failed: %s", e)
            return None

    def verify(self, grid, action: Any, game: Any = None) -> Optional[float]:
        """ACID (2607.02403): how much do we believe our own prediction?

        Predict the next latent under `action`, then ask the inverse-dynamics head to name
        the action from the resulting displacement. Returns the head's probability for the
        action we actually conditioned on, in [0, 1]; chance is 1/A ~= 0.14.

        A low score means the predicted transition is one that NO action plausibly
        produces -- the model has imagined something unrealizable. That is precisely the
        case where a world model is most dangerous to a planner, because the frame still
        LOOKS like a board. Callers should suppress the foresight row rather than hand the
        LLM a confident fabrication. Returns None when the IDM is off."""
        if not self.ok or self.idm is None or grid is None:
            return None
        try:
            p = self._parse(action)
            if p is None or p.reset:
                return None
            z = self.encode(grid)
            if z is None:
                return None
            zb = z.unsqueeze(0)
            with torch.no_grad():
                z_hat = self.predictor(
                    zb, self._onehot(torch.tensor([p.idx], device=self.device)),
                    self._game_ctx([game]),
                    coord=self._coord_map(
                        torch.tensor([p.x], device=self.device),
                        torch.tensor([p.y], device=self.device)))
                logits, _ = self.idm(z_hat - zb)
                return float(F.softmax(logits, dim=1)[0, p.idx].item())
        except Exception as e:
            logger.debug("lewm.verify failed: %s", e)
            return None

    def dream(self, grid, actions, ctx: Any = None, game: Any = None) -> list:
        """Roll the world model's IMAGINATION forward: encode `grid`, then for
        each action in `actions` predict the next latent and DECODE it to a grid.
        Returns [grid_after_a0, grid_after_a1, ...] (each a GxG list of ids) --
        the frames the model believes those actions produce. Best-effort; returns
        [] when disabled and skips any step that fails.

        Actions may carry coordinates ("ACTION6(19,16)" / {"id":6,"x":19,"y":16}), so the
        playground and the planner can finally dream a CLICK rather than a coordinate-less
        ACTION6 the model has no way to interpret."""
        if not self.ok or grid is None or not actions:
            return []
        try:
            z = self.encode(grid)
            if z is None:
                return []
            frames = []
            for a in actions:
                z = self.predict(z, a, ctx, game=game)
                if z is None:
                    break
                g = self.decode(z)
                if g is None:
                    break
                frames.append(g)
            return frames
        except Exception as e:
            logger.debug("lewm.dream failed: %s", e)
            return []

    def train_step(self) -> Optional[dict]:
        """One SGD step on a random minibatch: L = MSE + lambda*SIGReg.

        Returns a small metrics dict, or None if disabled / not enough data.
        This is the ONLY place the encoder + predictor are updated; the
        prediction target is the encoder's own output for the next frame (no
        EMA, no stop-gradient) -- SIGReg is what prevents collapse."""
        if not self.ok or len(self._buf) < max(4, min(self.batch, 8)):
            return None
        self.predictor.train()      # dropout ON for the training forwards; eval() restored below
        n = min(self.batch, len(self._buf))
        idx = self._sample_batch(n)
        # stored as uint8 -> cast back to long for the colour-embedding lookup
        ids_t = torch.stack([self._buf[i][0] for i in idx]).to(self.device).long()      # [n,G,G]
        a = torch.tensor([self._buf[i][1] for i in idx], dtype=torch.long, device=self.device)
        ids_tp1 = torch.stack([self._buf[i][2] for i in idx]).to(self.device).long()    # [n,G,G]
        ax = torch.tensor([self._buf[i].ax if len(self._buf[i]) > 6 else -1 for i in idx],
                          dtype=torch.long, device=self.device)
        ay = torch.tensor([self._buf[i].ay if len(self._buf[i]) > 6 else -1 for i in idx],
                          dtype=torch.long, device=self.device)
        games = [self._buf[i][3] for i in idx]
        coord = self._coord_map(ax, ay)                 # [n,P,Gh,Gw] or None
        gctx = self._game_ctx(games)                    # [n,ctx_dim] or None

        # Build aux-cond minibatches when the FiLM channel is on. Items lacking a
        # cond (grid-only transitions) contribute a zero vector, so mixed streams
        # train fine; if NO item in the batch has a cond, we skip film entirely.
        cond_t = cond_tp1 = None
        if self.film is not None:
            def _stack_cond(slot: int):
                any_c = False
                mats = []
                for i in idx:
                    c = self._buf[i][slot]
                    if c is not None:
                        any_c = True
                        mats.append(c.to(self.device).float())
                    else:
                        mats.append(torch.zeros(self.D_c, device=self.device))
                return torch.stack(mats) if any_c else None
            cond_t = _stack_cond(4)
            cond_tp1 = _stack_cond(5)

        z_t = self._encode_ids(ids_t, cond_t)   # [n, D]
        z_tp1 = self._encode_ids(ids_tp1, cond_tp1)  # [n, D]  (target: encoder's OWN output)
        # reason_ctx: condition the predictor on the CHILD's reasoning (cond_tp1) instead of
        # the game embedding -- the signal that says WHICH child this descend leads to.
        pred_ctx = cond_tp1 if (self.reason_ctx and cond_tp1 is not None) else gctx
        # The predictor sees the ADAPTED latent (identity when the adapter is off). The
        # contrastive term below shapes THIS view, so the forward model gets purpose-separated
        # features while encode()/the map keep the raw frozen geometry.
        za_t = self._adapt(z_t)
        za_tp1 = self._adapt(z_tp1)
        # C-JEPA object masking (identity anchor): corrupt ONLY the predictor's
        # input copy (the ADAPTED view -- identity when the adapter is off).
        # Everything below -- the MSE target, the identity baseline d_id,
        # SIGReg, and the IDM's real/pred branches -- reads the intact z_t.
        z_t_in = za_t
        if self.spatial and self.obj_mask_frac > 0:
            z_t_in = self._apply_object_mask(za_t, ids_t)
        z_hat = self.predictor(z_t_in, self._onehot(a), ctx=pred_ctx, coord=coord)

        # Change-weighted latent MSE. An unweighted mean lets the ~30-40% of transitions
        # that change NOTHING contribute a third of the gradient, all of it pointing at
        # identity. w is renormalised to mean 1, so the total gradient magnitude -- and
        # hence the balance against SIGReg -- is unchanged.
        with torch.no_grad():
            changed = (ids_t != ids_tp1).flatten(1).any(dim=1).float()          # [n]
            w_mse = 1.0 + self.mse_change_weight * changed
            w_mse = w_mse / w_mse.mean().clamp_min(1e-6)
        # De-gameable, identity-relative on-axis MSE (forward-predictor fix).
        #  * target_detach: stop-grad the target so MSE can't be cut by pulling successive
        #    latents together (the encoder is otherwise in the optimizer, non-detached).
        #  * identity_relative: pay the term only PAST the identity baseline d_id, i.e. train
        #    the gate's skill numerator directly instead of a per-dim mean that reads "small"
        #    while being worse than doing nothing. A floor on WHEN it switches off, not a
        #    direction change -- the real levers are detach/freeze + mse_lambda + cutting the
        #    off-axis terms (pred_ce, inv). All default to today's exact behaviour.
        z_tgt = z_tp1.detach() if self.target_detach else z_tp1
        d_hat = (z_hat - z_tgt).pow(2).mean(dim=1)                        # [n]
        if self.identity_relative:
            with torch.no_grad():
                d_id = (z_t.detach() - z_tgt).pow(2).mean(dim=1)          # identity baseline
            core = F.relu(d_hat - d_id + self.idrel_margin)              # paid only past identity
        else:
            core = d_hat
        mse = (core * w_mse).mean() if self.mse_change_weight > 0 else core.mean()
        # SIGReg over the union of current + next latents (more directions of
        # variation for the isotropy test). It exists to stop the ENCODER collapsing to a
        # constant -- so with the encoder frozen it has nothing to regularise and is only
        # a (large) constant on the loss. Skip it there; the frozen latent cannot collapse.
        if self.freeze_encoder or self.sig_lambda <= 0:
            sig = z_t.new_tensor(0.0)
        else:
            sig = self._sigreg(torch.cat([z_t, z_tp1], dim=0))
        loss = self.mse_lambda * mse + self.sig_lambda * sig

        # SUPERVISED CONTRASTIVE (Phase 1b): separate directory latents by their LLM READING.
        # Union both endpoints (parent z_t / child z_tp1) with their conds for more samples +
        # negatives; keep only rows with a real (non-zero) cond; label = argmax over the purpose
        # dims (0:13). Mean-pool the spatial map -> [m, Dch] (cheap: an m x m sim matrix, m<=2n),
        # normalise, SupCon over cosine sims. When the encoder is UNFROZEN this shapes its latent
        # geometry so parent/child of DIFFERENT kinds stop collapsing together -- the fix the
        # gates trace to. When frozen it is a no-op on the encoder (z detached upstream), so run
        # it unfrozen (with SIGReg off; contrastive is the anti-collapse).
        con_val = None
        if (self.contrastive_lambda > 0 and self.spatial
                and (cond_t is not None or cond_tp1 is not None)):
            zc = z_t.new_zeros((z_t.shape[0], self.D_c))
            z_all = torch.cat([za_t, za_tp1], dim=0)            # [2n, D] the PREDICTOR's view
            c_all = torch.cat([cond_t if cond_t is not None else zc,
                               cond_tp1 if cond_tp1 is not None else zc], dim=0)  # [2n, D_c]
            real = c_all.abs().sum(dim=1) > 1e-6
            if int(real.sum()) >= 4:
                z_real = z_all[real]
                lab = c_all[real][:, :13].argmax(dim=1)             # purpose label per dir
                m = z_real.shape[0]
                zp = z_real.view(m, self.Dch, self.Gh, self.Gw).mean(dim=(2, 3))  # [m, Dch]
                zp = F.normalize(zp, dim=1)
                eye = torch.eye(m, dtype=torch.bool, device=zp.device)
                same = (lab[:, None] == lab[None, :])
                if self.contrastive_mode == "repel":
                    # PUSH-ONLY: penalise cosine similarity between DIFFERENT-purpose dirs past
                    # a margin. No positive pull -> same-kind directories are never collapsed
                    # onto each other, so the map keeps its distinct landmarks (coverage) while
                    # the kinds still separate (the flatness fix the predictor needs).
                    neg = (~same) & ~eye
                    if bool(neg.any()):
                        cos = zp @ zp.t()                                     # [m, m] in [-1,1]
                        con = F.relu(cos[neg] - self.contrastive_margin).pow(2).mean()
                        loss = loss + self.contrastive_lambda * con
                        con_val = round(float(con.item()), 6)
                else:                                                        # full SupCon
                    sim = (zp @ zp.t()) / max(self.contrastive_temp, 1e-4)   # [m, m]
                    pos = same & ~eye
                    if bool(pos.any()):
                        sim = sim.masked_fill(eye, -1e9)
                        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
                        pc = pos.sum(dim=1)
                        valid = pc > 0
                        if bool(valid.any()):
                            con = -(logp * pos).sum(dim=1)[valid] / pc[valid].clamp_min(1)
                            con = con.mean()
                            loss = loss + self.contrastive_lambda * con
                            con_val = round(float(con.item()), 6)

        # PREDICT-THEN-DECODE: decode the PREDICTED next latent and penalise it
        # against the ACTUAL next grid, upweighting transitions that changed. The
        # latent MSE alone is minimised by identity (next≈current) because ARC
        # frames barely move turn-to-turn — so the predictor learns to ignore the
        # action. Grounding the prediction in the real next grid (via the trained
        # decoder, used here as a frozen renderer: _opt doesn't hold its params,
        # and _dec_opt.zero_grad() below clears these grads before the decoder's
        # own step) forces it to model each action's actual effect.
        pred_ce_val = None
        if self.pred_ce_lambda > 0:
            with torch.no_grad():
                # PER-CELL weighting. This used to weight whole TRANSITIONS ("did anything
                # change?"), which barely helped: inside a transition the CE was still a
                # flat mean over all 4096 cells, and only ~32 of them (0.8%) actually move.
                # So per-cell, "predict no change" scores 99.2% and identity stays optimal
                # -- which is precisely what we kept observing: every action decoded to the
                # same grid even while the inverse-dynamics head could read the action out
                # of the latent at 0.99. The predictor was moving the latent in directions
                # the DECODER ignores, because nothing ever paid it to move the pixels.
                # Upweight the cells that actually changed, by ~1/(their frequency), so the
                # handful of moving cells carry real gradient. Renormalised to mean 1, so
                # the term's overall scale (and its balance against MSE/SIGReg) is unchanged.
                cell_chg = (ids_t != ids_tp1).float()                     # [n,G,G] raw diff
                if self.llm_mask:
                    # Stage B: keep only diff-cells the LLM put inside the action's EFFECT
                    # region; down-weight the rest as incidental flicker/animation. Coarse
                    # box -> tolerant of gemma4-12b's +-1-cell registration error. This only
                    # denoises the per-cell CE WEIGHT (where flicker lives); it never edits a
                    # target. No effmask (old ckpt / no label) -> byte-identical raw diff.
                    for j, i in enumerate(idx):
                        m = self._buf[i].effmask if len(self._buf[i]) > 11 else None
                        if m is not None:
                            cell_chg[j] = cell_chg[j] * m.to(self.device).float()
                w_cell = 1.0 + self.change_weight * cell_chg
                w_cell = w_cell / w_cell.mean().clamp_min(1e-6)
            ce_map = F.cross_entropy(self.decoder(z_hat), ids_tp1, reduction="none")  # [n,G,G]
            pred_ce = (ce_map * w_cell).mean()
            loss = loss + self.pred_ce_lambda * pred_ce
            pred_ce_val = round(float(pred_ce.item()), 6)

        # INVERSE DYNAMICS -- the anti-collapse mechanism (replaces margin repulsion,
        # which was tried three ways and gamed three ways; see the ctor's comment).
        #
        # Ask the IDM to name the action from the latent DISPLACEMENT, on both branches:
        #   real branch  Δz  = enc(g_t+1) - enc(g_t)   -> trains the IDM on GENUINE board
        #                                                 change, and (via inv_enc_scale)
        #                                                 shapes the ENCODER so its latent
        #                                                 geometry is action-linearised.
        #                                                 This is what makes the prediction
        #                                                 TARGETS action-distinguishable --
        #                                                 which is what stops identity from
        #                                                 being the optimal answer.
        #   pred branch  Δẑ = pred(z_t,a) - z_t        -> cycle-consistency (ACID): if the
        #                                                 action you conditioned on is not
        #                                                 recoverable from what you
        #                                                 predicted, you did not predict it.
        #
        # The IDM's own weights are trained ONLY on the real branch: on the predicted
        # branch its parameters are frozen (grads reach the predictor, not the head). That
        # detail is load-bearing. A jointly-trained head can co-adapt to a "watermark"
        # channel the predictor scribbles the action into -- satisfying the loss while the
        # decoded grid never moves. That is the null-space attack again, one level up.
        # Freezing the head on the predicted branch makes it a CRITIC of real dynamics: to
        # fool it, Δẑ must actually look like a real displacement for that action.
        inv_a_val = inv_c_val = inv_acc = click_acc = None
        if self.idm is not None and self.inv_lambda > 0:
            dz_real = _GradScale.apply(z_tp1, self.inv_enc_scale) - \
                _GradScale.apply(z_t, self.inv_enc_scale)
            dz_pred = z_hat - z_t

            # ONLY transitions that actually CHANGED the board can supervise the IDM.
            # If nothing moved, the displacement is ~0 no matter which action you took, so
            # "which action produced this?" has NO answer -- and ~35% of ARC transitions are
            # no-ops (half of all clicks land on nothing). Training the head on those maps
            # 0 -> 7 different labels, which is pure label noise: it drags the head toward
            # the marginal, caps inv_acc, and worse, pressures the predictor to manufacture
            # spurious differences on transitions where the correct answer is "nothing
            # happens". Masking them out is not a convenience -- an unidentifiable target
            # is not a target.
            m_id = changed.bool()
            if bool(m_id.any()):
                la_r, lc_r = self.idm(dz_real[m_id])          # head TRAINS here
                for p_ in self.idm.parameters():
                    p_.requires_grad_(False)
                la_p, lc_p = self.idm(dz_pred[m_id])          # head is a FROZEN critic here
                for p_ in self.idm.parameters():
                    p_.requires_grad_(True)
                a_id = a[m_id]

                ce_a = 0.5 * (F.cross_entropy(la_r, a_id) + F.cross_entropy(la_p, a_id))
                inv = ce_a
                inv_a_val = round(float(ce_a.item()), 6)

                # WHERE did you click? Same rule: only clicks that CHANGED something. A
                # click that did nothing leaves no trace of where it landed.
                ax_id, ay_id = ax[m_id], ay[m_id]
                cmask = (ax_id >= 0) & (ay_id >= 0)
                if bool(cmask.any()) and self.inv_click_lambda > 0:
                    tgt = ((ay_id[cmask] * self.Gh) // self.G) * self.Gw + \
                          ((ax_id[cmask] * self.Gw) // self.G)
                    ce_c = 0.5 * (F.cross_entropy(lc_r[cmask], tgt) +
                                  F.cross_entropy(lc_p[cmask], tgt))
                    inv = inv + self.inv_click_lambda * ce_c
                    inv_c_val = round(float(ce_c.item()), 6)
                    with torch.no_grad():
                        click_acc = round(float((lc_r[cmask].argmax(1) == tgt).float()
                                                .mean().item()), 4)
                loss = loss + self.inv_lambda * inv
                with torch.no_grad():
                    # THE diagnostic. Chance is 1/A = 0.14. If this does not climb, the
                    # latent does not carry the action and nothing downstream can.
                    inv_acc = round(float((la_r.argmax(1) == a_id).float().mean().item()), 4)

        # ROLLOUT (V-JEPA 2-AC): teacher forcing only ever trains one-step prediction from
        # a REAL latent, but we ship multi-step dreams that feed the predictor its OWN
        # output. Take a second step from z_hat and compare to the real z_{t+2}, so the
        # model learns to survive its own error instead of compounding it.
        roll_val = None
        chain = self._chain_batch(idx) if self.roll_lambda > 0 else None
        if chain is not None:
            pos, a2, ax2, ay2, ids_tp2 = chain
            # the predictor always consumes the ADAPTED view -- including its own output when
            # we feed a dream back in (z_hat lives in the frozen space, same as encode()).
            z2 = self.predictor(self._adapt(z_hat[pos]), self._onehot(a2),
                                ctx=None if gctx is None else gctx[pos],
                                coord=self._coord_map(ax2, ay2))
            with torch.no_grad():
                z_tp2 = self._encode_ids(ids_tp2, None)
            roll = F.l1_loss(z2, z_tp2)
            loss = loss + self.roll_lambda * roll
            roll_val = round(float(roll.item()), 6)

        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        self._step += 1

        # Decoder co-training: reconstruct the CURRENT grid from a DETACHED latent
        # (no grad into the encoder -> JEPA dynamics are byte-for-byte unaffected).
        # The decoder tracks the evolving encoder so "dream" frames stay fresh.
        rec_val = None
        try:
            if self._dec_opt is not None:
                dec_logits = self.decoder(z_t.detach())      # [n, C, G, G]
                rec = F.cross_entropy(dec_logits, ids_t)      # ids_t: [n, G, G] targets
                self._dec_opt.zero_grad()
                rec.backward()
                self._dec_opt.step()
                rec_val = round(float(rec.item()), 6)
            else:
                # frozen: still REPORT reconstruction so a regression would be visible,
                # but never step it.
                with torch.no_grad():
                    rec_val = round(float(F.cross_entropy(
                        self.decoder(z_t), ids_t).item()), 6)
        except Exception as e:
            logger.debug("lewm: decoder co-train step skipped: %s", e)

        # INVERSE-DYNAMICS TRUST HEAD (detached) -- mirror of the decoder co-train above.
        # Names the action from the REAL latent displacement on a fully stop-grad'd latent,
        # stepped by its OWN optimizer, so NO gradient reaches predictor/encoder (self._opt
        # already saw the champion loss untouched). This is the anti-collapse guarantee:
        # skill_med cannot move by construction. inv_a_val/inv_acc were pre-set to None at
        # the top of train_step and the legacy inv block is skipped when inv_lambda==0.
        if self._idm_opt is not None:
            try:
                m_id = changed.bool()                     # only transitions that moved
                if bool(m_id.any()):
                    dz_real = z_tp1.detach()[m_id] - z_t.detach()[m_id]
                    a_id = a[m_id]
                    la_r, _ = self.idm(dz_real)       # click head untrained; verify() skips it
                    ce_a = F.cross_entropy(la_r, a_id)
                    self._idm_opt.zero_grad()
                    ce_a.backward()
                    self._idm_opt.step()
                    inv_a_val = round(float(ce_a.item()), 6)
                    with torch.no_grad():
                        inv_acc = round(float((la_r.argmax(1) == a_id)
                                              .float().mean().item()), 4)
            except Exception as e:
                logger.debug("lewm: detached trust-head step skipped: %s", e)

        with torch.no_grad():
            z_std = float(z_t.std().item())
            # EMA of encoded latents -> the /probe novelty/frontier signal.
            bm = z_t.mean(dim=0).detach()
            if self._z_ema is None:
                self._z_ema = bm.clone()
            else:
                self._z_ema.mul_(self._z_ema_beta).add_(bm, alpha=1.0 - self._z_ema_beta)
            # PROXY for wm_latent_gate skill: per-sample identity ratio, SUM over dims, on
            # CHANGED rows, e_id>1e-9 filtered -- the gate's own estimator on this predictor
            # output. NOT the authority (the gate runs predict() in a separate process on the
            # buffer); a valid live watch that "skill climbs" tracks "gate crosses 0". Report
            # the MEDIAN (the mean is tail-dominated, same as the gate) + the win-rate.
            skill_med = float("nan")
            skill_win = float("nan")
            ch = changed.bool()
            if bool(ch.any()):
                ep_ = (z_hat[ch] - z_tp1[ch]).pow(2).sum(dim=1)
                ei_ = (z_t[ch] - z_tp1[ch]).pow(2).sum(dim=1)
                gd = ei_ > 1e-9
                if bool(gd.any()):
                    sk = 1.0 - ep_[gd] / ei_[gd]
                    skill_med = float(sk.median().item())
                    skill_win = float((sk > 0).float().mean().item())
        lval = float(loss.item())
        if self._init_loss is None:
            self._init_loss = lval
        self.predictor.eval()           # restore the eval() invariant (dropout off for inference)
        return {
            "step": self._step,
            "loss": round(lval, 6),
            "mse": round(float(mse.item()), 6),
            "sigreg": round(float(sig.item()), 6),
            "con": con_val,            # supervised-contrastive loss (Phase 1b encoder fix)
            "pred_ce": pred_ce_val,
            "inv_act": inv_a_val,      # inverse-dynamics CE on the action
            "inv_click": inv_c_val,    # inverse-dynamics CE on the click cell
            "inv_acc": inv_acc,        # <- THE number to watch. chance = 1/A = 0.14
            "click_acc": click_acc,    # chance = 1/(Gh*Gw) = 0.004
            # forward-predictor skill proxy (== wm_latent_gate estimator; >0 = beats identity)
            "skill_med": round(skill_med, 4) if skill_med == skill_med else None,
            "skill_win": round(skill_win, 4) if skill_win == skill_win else None,
            "rollout": roll_val,
            "recon": rec_val,
            "z_std": round(z_std, 6),
            "buffer": len(self._buf),
            "total_seen": self._total_seen,
            "n_games": len(self._games),
        }

    def _sample_batch(self, n: int):
        """Minibatch indices, with the no-op class held to `noop_frac`.

        At the natural rate 30-40% of transitions change zero cells (many ARC clicks land
        on nothing), so that fraction of every gradient explicitly rewards identity --
        the exact failure we are fixing. Do NOT drive it to zero: correctly predicting a
        genuine no-op IS dynamics, and a model that never predicts one is just broken in
        the other direction."""
        if self.noop_frac < 0 or not self._buf:
            return np.random.choice(len(self._buf), size=n, replace=False)
        # Cached class partition. Rebuilding it per step is an O(buffer) scan on the hot
        # path -- at 18k transitions that alone was most of a 6.6 s/step.
        if self._cls_dirty or self._cls_chg is None:
            chg, noop = [], []
            for i, b in enumerate(self._buf):
                (chg if (b.chg if len(b) > 10 else True) else noop).append(i)
            self._cls_chg, self._cls_noop = chg, noop
            self._cls_dirty = False
        chg, noop = self._cls_chg, self._cls_noop
        if not noop or not chg:
            return np.random.choice(len(self._buf), size=n, replace=False)
        n_noop = min(len(noop), int(round(n * self.noop_frac)))
        n_chg = min(len(chg), n - n_noop)
        n_noop = min(len(noop), n - n_chg)      # backfill if `chg` was short
        pick = list(np.random.choice(chg, size=n_chg, replace=False))
        pick += list(np.random.choice(noop, size=n_noop, replace=False))
        return np.array(pick)

    def _chain_batch(self, idx):
        """For rows in the batch that have a buffered SUCCESSOR (same game, same episode,
        step t+1), return (batch_pos, a2, ax2, ay2, ids_tp2) for the 2-step rollout.

        The successor is looked up by (game, ep, t+1) and then VALIDATED by grid equality:
        row i's next_grid must equal row j's grid. Reservoir eviction overwrites random
        slots and two concurrent runs can interleave on the same game, so a stale or
        crossed link is possible -- the equality check makes it self-healing rather than
        silently training on a transition pair that never happened."""
        if not self._succ:
            return None
        pos, a2, ax2, ay2, tp2 = [], [], [], [], []
        for bp, i in enumerate(idx):
            b = self._buf[i]
            if len(b) <= 10 or b.game is None or b.ep < 0:
                continue
            j = self._succ.get((b.game, b.ep, b.t + 1))
            if j is None or j >= len(self._buf):
                continue
            nb = self._buf[j]
            if len(nb) <= 10 or not torch.equal(nb.ids_t, b.ids_tp1):
                continue                       # stale/crossed link -> drop, don't trust it
            pos.append(bp)
            a2.append(nb.a)
            ax2.append(nb.ax)
            ay2.append(nb.ay)
            tp2.append(nb.ids_tp1)
        if len(pos) < 2:
            return None
        dev = self.device
        return (torch.tensor(pos, dtype=torch.long, device=dev),
                torch.tensor(a2, dtype=torch.long, device=dev),
                torch.tensor(ax2, dtype=torch.long, device=dev),
                torch.tensor(ay2, dtype=torch.long, device=dev),
                torch.stack(tp2).to(dev).long())

    def _reindex_succ(self) -> None:
        """Rebuild the (game, ep, t) -> slot index used by _chain_batch."""
        self._cls_dirty = True
        self._succ = {}
        for i, b in enumerate(self._buf):
            if len(b) > 10 and b.game is not None and b.ep >= 0:
                self._succ[(b.game, b.ep, b.t)] = i

    def dataset_stats(self) -> dict:
        """Diversity/accumulation readout for /health + eval.

        `actions` and `drops` are the two that matter now: an action histogram piled onto
        index 0, or a climbing `unparseable` count, is the signature of the mislabelling
        bug that made action-conditioned dynamics unlearnable. It used to be invisible."""
        acts: dict = {}
        n_click = n_noop = 0
        for b in self._buf:
            acts[b[1]] = acts.get(b[1], 0) + 1
            if len(b) > 10:
                n_click += int(b.ax >= 0)
                n_noop += int(not b.chg)
        n = max(len(self._buf), 1)
        return {
            "buffer": len(self._buf),
            "buffer_cap": self.buffer_cap,
            "policy": self.buffer_policy,
            "total_seen": self._total_seen,
            "n_games": len(self._games),
            "games": dict(sorted(self._games.items(), key=lambda kv: -kv[1])[:20]),
            "actions": {f"ACTION{k + 1}": v for k, v in sorted(acts.items())},
            "n_click": n_click,
            "noop_frac": round(n_noop / n, 4),
            "drops": dict(self._drops),
            "chains": len(self._succ),
            "games_registered": len(self._game_slots),
        }

    def _sigreg(self, z):
        """SIGReg loss for a batch of latents z [B, D].

        Standardize per-dim, project onto M fixed random unit directions, and
        for each 1D projection compute the Epps-Pulley statistic (squared
        difference between the empirical characteristic-function modulus and
        the standard-Gaussian one, |ecf(t)|^2 vs exp(-t^2), averaged over test
        points t). By Cramer-Wold, driving every projection's statistic to 0
        drives z toward N(0, I). Fully differentiable.

        Spatial latent: z arrives flat [B, L]; treat every TOKEN as a sample
        ([B*Gh*Gw, Dch]) so isotropy is enforced on the per-token distribution
        (prevents global AND spatial collapse, and gives far more samples for a
        better Epps-Pulley estimate)."""
        if self.spatial:
            nb = z.shape[0]
            z = z.view(nb, self.Dch, self.Gh * self.Gw).permute(0, 2, 1).reshape(-1, self.Dch)
        B = z.shape[0]
        if B < 4:
            return z.new_tensor(0.0)
        zs = (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-8)      # [B, feat]
        proj = zs @ self._sig_dirs                            # [B, M]
        t = self._sig_t                                       # [T]
        # arg[t, b, m] = t * proj[b, m]
        arg = t[:, None, None] * proj[None, :, :]             # [T, B, M]
        cos_m = arg.cos().mean(dim=1)                         # [T, M]
        sin_m = arg.sin().mean(dim=1)                         # [T, M]
        ecf_sq = cos_m ** 2 + sin_m ** 2                      # [T, M]
        gcf_sq = (-(t ** 2)).exp()[:, None]                   # [T, 1]
        ep = ((ecf_sq - gcf_sq) ** 2).mean(dim=0)             # [M]  per-direction stat
        return ep.mean()

    def surprise(self, grid: Any, action: Any, next_grid: Any,
                 cond: Any = None, next_cond: Any = None, ctx: Any = None) -> Optional[float]:
        """||predictor(enc(grid), a) - enc(next_grid)||^2 -- the model's own
        prediction error, a stuck/anomaly signal. Optional aux conds fuse into the
        encodings and optional ctx into the predictor. None when disabled."""
        if not self.ok:
            return None
        try:
            z = self.encode(grid, cond)
            z_next = self.encode(next_grid, next_cond)
            if z is None or z_next is None:
                return None
            z_hat = self.predict(z, action, ctx)
            return float(((z_hat - z_next) ** 2).sum().item())
        except Exception as e:
            logger.debug("lewm.surprise failed: %s", e)
            return None

    def probe(self, grid: Any, cond: Any = None) -> Optional[dict]:
        """Self-factualization readout: decode where-am-I / how-novel-is-here from
        the current latent. Returns the latent plus derived scalars -- its norm and
        a novelty = ||z - running_mean(z)|| (high = frontier / unfamiliar region of
        latent space, the intrinsic-curiosity drive; low = well-understood). None
        when disabled. This is the /probe endpoint the .JEPA integration specced."""
        if not self.ok:
            return None
        z = self.encode(grid, cond)
        if z is None:
            return None
        with torch.no_grad():
            znorm = float(z.norm().item())
            novelty = float((z - self._z_ema).norm().item()) if self._z_ema is not None else None
        return {
            "z": z.detach().cpu().flatten().tolist(),
            "z_norm": round(znorm, 6),
            "z_mean": round(float(z.mean().item()), 6),
            "z_std": round(float(z.std().item()), 6),
            "novelty": round(novelty, 6) if novelty is not None else None,
            "D": self.D, "D_c": self.D_c, "ctx_dim": self.ctx_dim,
        }

    def value(self, grid: Any, cond: Any = None) -> Optional[float]:
        """V(grid): the head's estimate of discounted future EXTRINSIC return (task value:
        high = a level is reachable, low = near GAME_OVER). None when the head is absent or
        the model is disabled. Under no_grad."""
        if not self.ok or self.value_head is None:
            return None
        z = self.encode(grid, cond)
        if z is None:
            return None
        with torch.no_grad():
            return float(self.value_head(z.unsqueeze(0)).squeeze(0).item())

    def train_value_step(self, ids_batch: Any, returns_batch: Any) -> Optional[dict]:
        """One supervised regression step for the value head ONLY, on a DETACHED frozen
        latent: minimise smooth_l1(V(enc(grid).detach()), precomputed n-step return). The
        encoder never moves (detach), so this is safe to run on the banked forward-fixed
        checkpoint. `ids_batch` is [n,G,G] ints; `returns_batch` is [n] floats. Returns a
        small metrics dict, or None if the head is absent. This is the offline-proof path;
        online value learning (in train_step) reuses the same target."""
        if not self.ok or self.value_head is None:
            return None
        ids = torch.as_tensor(np.asarray(ids_batch), dtype=torch.long, device=self.device)
        if ids.dim() == 2:
            ids = ids.unsqueeze(0)
        g = torch.as_tensor(np.asarray(returns_batch), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self._encode_ids(ids).detach()       # frozen latent; encoder never trains here
        v = self.value_head(z)                       # [n]
        loss = F.smooth_l1_loss(v, g)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        with torch.no_grad():
            # explained-variance-ish readout: how well V tracks the returns this batch
            var = float(g.var().item())
            resid = float((v - g).var().item())
            ev = 1.0 - resid / var if var > 1e-8 else 0.0
        return {"value_loss": round(float(loss.item()), 6),
                "value_mean": round(float(v.mean().item()), 4),
                "return_mean": round(float(g.mean().item()), 4),
                "explained_var": round(ev, 4)}

    def plan(
        self,
        grid: Any,
        goal_grid: Any,
        horizon: Optional[int] = None,
        iters: Optional[int] = None,
    ) -> List[str]:
        """Discrete-action CEM planning in latent space.

        Maintain a per-step categorical distribution over ACTION1..ACTION7.
        Each iteration: sample `cem_samples` horizon-H action sequences, roll
        them out through the predictor in latent space starting from enc(grid),
        score each by terminal ||z_H - z_goal||^2, refit the per-step
        categoricals to the top-`cem_elites` (with Laplace smoothing). Return
        the best elite's action sequence as ACTIONk labels. Empty list when
        disabled."""
        if not self.ok:
            return []
        H = int(horizon if horizon is not None else self.cem_horizon)
        it = int(iters if iters is not None else self.cem_iters)
        N = self.cem_samples
        K = min(self.cem_elites, N)
        try:
            with torch.no_grad():
                z0 = self.encode(grid)
                z_goal = self.encode(goal_grid)
                if z0 is None or z_goal is None:
                    return []
                z0 = z0.to(self.device)
                z_goal = z_goal.to(self.device)
                # per-step categorical logits over A actions, init uniform
                probs = torch.full((H, self.A), 1.0 / self.A, device=self.device)
                best_seq = None
                best_cost = float("inf")
                for _ in range(max(1, it)):
                    # sample N sequences: [N, H]
                    seqs = torch.stack(
                        [torch.multinomial(probs[h], N, replacement=True) for h in range(H)],
                        dim=1,
                    )  # [N, H]
                    z = z0.unsqueeze(0).expand(N, -1).contiguous()  # [N, D]
                    for h in range(H):
                        a_oh = self._onehot(seqs[:, h])
                        z = self.predictor(z, a_oh)
                    cost = ((z - z_goal.unsqueeze(0)) ** 2).sum(dim=-1)  # [N]
                    order = torch.argsort(cost)
                    elite_idx = order[:K]
                    if float(cost[order[0]].item()) < best_cost:
                        best_cost = float(cost[order[0]].item())
                        best_seq = seqs[order[0]].clone()
                    elites = seqs[elite_idx]  # [K, H]
                    # refit per-step categoricals from elite action counts (+1 smoothing)
                    new_probs = torch.ones(H, self.A, device=self.device)
                    for h in range(H):
                        counts = torch.bincount(elites[:, h], minlength=self.A).float()
                        new_probs[h] = new_probs[h] + counts
                    probs = new_probs / new_probs.sum(dim=1, keepdim=True)
                if best_seq is None:
                    return []
                return [_ACTION_ORDER[int(i)] if int(i) < len(_ACTION_ORDER) else f"ACTION{int(i)+1}"
                        for i in best_seq.tolist()]
        except Exception as e:
            logger.debug("lewm.plan failed: %s", e)
            return []

    def plan_curious(
        self,
        grid: Any,
        game: Any = None,
        horizon: Optional[int] = None,
        iters: Optional[int] = None,
        novelty_w: Optional[float] = None,
        change_w: Optional[float] = None,
    ) -> dict:
        """GOAL-FREE curiosity planning through the (fixed) forward model.

        plan() needs a goal_grid that ARC-AGI-3 never provides; this maximises INTRINSIC
        reward instead, so it plans with nothing but the world model. Same discrete-action
        CEM as plan(), but each horizon-H rollout is scored by cumulative intrinsic reward:

            reward = novelty_w * ||z_h - z_ema||   +   change_w * ||z_h - z_{h-1}||
                     (frontier: distance from the running latent mean, the /probe signal)
                     (change:   how much this action actually MOVES the state -- rewards
                      effect-having actions over no-ops)

        Unlike plan(), the rollout passes the GAME embedding (ctx) to the predictor -- plan()
        passed neither game nor coords, so it rolled out the cross-game-averaged dynamics the
        model was explicitly fixed to avoid. Coords are omitted (a coord-free rollout; the
        planner searches over WHICH action, not where to click).

        Only meaningful now that the forward map beats identity (63%, rank 2.22); over the
        old broken predictor every rollout was noise. Returns
        {"actions": [ACTIONk,...], "reward": float, "per_action_reward": [...]} -- or
        {"actions": []} when disabled. Receding-horizon callers execute actions[0] and replan.
        """
        if not self.ok:
            return {"actions": []}
        H = int(horizon if horizon is not None else self.cem_horizon)
        it = int(iters if iters is not None else self.cem_iters)
        nw = _envf("ARC_WM_CURIOUS_NOVELTY_W", 1.0) if novelty_w is None else float(novelty_w)
        cw = _envf("ARC_WM_CURIOUS_CHANGE_W", 1.0) if change_w is None else float(change_w)
        N = self.cem_samples
        K = min(self.cem_elites, N)
        try:
            with torch.no_grad():
                z0 = self.encode(grid)
                if z0 is None:
                    return {"actions": []}
                z0 = z0.to(self.device)
                zema = self._z_ema.to(self.device) if self._z_ema is not None else None
                # game embedding broadcast to the whole sample batch (the rollout fix)
                gctx = self._game_ctx([game] * N) if game is not None else None
                probs = torch.full((H, self.A), 1.0 / self.A, device=self.device)
                best_seq = None
                best_reward = -float("inf")
                best_per = None
                for _ in range(max(1, it)):
                    seqs = torch.stack(
                        [torch.multinomial(probs[h], N, replacement=True) for h in range(H)],
                        dim=1,
                    )  # [N, H]
                    z = z0.unsqueeze(0).expand(N, -1).contiguous()  # [N, D]
                    total = torch.zeros(N, device=self.device)
                    per_step = []  # mean reward per horizon step, for readout
                    for h in range(H):
                        a_oh = self._onehot(seqs[:, h])
                        z_next = self.predictor(z, a_oh, ctx=gctx)      # ctx = game (rollout fix)
                        change = (z_next - z).pow(2).sum(dim=-1).sqrt()  # [N] how much it moved
                        if zema is not None:
                            nov = (z_next - zema.unsqueeze(0)).pow(2).sum(dim=-1).sqrt()
                        else:
                            nov = torch.zeros(N, device=self.device)
                        r = nw * nov + cw * change
                        total = total + r
                        per_step.append(float(r.mean().item()))
                        z = z_next
                    # TERMINAL VALUE (value-guided planning): add gamma^H * V(z_H) so the plan
                    # accounts for TASK value beyond the myopic horizon -- reach toward a level,
                    # away from the cliff -- not just immediate novelty. z holds z_H for all N.
                    # Guarded: no head or zero weight -> pure curiosity (byte-identical).
                    if self.value_head is not None and self.value_terminal_w > 0:
                        total = total + (self.value_terminal_w * (self.gamma ** H)
                                         * self.value_head(z))
                    order = torch.argsort(total, descending=True)   # MAXIMISE reward
                    if float(total[order[0]].item()) > best_reward:
                        best_reward = float(total[order[0]].item())
                        best_seq = seqs[order[0]].clone()
                        best_per = per_step
                    elites = seqs[order[:K]]                          # [K, H]
                    new_probs = torch.ones(H, self.A, device=self.device)
                    for h in range(H):
                        counts = torch.bincount(elites[:, h], minlength=self.A).float()
                        new_probs[h] = new_probs[h] + counts
                    probs = new_probs / new_probs.sum(dim=1, keepdim=True)
                if best_seq is None:
                    return {"actions": []}
                acts = [_ACTION_ORDER[int(i)] if int(i) < len(_ACTION_ORDER) else f"ACTION{int(i)+1}"
                        for i in best_seq.tolist()]
                return {"actions": acts, "reward": round(best_reward, 6),
                        "per_action_reward": [round(p, 6) for p in (best_per or [])]}
        except Exception as e:
            logger.debug("lewm.plan_curious failed: %s", e)
            return {"actions": []}

    # -- persistence ------------------------------------------------------------

    def _config_dict(self) -> dict:
        """The minimal architecture config needed to reconstruct a matching
        model before loading weights: latent dim, grid size, SIGReg M, action
        count (plus the other shape-determining hyperparameters)."""
        return {
            "G": self.G,
            "C": self.C,
            "Cemb": self.Cemb,
            "D": self.D,
            "A": self.A,
            "sig_M": self.sig_M,
            "sig_T": self.sig_T,
            "D_c": self.D_c,
            "ctx_dim": self.ctx_dim,
            "spatial": bool(self.spatial),
            "Dch": self.Dch,
            "Gh": self.Gh,
            "Gw": self.Gw,
            "coord_planes": self.coord_planes,
            "coord_sigma": self.coord_sigma,
            "n_game_slots": self.n_game_slots,
            "game_key": self.game_key,
            "inv_hidden": self.inv_hidden,
            "idm_present": self.idm is not None,
            "value_hidden": self.value_hidden,
            "value_head_present": self.value_head is not None,
            "fs_adapter_present": self.fs_adapter is not None,
        }

    def save(self, path: str) -> bool:
        """Persist encoder + predictor state_dicts, the optimizer state, the
        fixed SIGReg projection basis, and the architecture config to `path`
        via torch.save. Returns True on success, False (no raise) when disabled
        or on any I/O error. A model trained offline can then be reloaded with
        LeWorldModel.load(path) and reused across solver runs."""
        if not self.ok:
            logger.warning("lewm.save: torch unavailable; nothing to save")
            return False
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            payload = {
                "format": "lewm-v3",   # v3: coord/game conditioning + IDM + buf_schema 2
                "config": self._config_dict(),
                "encoder": self.encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "opt": self._opt.state_dict(),
                # SIGReg basis is a fixed random (non-learned) tensor; persist it
                # so surprise/plan geometry is bit-reproducible across reloads.
                "sig_dirs": self._sig_dirs.detach().cpu(),
                "sig_t": self._sig_t.detach().cpu(),
                "step": self._step,
                "init_loss": self._init_loss,
                "z_ema": None if self._z_ema is None else self._z_ema.detach().cpu(),
            }
            # Move module tensors to CPU for a portable, device-agnostic file.
            payload["encoder"] = {k: v.detach().cpu() for k, v in payload["encoder"].items()}
            payload["predictor"] = {k: v.detach().cpu() for k, v in payload["predictor"].items()}
            # FiLM aux-fusion head (only present when the aux channel is enabled).
            if self.film is not None:
                payload["film"] = {k: v.detach().cpu() for k, v in self.film.state_dict().items()}
            # Pixel decoder (dream mode). Optional key -> older loaders ignore it,
            # and loaders WITH a decoder guard on ckpt.get("decoder"), so this is
            # fully backward/forward compatible (mirrors the film precedent).
            if getattr(self, "decoder", None) is not None:
                payload["decoder"] = {k: v.detach().cpu() for k, v in self.decoder.state_dict().items()}
                try:
                    payload["dec_opt"] = self._dec_opt.state_dict()
                except Exception:
                    pass
            # Persist the replay buffer (compact uint8) so the accumulated, multi-game
            # transition DATASET survives restarts -- not just the weights. This is what
            # lets the dataset COMPOUND across runs instead of restarting empty each boot.
            if self.idm is not None:
                payload["idm"] = {k: v.detach().cpu()
                                  for k, v in self.idm.state_dict().items()}
            if self.value_head is not None:
                payload["value_head"] = {k: v.detach().cpu()
                                         for k, v in self.value_head.state_dict().items()}
            if self.fs_adapter is not None:
                payload["fs_adapter"] = {k: v.detach().cpu()
                                         for k, v in self.fs_adapter.state_dict().items()}
            if self.game_emb is not None:
                payload["game_emb"] = {k: v.detach().cpu()
                                       for k, v in self.game_emb.state_dict().items()}
                payload["game_slots"] = dict(self._game_slots)
            if self._buf:
                # buf_schema 2 adds the click coords + episode chain. A schema-1 buffer is
                # DISCARDED on load: its action labels were destroyed at write time (clicks
                # coerced to index 0, coords thrown away) and nothing here can recover them.
                payload["buf_schema"] = 2
                payload["buf_ids_t"] = torch.stack([b[0] for b in self._buf])     # [N,G,G] uint8
                payload["buf_a"] = torch.tensor([b[1] for b in self._buf], dtype=torch.long)
                payload["buf_ids_tp1"] = torch.stack([b[2] for b in self._buf])   # [N,G,G] uint8
                payload["buf_games"] = [b[3] for b in self._buf]
                # Per-item aux conds (f16 tensor or None) -> keep the multimodal
                # transition dataset whole across restarts, not just the grids.
                payload["buf_cond_t"] = [b[4] if len(b) > 4 else None for b in self._buf]
                payload["buf_cond_tp1"] = [b[5] if len(b) > 5 else None for b in self._buf]
                payload["buf_ax"] = torch.tensor([b.ax for b in self._buf], dtype=torch.int16)
                payload["buf_ay"] = torch.tensor([b.ay for b in self._buf], dtype=torch.int16)
                payload["buf_ep"] = torch.tensor([b.ep for b in self._buf], dtype=torch.int32)
                payload["buf_t"] = torch.tensor([b.t for b in self._buf], dtype=torch.int32)
                payload["buf_chg"] = torch.tensor([b.chg for b in self._buf], dtype=torch.bool)
            payload["total_seen"] = self._total_seen
            payload["games"] = self._games
            payload["drops"] = dict(self._drops)
            # ATOMIC write (2026-07-15): a docker stop mid-torch.save TRUNCATED
            # lewm-live.ckpt (28MB of 53MB, "failed finding central directory")
            # and the service booted disabled. Write to a sibling tmp file and
            # os.replace() it into place -- readers only ever see a complete file.
            #
            # 2026-07-26: that was NOT enough -- lewm-live.ckpt was truncated the
            # SAME way again (60,187,545 bytes, zip end-of-central-directory
            # missing) and the service came up ok=False for 8 days: every
            # /dream|/imagine|/surprise 503'd, so the live page's imagination and
            # WM-learning panels were blank. os.replace() only orders the RENAME;
            # on a Docker-Desktop Windows bind mount the tmp file's tail can still
            # be in flight when the container is killed, so the rename publishes a
            # short file. Three additions close it for good:
            #   1. fsync the tmp file (and its directory) BEFORE the rename, so the
            #      bytes are durable before anything points at them;
            #   2. VERIFY the tmp is a readable zip archive first -- is_zipfile()
            #      reads the end-of-central-directory record, i.e. precisely the
            #      thing that goes missing. A bad tmp is discarded, keeping the
            #      last good checkpoint in place instead of publishing a brick;
            #   3. keep the outgoing checkpoint as <path>.prev, so even a
            #      corruption that slips through has a same-generation rollback
            #      (load() falls back to it automatically).
            tmp = f"{path}.tmp-{os.getpid()}"
            try:
                torch.save(payload, tmp)
                # "rb+", not "rb": Windows FlushFileBuffers requires WRITE access,
                # so fsync on a read-only handle is EBADF and save() failed on every
                # Windows host (found 2026-08-01 saving the first value-head ckpt).
                # Linux fsync accepts either; rb+ is portable.
                with open(tmp, "rb+") as fh:     # durability before visibility
                    os.fsync(fh.fileno())
                import zipfile
                if not zipfile.is_zipfile(tmp):
                    logger.error("lewm.save: freshly written %s is NOT a complete zip "
                                 "(truncated write) -- keeping previous checkpoint %s",
                                 tmp, path)
                    return False
                if os.path.exists(path):         # same-generation rollback copy
                    try:
                        import shutil
                        shutil.copy2(path, f"{path}.prev")
                    except Exception as e:       # noqa: BLE001
                        logger.warning("lewm.save: could not refresh %s.prev (%s)", path, e)
                os.replace(tmp, path)
                try:                             # persist the rename itself
                    d_fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
                    try:
                        os.fsync(d_fd)
                    finally:
                        os.close(d_fd)
                except (OSError, AttributeError):
                    pass                         # not supported on every fs/platform
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            return True
        except Exception as e:  # never raise into the caller
            logger.warning("lewm.save failed: %s", e)
            return False

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "LeWorldModel":
        """Reconstruct a ready-to-use LeWorldModel from a checkpoint written by
        save(). The architecture is rebuilt from the stored config (so latent
        dim / grid size / SIGReg M / action count match the trained weights),
        weights + optimizer + SIGReg basis are restored, and everything is moved
        to the best available device (respecting `device` / ARC_WM_DEVICE).

        Degrades gracefully: if torch is unavailable, or the file is missing /
        corrupt, returns a disabled model (self.ok == False) rather than
        raising, so a bad checkpoint can never break the solver's hot path."""
        if not _TORCH_OK:
            logger.warning("lewm.load: torch unavailable; returning disabled model")
            return cls()
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            model = cls(
                grid=cfg.get("G"),
                latent=cfg.get("D"),
                n_actions=cfg.get("A"),
                device=device,
                seed=seed,
            )
            if not model.ok:
                return model
            # Migration path: ARC_WM_MIGRATE_BUFFER=1 keeps the freshly-built
            # (env-configured) modules and restores ONLY the replay buffer (raw
            # uint8 grids -> architecture independent). This is how the flat G=64
            # checkpoint's accumulated dataset carries into the fresh spatial model
            # WITHOUT the checkpoint's own (flat) arch config overriding the target
            # spatial arch. ARC_WM_RESET_PREDICTOR keeps encoder+decoder but starts
            # the predictor fresh (escape a stuck predictor).
            migrate = os.environ.get("ARC_WM_MIGRATE_BUFFER", "") in ("1", "true", "yes")
            reset_pred = os.environ.get("ARC_WM_RESET_PREDICTOR", "") in ("1", "true", "yes")

            if not migrate:
                # Normal resume: the checkpoint's arch config WINS for anything that
                # determines the shape of a STORED weight (it must match those weights).
                # spatial/Dch/Gh/Gw are new in the v3 spatial format; older ckpts lack
                # them -> treated as flat.
                for attr in ("C", "Cemb", "sig_M", "sig_T", "Dch", "Gh", "Gw"):
                    val = cfg.get(attr)
                    if val is not None:
                        setattr(model, attr, int(val))
                # D_c: adopt the LARGER of ckpt/env -- a reasoning-cond ckpt loaded with the
                # env flag unset must keep its FiLM head (and its weights) rather than
                # silently dropping the conditioning it was trained with.
                model.D_c = max(int(cfg.get("D_c", 0) or 0), int(model.D_c))
                # ctx_dim / coord_planes / inv_hidden size modules that are NEW (game_film,
                # the coord stem, the IDM). Letting the checkpoint win here would mean any
                # pre-existing ckpt (which has ctx_dim=0) silently switches game and click
                # conditioning back OFF on every resume -- a feature that fails closed and
                # says nothing. The ENV wins for these; the non-strict loader fills the new
                # modules fresh and logs exactly which keys it initialised.
                for attr in ("ctx_dim", "coord_planes", "inv_hidden", "n_game_slots"):
                    val = cfg.get(attr)
                    if val is not None and int(val) != int(getattr(model, attr)):
                        logger.warning("lewm.load: %s: ckpt=%s ENV=%s -> using ENV (new "
                                       "conditioning module will be fresh-initialised)",
                                       attr, val, getattr(model, attr))
                if "spatial" in cfg:
                    model.spatial = bool(cfg["spatial"])
                # value head: adopt the ckpt's hidden dim, and BUILD the head whenever the
                # checkpoint carries one (so its weights restore even if the value env flags
                # are off this run -- e.g. a read-only /value or gate call on a value ckpt).
                if cfg.get("value_hidden") is not None:
                    model.value_hidden = int(cfg["value_hidden"])
                if cfg.get("value_head_present"):
                    model.value_head_on = True
                # BUILD the IDM whenever the ckpt carries one, so verify()/dream confidence
                # light up on a read-only serve-time load with the env flag unset.
                if cfg.get("idm_present"):
                    model.inv_detached = True
                # Same rule for the predictor's adapter: BUILD it whenever the ckpt carries one,
                # so predict() reproduces the trained forward model even when the env flag is
                # off this run (e.g. a read-only task-gate call on an adapter checkpoint). An
                # adapter ckpt loaded WITHOUT the adapter would silently predict from the raw
                # latent -- a different model, reported as the same one.
                if cfg.get("fs_adapter_present"):
                    model.fs_adapter_on = True
                model._sig_feat_dim = model.Dch if model.spatial else int(cfg.get("D", model.D))
                model.D = (model.Dch * model.Gh * model.Gw) if model.spatial else int(cfg.get("D", model.D))
                model._build_modules()
            # (migrate: keep the ctor's env-derived spatial arch + fresh modules)

            def _guarded(module, key, label):
                """Restore a module NON-STRICTLY.

                This used to be a strict load_state_dict(). Strict means that the moment
                a new parameter is added to a module (the coord stem, the game FiLM), an
                OLD checkpoint raises KeyError -> the except branch below silently
                reinitialises the ENTIRE module. That would have thrown away the trained
                encoder/predictor -- and the banked 0.94 reconstruction with it -- while
                logging one line that reads like a routine notice. Load what matches,
                report what didn't, and never discard a whole module over a new key."""
                sd = ckpt.get(key)
                if sd is None:
                    return False
                try:
                    res = module.load_state_dict(sd, strict=False)
                    if getattr(res, "missing_keys", None):
                        logger.warning("lewm.load: %s NEW keys, fresh-initialised: %s",
                                       label, list(res.missing_keys))
                    if getattr(res, "unexpected_keys", None):
                        logger.warning("lewm.load: %s DROPPED stale keys: %s",
                                       label, list(res.unexpected_keys))
                    module.to(model.device)
                    return True
                except Exception as e:
                    # only a genuine SHAPE conflict reaches here now
                    logger.warning("lewm.load: %s not restored (fresh): %s", label, e)
                    return False

            if migrate:
                logger.warning("lewm.load: ARC_WM_MIGRATE_BUFFER set -> FRESH weights, "
                               "restoring ONLY the replay buffer (spatial=%s Dch=%s)",
                               model.spatial, model.Dch)
                enc_ok = False
            else:
                enc_ok = _guarded(model.encoder, "encoder", "encoder")
                if reset_pred:
                    logger.warning("lewm.load: ARC_WM_RESET_PREDICTOR set -> FRESH predictor")
                else:
                    _guarded(model.predictor, "predictor", "predictor")
                if model.film is not None:
                    _guarded(model.film, "film", "film")
                _guarded(model.decoder, "decoder", "decoder")
                if model.idm is not None:
                    _guarded(model.idm, "idm", "idm")
                if model.value_head is not None:
                    _guarded(model.value_head, "value_head", "value_head")
                if model.fs_adapter is not None:
                    _guarded(model.fs_adapter, "fs_adapter", "fs_adapter")
                if model.game_emb is not None:
                    _guarded(model.game_emb, "game_emb", "game_emb")
                    model._game_slots = dict(ckpt.get("game_slots") or {})
                if ckpt.get("dec_opt") is not None:
                    try:
                        model._dec_opt.load_state_dict(ckpt["dec_opt"])
                    except Exception as e:
                        logger.debug("lewm.load: decoder optimizer not restored: %s", e)
            # z_ema / opt / sig_dirs: restore only when shapes are compatible
            # (a fresh spatial build won't match a flat checkpoint's tensors).
            ze = ckpt.get("z_ema")
            if ze is not None and not migrate and tuple(ze.shape) == (model.D,):
                try:
                    model._z_ema = ze.to(model.device)
                except Exception:
                    model._z_ema = None
            if not migrate and "opt" in ckpt and enc_ok:
                try:
                    model._opt.load_state_dict(ckpt["opt"])
                except Exception as e:
                    # EXPECTED whenever the param SET changes -- e.g. FREEZE_ENCODER drops the
                    # encoder group, a future D_c>0 inserts a FiLM param. Adam's accumulated
                    # second-moment estimates are then gone, so the predictor sees a brief
                    # effective-LR transient. WARN (not debug): "loads warm" is true for
                    # WEIGHTS, not the optimizer -- run enough steps for the transient to wash.
                    logger.warning("lewm.load: optimizer state NOT restored (param set changed: "
                                   "freeze/D_c) -> Adam moments reset. %s", e)
            sd = ckpt.get("sig_dirs")
            if sd is not None and sd.shape[0] == model._sig_feat_dim:
                model._sig_dirs = sd.to(model.device)
            if "sig_t" in ckpt:
                model._sig_t = ckpt["sig_t"].to(model.device)
            model._step = int(ckpt.get("step", 0))
            model._init_loss = ckpt.get("init_loss", None)
            # Restore the accumulated transition dataset if present (older ckpts lack it,
            # in which case the buffer simply starts empty and refills).
            if "buf_ids_t" in ckpt:
                schema = int(ckpt.get("buf_schema", 1))
                keep_legacy = _envi("ARC_WM_BUFFER_KEEP_LEGACY", 0) not in (0,)
                if schema < 2 and not keep_legacy:
                    # A schema-1 buffer's action labels are CORRUPT and unrecoverable: at
                    # write time every click was coerced to index 0 (colliding with ACTION1)
                    # with its coordinates discarded, as was every RESET and every
                    # first-turn None. In the live buffer that was 583 of 710 rows. Fitting
                    # a dynamics model to it makes identity the correct answer. Weights are
                    # KEPT (the 0.94 reconstruction lives there, and it is sound); only the
                    # poisoned transitions are dropped. Set ARC_WM_BUFFER_KEEP_LEGACY=1 to
                    # override, which is a rollback lever, not a good idea.
                    logger.warning(
                        "lewm.load: legacy buffer (schema 1) DISCARDED -- %d transitions "
                        "with corrupt action labels. Weights retained; the buffer refills "
                        "from correctly-labelled play.", len(ckpt["buf_a"]))
                else:
                    try:
                        bt, ba, btp = ckpt["buf_ids_t"], ckpt["buf_a"], ckpt["buf_ids_tp1"]
                        N = len(ba)
                        bg = ckpt.get("buf_games") or [None] * N
                        # v1 checkpoints lack conds -> restore as None (grid-only), so
                        # an old dataset upgrades in place to the newer schema.
                        bct = ckpt.get("buf_cond_t") or [None] * N
                        bctp = ckpt.get("buf_cond_tp1") or [None] * N
                        bax = ckpt.get("buf_ax")
                        bay = ckpt.get("buf_ay")
                        bep = ckpt.get("buf_ep")
                        bt_ = ckpt.get("buf_t")
                        bchg = ckpt.get("buf_chg")
                        model._buf = [
                            _Tr(bt[i].to(torch.uint8), int(ba[i]), btp[i].to(torch.uint8),
                                bg[i], bct[i], bctp[i],
                                int(bax[i]) if bax is not None else -1,
                                int(bay[i]) if bay is not None else -1,
                                int(bep[i]) if bep is not None else -1,
                                int(bt_[i]) if bt_ is not None else -1,
                                bool(bchg[i]) if bchg is not None
                                else bool((bt[i] != btp[i]).any().item()))
                            for i in range(N)]
                        model._reindex_succ()
                    except Exception as e:
                        logger.debug("lewm.load: buffer not restored: %s", e)
            model._total_seen = int(ckpt.get("total_seen", len(model._buf)))
            model._games = dict(ckpt.get("games") or {})
            model._drops = dict(ckpt.get("drops") or model._drops)
            # Re-register every game present in the restored buffer, so slots survive a
            # restart even if the ckpt predates the registry.
            for b in model._buf:
                if b[3] is not None:
                    model._game_slot(b[3], assign=True)
            return model
        except Exception as e:
            logger.warning("lewm.load failed (%s); returning disabled model", e)
            m = cls()
            m.ok = False
            return m

    def note(self) -> str:
        """One-line status for the planner prompt (mirrors prism_micro.note())."""
        if not self.ok:
            return ""
        return (
            f"lewm (learned latent world model, JEPA/LeWM): {len(self._buf)} transitions buffered, "
            f"{self._step} train steps; latent D={self.D}, SIGReg M={self.sig_M} lambda={self.sig_lambda}. "
            f"CEM plans horizon={self.cem_horizon} in latent space."
        )
