"""concept: a next-concept JEPA world model over embedding sequences.

Three published ideas, one engine:

  * Next Concept Prediction (ConceptLM, arXiv:2602.08984). Mean-pool every
    ``k`` consecutive embeddings into a concept, quantize concepts with a
    product codebook (S segments x N codes, SimVQ MLP over the codes so they
    do not collapse), and predict the NEXT concept as the current one plus a softmax-weighted sum
    of codes per segment. Loss = MSE(predicted, true next concept) + VQ loss.
  * Coarse-to-fine codes (Next-Concept Modeling, HazyResearch ncm). The
    quantizer is RESIDUAL across ``levels``: level 0 is the coarse concept
    (a topic-sized id), each later level quantizes what the previous left.
    ``concept_ids`` returns the whole hierarchy, coarse first.
  * SIGReg (LeWorldModel / LeJEPA). The prediction target is the encoder's
    own output with NO stop-gradient and NO EMA; the Gaussian regularizer on
    the concept latents is what prevents collapse, exactly as in lewm.py.

Observations are float sequences ``[T, d_in]`` (token hidden states from any
frozen LM, or one embedding per agent step). The engine never fetches
embeddings itself -- the caller supplies them -- so it stays dependency-light.

``plan(context, candidates)`` ranks candidate continuations (for example K
parallel drafts from a diffusion LM) by how close each one's concept lands to
the predicted next concept: the world model chooses, the fast generator
proposes.

Torch is optional at import time. Without it ``ok`` is False and every method
returns None / [] -- it never raises into a caller.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

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


def enabled() -> bool:
    """Usable here: torch imports. No env gate -- nothing consumes it by default."""
    return _TORCH_OK


def _sigreg(z: "torch.Tensor", dirs: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
    """Epps-Pulley SIGReg over a batch of latents z [B, D], tested against N(0, I).

    Unlike lewm.py the latents are NOT standardized first: a next-latent MSE
    with no stop-gradient can otherwise win by shrinking the whole latent scale
    (measured: in-order and shuffled surprise both fell to ~0.07). Testing the
    raw projections pins mean 0 AND unit scale, as LeJEPA does."""
    if z.shape[0] < 4:
        return z.new_tensor(0.0)
    proj = z @ dirs
    arg = t[:, None, None] * proj[None, :, :]
    # Complex ECF vs the N(0,1) characteristic function exp(-t^2/2). The
    # modulus-only form is translation-invariant, so a collapse to ONE point
    # scores |ecf|=1 everywhere with zero gradient -- measured stuck at 0.65.
    re = arg.cos().mean(dim=1) - (-(t ** 2) / 2).exp()[:, None]
    im = arg.sin().mean(dim=1)
    return (re ** 2 + im ** 2).mean()


if _TORCH_OK:

    class _ProductCodebook(nn.Module):
        """S independent codebooks of N codes; codes pass through a SimVQ MLP."""

        def __init__(self, dim: int, segments: int, codes: int):
            super().__init__()
            self.S, self.N, self.seg = segments, codes, dim // segments
            self.raw = nn.Parameter(torch.randn(segments, codes, self.seg) * 0.5)
            self.mlp = nn.Sequential(nn.Linear(self.seg, self.seg), nn.ReLU(),
                                     nn.Linear(self.seg, self.seg))

        def codes(self) -> "torch.Tensor":
            return self.mlp(self.raw)                                   # [S, N, seg]

        def quantize(self, c: "torch.Tensor"):
            """c [..., D] -> (quantized [..., D], ids [..., S])."""
            book = self.codes()
            cs = c.reshape(*c.shape[:-1], self.S, self.seg)             # [..., S, seg]
            d = ((cs.unsqueeze(-2) - book) ** 2).sum(-1)                # [..., S, N]
            ids = d.argmin(-1)                                          # [..., S]
            seg_idx = torch.arange(self.S, device=c.device).expand_as(ids)
            q = book[seg_idx, ids]                                      # [..., S, seg]
            return q.reshape(c.shape), ids

        def soft(self, logits: "torch.Tensor") -> "torch.Tensor":
            """logits [..., S, N] -> weighted code sum [..., D] (differentiable)."""
            w = logits.softmax(-1)
            q = torch.einsum("...sn,snd->...sd", w, self.codes())
            return q.reshape(*q.shape[:-2], self.S * self.seg)

    class _ConceptNet(nn.Module):
        def __init__(self, d_in: int, dim: int, segments: int, codes: int,
                     levels: int, layers: int, heads: int, max_concepts: int):
            super().__init__()
            # LayerNorm without affine pins every concept to norm sqrt(dim): the
            # next-latent MSE cannot win by shrinking the latent scale, and
            # SIGReg then only has to rule out the remaining collapse (every
            # concept the same direction). Without it: z std 0.02, sigreg stuck.
            self.enc = nn.Sequential(nn.Linear(d_in, dim), nn.GELU(), nn.Linear(dim, dim),
                                     nn.LayerNorm(dim, elementwise_affine=False))
            self.pos = nn.Parameter(torch.zeros(max_concepts, dim))
            block = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.ctx = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
            self.books = nn.ModuleList(_ProductCodebook(dim, segments, codes)
                                       for _ in range(levels))
            self.heads = nn.ModuleList(nn.Linear(dim, segments * codes) for _ in range(levels))
            self.S, self.N = segments, codes

        def concepts(self, x: "torch.Tensor", k: int) -> "torch.Tensor":
            """x [B, T, d_in] -> concept latents [B, T//k, dim] (mean-pool then encode)."""
            b, t, d = x.shape
            m = t // k
            pooled = x[:, : m * k].reshape(b, m, k, d).mean(2)
            return self.enc(pooled)

        def predict(self, c: "torch.Tensor") -> "torch.Tensor":
            """c [B, M, dim] -> predicted NEXT concept at every position [B, M, dim]."""
            m = c.shape[1]
            mask = torch.triu(torch.full((m, m), float("-inf"), device=c.device), 1)
            h = self.ctx(c + self.pos[:m], mask=mask)
            # RESIDUAL: next = current + a soft code mix. Predicting the absolute
            # next concept from codes regressed to the mean: on real text plan@1
            # was 8% (chance 10%) while "pick the candidate nearest the last
            # concept" scored 52%. As a delta it is 82% (n=120, 2026-10-19).
            out = c
            for head, book in zip(self.heads, self.books):
                logits = head(h).reshape(*h.shape[:-1], self.S, self.N)
                out = out + book.soft(logits)
            return out

        def residual_quantize(self, c: "torch.Tensor"):
            """Coarse-to-fine residual VQ: returns (sum of levels, [ids per level], vq_loss)."""
            resid, total, ids, loss = c, torch.zeros_like(c), [], c.new_tensor(0.0)
            for book in self.books:
                q, i = book.quantize(resid.detach())
                # codebook-only loss: ConceptLM confines L_VQ to the quantizer so
                # the codes track the latents, never the reverse (a commitment
                # term pulled the encoder onto a few codes and collapsed it).
                loss = loss + F.mse_loss(q, resid.detach())
                total = total + q
                ids.append(i)
                resid = resid - q.detach()
            return total, ids, loss


class ConceptWorldModel:
    """Next-concept JEPA engine. See the module docstring for the recipe."""

    def __init__(self, d_in: int, dim: int = 128, k: int = 4, segments: int = 8,
                 codes: int = 64, levels: int = 2, layers: int = 2, heads: int = 4,
                 max_concepts: int = 256, sigreg_lambda: float = 1.0, lr: float = 1e-3,
                 buffer_size: int = 4096, seed: int = 0, device: Optional[str] = None):
        self.ok = False
        self.k, self.d_in, self.lam = k, d_in, sigreg_lambda
        self.max_concepts = max_concepts
        self._buf: List[Any] = []
        self._buf_cap = buffer_size
        self.steps = 0
        if not _TORCH_OK:
            logger.warning("concept: torch unavailable; engine disabled (ok=False)")
            return
        if dim % segments:
            raise ValueError(f"dim {dim} must divide into {segments} segments")
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device or os.environ.get("AWPREDICT_DEVICE", "cpu"))
        self.net = _ConceptNet(d_in, dim, segments, codes, levels, layers, heads,
                               max_concepts).to(self.device)
        dirs = torch.randn(dim, 64, generator=g)
        self._sig_dirs = (dirs / dirs.norm(dim=0, keepdim=True)).to(self.device)
        self._sig_t = torch.linspace(0.2, 3.0, 17).to(self.device)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.cfg = dict(d_in=d_in, dim=dim, k=k, segments=segments, codes=codes,
                        levels=levels, layers=layers, heads=heads, max_concepts=max_concepts)
        self.ok = True

    # -- helpers -------------------------------------------------------------
    def _seq(self, obs: Any) -> Optional["torch.Tensor"]:
        """obs [T, d_in] or [B, T, d_in] -> float tensor [B, T, d_in] on device."""
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3 or x.shape[-1] != self.d_in or x.shape[1] < self.k:
            return None
        return x

    # -- WorldModel protocol -------------------------------------------------
    def observe(self, obs: Any, action: Any = None, next_obs: Any = None,
                *args: Any, **kwargs: Any) -> bool:
        """Buffer one sequence. ``next_obs`` (if given) is appended to ``obs``."""
        if not self.ok:
            return False
        seq = obs if next_obs is None else _concat(obs, next_obs)
        x = self._seq(seq)
        if x is None or x.shape[1] < 2 * self.k:
            return False
        self._buf.append(x[0].detach().cpu())
        if len(self._buf) > self._buf_cap:
            self._buf.pop(0)
        return True

    def encode(self, obs: Any, cond: Any = None) -> Optional["torch.Tensor"]:
        """Sequence -> concept latents [M, dim]."""
        if not self.ok:
            return None
        x = self._seq(obs)
        if x is None:
            return None
        with torch.no_grad():
            return self.net.concepts(x, self.k)[0]

    def predict(self, z: Any, action: Any = None, **kwargs: Any) -> Optional["torch.Tensor"]:
        """Concept latents [M, dim] -> predicted next concept [dim]."""
        if not self.ok or z is None:
            return None
        with torch.no_grad():
            c = torch.as_tensor(z, dtype=torch.float32, device=self.device)
            return self.net.predict(c.unsqueeze(0)[:, -self.max_concepts:])[0, -1]

    def surprise(self, obs: Any, action: Any = None, next_obs: Any = None,
                 *args: Any, **kwargs: Any) -> Optional[float]:
        """Mean squared next-concept prediction error over the sequence."""
        if not self.ok:
            return None
        seq = obs if next_obs is None else _concat(obs, next_obs)
        x = self._seq(seq)
        if x is None or x.shape[1] < 2 * self.k:
            return None
        with torch.no_grad():
            c = self.net.concepts(x, self.k)
            pred = self.net.predict(c[:, :-1])
            return float(((pred - c[:, 1:]) ** 2).sum(-1).mean().item())

    def train_step(self, batch_size: int = 32, steps: int = 1,
                   *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self.ok or not self._buf:
            return None
        out: Dict[str, Any] = {}
        self.net.train()
        for _ in range(max(1, steps)):
            idx = torch.randint(len(self._buf), (min(batch_size, len(self._buf)),))
            t = min(self._buf[i].shape[0] for i in idx.tolist())
            t = (min(t, self.max_concepts * self.k) // self.k) * self.k
            x = torch.stack([self._buf[i][:t] for i in idx.tolist()]).to(self.device)
            c = self.net.concepts(x, self.k)
            pred = self.net.predict(c[:, :-1])
            l_ncp = F.mse_loss(pred, c[:, 1:])
            _, _, l_vq = self.net.residual_quantize(c)
            l_sig = _sigreg(c.reshape(-1, c.shape[-1]), self._sig_dirs, self._sig_t)
            loss = l_ncp + l_vq + self.lam * l_sig
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            self.steps += 1
            out = {"loss": loss.item(), "ncp": l_ncp.item(), "vq": l_vq.item(),
                   "sigreg": l_sig.item(), "steps": self.steps}
        self.net.eval()
        return out

    def plan(self, context: Any, candidates: Sequence[Any] = (), *args: Any,
             **kwargs: Any) -> List[int]:
        """Rank candidate continuations of ``context`` best-first.

        Each candidate is a ``[t, d_in]`` sequence (t >= k). Score = squared
        distance between the predicted next concept and the candidate's first
        concept. Use it to pick among K parallel drafts.
        """
        if not self.ok or not candidates:
            return []
        z = self.encode(context)
        if z is None:
            return []
        target = self.predict(z)
        scores = []
        for i, cand in enumerate(candidates):
            zc = self.encode(cand)
            d = float("inf") if zc is None else float(((zc[0] - target) ** 2).sum())
            scores.append((d, i))
        return [i for _, i in sorted(scores)]

    # -- concept vocabulary --------------------------------------------------
    def concept_ids(self, obs: Any) -> Optional[List[List[List[int]]]]:
        """Discrete coarse-to-fine codes per chunk: [M][level][segment]."""
        if not self.ok:
            return None
        x = self._seq(obs)
        if x is None:
            return None
        with torch.no_grad():
            _, ids, _ = self.net.residual_quantize(self.net.concepts(x, self.k))
        stacked = torch.stack(ids, dim=-2)[0]                          # [M, levels, S]
        return stacked.tolist()

    def codebook_usage(self, obs_batch: Sequence[Any]) -> Optional[float]:
        """Fraction of level-0 codes used across ``obs_batch`` (mean over segments)."""
        if not self.ok:
            return None
        seen = [set() for _ in range(self.net.S)]
        for obs in obs_batch:
            ids = self.concept_ids(obs)
            for chunk in ids or []:
                for s, code in enumerate(chunk[0]):
                    seen[s].add(code)
        return sum(len(s) for s in seen) / (self.net.S * self.net.N)

    # -- persistence ---------------------------------------------------------
    def save(self, path: str, *args: Any, **kwargs: Any) -> bool:
        if not self.ok:
            return False
        tmp = f"{path}.tmp"
        try:
            torch.save({"cfg": self.cfg, "state": self.net.state_dict(),
                        "steps": self.steps}, tmp)
            os.replace(tmp, path)
            return True
        except OSError as exc:
            logger.warning("concept.save failed: %s", exc)
            return False

    def load(self, path: str, *args: Any, **kwargs: Any) -> bool:
        if not self.ok or not os.path.exists(path):
            return False
        try:
            ck = torch.load(path, map_location=self.device, weights_only=True)
            if ck.get("cfg") != self.cfg:
                logger.warning("concept.load: config mismatch %s != %s", ck.get("cfg"), self.cfg)
                return False
            self.net.load_state_dict(ck["state"])
            self.steps = int(ck.get("steps", 0))
            self.net.eval()
            return True
        except (OSError, RuntimeError, KeyError) as exc:
            logger.warning("concept.load failed: %s", exc)
            return False


def _concat(a: Any, b: Any) -> Any:
    if _TORCH_OK:
        return torch.cat([torch.as_tensor(a, dtype=torch.float32),
                          torch.as_tensor(b, dtype=torch.float32)], dim=-2)
    return list(a) + list(b)
