"""Synthetic demo worlds for ``awpredict viz --demo``.

Each builder returns ``(model, observations, actions, meta)``: a SMALL engine
trained briefly through the package's own training path (``observe`` +
``train_step``) on a world it can learn on CPU -- a couple of minutes for lewm
(1500 steps), seconds for mlp and concept -- plus an evaluation sequence to dream
along. For lewm and concept that sequence is freshly generated (a fixed path from
a fixed start; a new random drift), not a replay of the training buffer. The mlp
ring is small enough that every edge was seen in training: its demo shows
recall of a memorised dynamics, which is all a hash-embedding latent can offer.

  lewm     a 3x3 block moving on a 16x16 grid under ACTION1..4 (up/down/left/right)
  mlp      a point drifting around a 12-position ring, as state description strings
  concept  a 16-d vector drifting at a constant velocity, as an embedding sequence

Every builder raises :class:`awpredict.viz.VizUnavailableError` when it cannot
run (torch or numpy missing) so the CLI can exit 2 instead of writing nothing.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple

from .viz import VizUnavailableError

DemoWorld = Tuple[Any, List[Any], List[Any], Dict[str, Any]]

#: ARC palette ids used by the block world (idx 5 is black, 14 is green).
_BG, _BLOCK = 5, 14
_BLOCK_SIZE = 3
_GRID = 16
#: ARC action -> (dy, dx). ACTION1..4 read as up/down/left/right here.
_MOVES = {"ACTION1": (-1, 0), "ACTION2": (1, 0), "ACTION3": (0, -1), "ACTION4": (0, 1)}


def _need_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy ships with the dev extra
        raise VizUnavailableError("numpy is not installed") from exc
    return np


@contextmanager
def _env(overrides: Dict[str, str]) -> Iterator[None]:
    """Set env vars for the duration of a model construction, then restore them."""
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _block_grid(np: Any, y: int, x: int) -> List[List[int]]:
    g = np.full((_GRID, _GRID), _BG, dtype=int)
    g[y:y + _BLOCK_SIZE, x:x + _BLOCK_SIZE] = _BLOCK
    return g.tolist()


def _block_step(y: int, x: int, action: str) -> Tuple[int, int]:
    dy, dx = _MOVES[action]
    hi = _GRID - _BLOCK_SIZE
    return min(max(y + dy, 0), hi), min(max(x + dx, 0), hi)


def lewm_demo(steps: int = 6, train_steps: int = 1500, seed: int = 0) -> DemoWorld:
    """A moving block on a grid, learned by LeWorldModel."""
    np = _need_numpy()
    from .core import lewm

    if not lewm._TORCH_OK:
        raise VizUnavailableError("torch is not installed (lewm needs it)")
    # A small spatial latent so the demo trains on CPU in seconds. The overrides
    # are scoped to construction; the caller's environment is left untouched.
    with _env({"ARC_WM_SPATIAL": "1", "ARC_WM_DCH": "16", "ARC_WM_BATCH": "32",
               "ARC_WM_FREEZE_ENCODER": "0"}):
        model = lewm.LeWorldModel(grid=_GRID, device="cpu", seed=seed)
    if not model.ok:
        raise VizUnavailableError("LeWorldModel did not build (ok=False)")

    rng = np.random.default_rng(seed)
    actions = sorted(_MOVES)
    game = "demo-block"
    hi = _GRID - _BLOCK_SIZE
    for _ in range(60):                       # 60 random-walk episodes x 20 steps
        y, x = int(rng.integers(0, hi + 1)), int(rng.integers(0, hi + 1))
        grid = _block_grid(np, y, x)
        model.observe(grid, "RESET", grid, game=game)
        for _ in range(20):
            a = actions[int(rng.integers(0, len(actions)))]
            y, x = _block_step(y, x, a)
            nxt = _block_grid(np, y, x)
            model.observe(grid, a, nxt, game=game)
            grid = nxt
    losses = [m for m in (model.train_step() for _ in range(train_steps)) if m]

    # Held-out evaluation: a fixed path (right x3, down x3, ...) from a fresh start.
    y, x = 2, 1
    path = ["ACTION4", "ACTION4", "ACTION4", "ACTION2", "ACTION2", "ACTION2",
            "ACTION3", "ACTION1"][:max(steps, 1)]
    obs = [_block_grid(np, y, x)]
    for a in path:
        y, x = _block_step(y, x, a)
        obs.append(_block_grid(np, y, x))
    meta = {"world": "3x3 block on a 16x16 grid; ACTION1..4 = up/down/left/right",
            "train_steps": len(losses), "game": game,
            "final_loss": _last_loss(losses)}
    return _GameBound(model, game), obs, path, meta


class _GameBound:
    """Pins ``game=`` on predict/dream so the engine uses the demo's game embedding.

    Every other attribute (encode, decode, ok, ...) passes straight through, so the
    wrapped object still satisfies the WorldModel protocol.
    """

    def __init__(self, model: Any, game: str) -> None:
        self._model = model
        self._game = game

    def predict(self, z: Any, action: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("game", self._game)
        return self._model.predict(z, action, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


def mlp_demo(steps: int = 6, ring: int = 12, seed: int = 0) -> DemoWorld:
    """A point drifting around a ring, learned by MLPWorldModel's neural path."""
    _need_numpy()
    from .core import mlp

    if not mlp._TORCH_AVAILABLE:
        raise VizUnavailableError("torch is not installed (the MLP neural path needs it)")
    mlp.torch.manual_seed(seed)
    model = mlp.MLPWorldModel()
    transitions = []
    for _ in range(4):                         # every edge seen several times
        for i in range(ring):
            for a, d in (("cw", 1), ("ccw", -1)):
                s, n = f"ring pos={i}", f"ring pos={(i + d) % ring}"
                model.observe(hash(s), a, hash(n), 0.0, False, s, n)
                transitions.append(mlp.Transition(s, hash(s), a, n, hash(n), 0.0, False))
    status = model.train_step(transitions)
    obs = [f"ring pos={(3 + k) % ring}" for k in range(max(steps, 1) + 1)]
    acts = ["cw"] * max(steps, 1)
    meta = {"world": f"a point drifting clockwise around a {ring}-position ring",
            "trained": bool(status), "transitions": len(transitions)}
    return model, obs, acts, meta


def concept_demo(steps: int = 6, d_in: int = 16, k: int = 2, train_steps: int = 300,
                 seed: int = 0) -> DemoWorld:
    """A vector drifting at constant velocity, learned by ConceptWorldModel."""
    np = _need_numpy()
    from .core import concept

    if not concept._TORCH_OK:
        raise VizUnavailableError("torch is not installed (concept needs it)")
    model = concept.ConceptWorldModel(d_in=d_in, dim=64, k=k, segments=4, codes=16,
                                      levels=2, layers=2, heads=4, seed=seed)
    if not model.ok:
        raise VizUnavailableError("ConceptWorldModel did not build (ok=False)")
    rng = np.random.default_rng(seed)
    length = k * (max(steps, 1) + 4)

    def drift() -> Any:
        x0 = rng.normal(0.0, 1.0, d_in)
        v = rng.normal(0.0, 0.25, d_in)
        t = np.arange(length)[:, None]
        return (x0 + t * v + rng.normal(0.0, 0.02, (length, d_in))).astype("float32")

    for _ in range(256):
        model.observe(drift())
    losses = [model.train_step(batch_size=32) for _ in range(train_steps)]
    seq = drift()
    obs = [row.tolist() for row in seq]
    meta = {"world": f"a {d_in}-d vector drifting at constant velocity; concept = mean of {k}",
            "train_steps": model.steps, "final_loss": _last_loss([m for m in losses if m])}
    return model, obs, [], meta


def _last_loss(losses: List[Dict[str, Any]]) -> Any:
    if not losses:
        return None
    v = losses[-1].get("loss")
    return round(float(v), 5) if isinstance(v, (int, float)) else None


DEMOS = {"lewm": lewm_demo, "mlp": mlp_demo, "concept": concept_demo}
