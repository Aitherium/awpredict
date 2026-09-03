"""obs_adapter: the observation seam — the ONLY modality-locked part of a world model.

The engine (predictor, curiosity, planner, decoder) operates only on a spatial
latent ``[Dch, Gh, Gw]``; it never sees the raw observation. The only piece that
knows what a "grid" is, is the front door: an adapter that turns an observation
into a ``[H, W]`` map of small integer category ids. Generalizing a world model
to a new environment means supplying a different front door — nothing downstream
changes. This module formalizes that seam and ships the reference implementations
that made the filesystem-explorer result possible:

* ``ObsAdapter`` — the protocol: ``token_grid(obs) -> [H,W] ids`` (spatial
  structure) and ``cond(obs) -> [D_c] | None`` (global context, fused via FiLM).
  Never confuse the two: FiLM is one global affine over all tokens, it cannot
  place an entity at a location.
* ``GridAdapter`` — the identity adapter for ARC-style grids (reference).
* ``StructuredAdapter`` — a tree (filesystem, JSON, deps) rendered as a
  slice-and-dice treemap: area ~ size, colour = category id. This is the
  filesystem path — and the "decodable observation" lesson it encodes is the
  load-bearing one: if a subdirectory renders as a flat uniform block, the
  parent frame carries ZERO information about what is inside it, and
  (parent, click) -> child is not a learnable function. The fix costs one cheap
  per-child peek (dominant content type + real byte size), and it is exactly
  what ``token_grid_with_boxes`` + the cartographer's ``_peek`` provide.

Extracted whole from the Aitherium ARC-AGI-3 agent fork
(``agents/obs_adapter.py``); the protocol and both adapters are byte-identical
in behaviour to what the 23MB filesystem-explorer result was measured with.

Numpy is OPTIONAL at import time: every class exposes ``ok`` and returns
None/[] from methods when it is missing — degrade loudly, never raise into a
caller's turn loop.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on numpy-less installs
    _NP_AVAILABLE = False
    np = None  # type: ignore[assignment]


@runtime_checkable
class ObsAdapter(Protocol):
    """The observation seam: raw environment state -> the engine's token grid.

    ``token_grid`` is the only thing the engine's encoder sees; ``cond`` is an
    optional global side-channel (FiLM) for context the grid cannot carry.
    """

    name: str
    n_categories: int

    def token_grid(self, obs: Any) -> Any:
        """[H, W] int array of category ids in [0, n_categories). H/W arbitrary;
        the model pad/crops to its own G. Returns None when degraded."""
        ...

    def cond(self, obs: Any) -> Optional[Any]:
        """[D_c] float global-context vector for the FiLM seam, or None."""
        ...


class GridAdapter:
    """ARC grid -> itself. The model's tokenizer already clamps to [0,C) and
    pad/crops, so this is a pass-through: ``encode_obs(model, GridAdapter(), g)``
    == ``model.encode(g)``."""

    name = "grid"
    n_categories = 16
    ok = True

    def token_grid(self, obs: Any) -> Any:
        if not _NP_AVAILABLE:
            self.ok = False
            return None
        arr = np.asarray(obs)
        while arr.ndim > 2:
            arr = arr[-1]
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        return np.nan_to_num(arr).astype(np.int64)

    def cond(self, obs: Any) -> Optional[Any]:
        return None


class StructuredAdapter:
    """A tree of {"type","size","children"} -> a slice-and-dice treemap painted
    with per-leaf TYPE ids. The filesystem/network path: area ~ size and colour =
    classification. Leaves with a type outside the vocab fall to id 15 ("other")
    — dropped, never coerced into a wrong category.

    ``token_grid_with_boxes`` additionally returns, per DIRECT-child index, the
    (r0,r1,c0,c1) rectangle it painted — the click target for a "descend into
    child i" action. Valid only for a SHALLOW obs (children are leaves), which
    is exactly what ``explorer.fs_cartographer.observe_dir`` emits. The grid it
    returns is BYTE-IDENTICAL to ``token_grid`` on the same obs.
    """

    name = "structured"

    def __init__(self, type_vocab: Optional[List[str]] = None, grid: int = 64) -> None:
        self.grid = int(grid)
        self.type_vocab = type_vocab or [
            "dir", "code", "data", "media", "doc", "archive", "binary",
            "config", "log", "cache", "model", "secret", "unknown", "gap"]
        self.n_categories = 16
        self._id = {t: i + 1 for i, t in enumerate(self.type_vocab[:14])}
        self._other = 15
        self.ok = True

    # -- observation -> leaves -------------------------------------------------
    def _leaves(self, node: Any) -> List[Tuple[str, float]]:
        """Flatten to (type, size) leaves. A dir with children recurses; a leaf
        keeps its own size. Unknown types are kept AS "unknown" (id 15) — a
        dropped label is a dropped transition, never a default class."""
        kids = node.get("children")
        if kids:
            out: List[Tuple[str, float]] = []
            for k in kids:
                out.extend(self._leaves(k))
            return out
        t = str(node.get("type", "unknown")).lower()
        return [(t, max(1.0, float(node.get("size", 1))))]

    # -- the treemap -----------------------------------------------------------
    def token_grid(self, obs: Any) -> Any:
        if not _NP_AVAILABLE:
            self.ok = False
            return None
        grid_size = self.grid
        out = np.zeros((grid_size, grid_size), dtype=np.int64)
        leaves = self._leaves(obs) if isinstance(obs, dict) else []
        if not leaves:
            return out
        self._place(sorted(leaves, key=lambda x: -x[1]), out, 0, grid_size,
                    0, grid_size, True, boxes=None)
        return out

    def token_grid_with_boxes(self, obs: Any) -> Tuple[Any, Dict[int, Tuple[int, int, int, int]]]:
        """(grid, boxes) — see class docstring. A child whose rectangle collapsed
        below one cell is absent from ``boxes`` (its descend has no click
        location); the caller must skip it."""
        if not _NP_AVAILABLE:
            self.ok = False
            return None, {}
        grid_size = self.grid
        out = np.zeros((grid_size, grid_size), dtype=np.int64)
        kids = obs.get("children") if isinstance(obs, dict) else None
        if not kids:
            return out, {}
        items = [(i, str(k.get("type", "unknown")).lower(), max(1.0, float(k.get("size", 1))))
                 for i, k in enumerate(kids)]
        boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self._place(sorted(items, key=lambda x: -x[2]), out, 0, grid_size,
                    0, grid_size, True, boxes=boxes)
        return out, boxes

    def _place(self, items: List[Any], out: Any, r0: int, r1: int, c0: int, c1: int,
               horizontal: bool, boxes: Optional[Dict[int, Tuple[int, int, int, int]]]) -> None:
        """Slice-and-dice: alternate horizontal/vertical splits proportional to
        size. Items are tuples whose LAST element is the size and whose first is
        the key (type for the grid-only path, (child_index, type) for the boxes
        path) — the sort key (-size) and the stable tie order are identical in
        both paths, which is what keeps the two grids byte-identical."""
        if not items or r1 <= r0 or c1 <= c0:
            return
        if len(items) == 1:
            it = items[0]
            typ = it[0] if len(it) == 2 else it[1]     # (type,size) vs (idx,type,size)
            out[r0:r1, c0:c1] = self._id.get(typ, self._other)
            if boxes is not None and len(it) == 3:
                boxes[it[0]] = (r0, r1, c0, c1)
            return
        tot = sum(s for _, s in items) if len(items[0]) == 2 else sum(s for _, _, s in items)
        acc, cut = 0.0, 0
        for j, it in enumerate(items):
            s = it[-1]
            acc += s
            if acc >= tot / 2:
                cut = j + 1
                break
        cut = max(1, min(len(items) - 1, cut))
        left, right = items[:cut], items[cut:]
        frac = sum(s for _, _, s in left) / tot if len(items[0]) == 3 \
            else sum(s for _, s in left) / tot
        if horizontal:
            mid = c0 + int((c1 - c0) * frac)
            self._place(left, out, r0, r1, c0, mid, False, boxes)
            self._place(right, out, r0, r1, mid, c1, False, boxes)
        else:
            mid = r0 + int((r1 - r0) * frac)
            self._place(left, out, r0, mid, c0, c1, True, boxes)
            self._place(right, out, mid, r1, c0, c1, True, boxes)

    def cond(self, obs: Any) -> Optional[Any]:
        return None


def encode_obs(model: Any, adapter: ObsAdapter, obs: Any) -> Any:
    """Any observation -> the world model's spatial latent z, via ``adapter``.
    This is the whole generalization: swap the adapter, reuse the entire
    predictor/curiosity/value/planner."""
    grid = adapter.token_grid(obs)
    if grid is None:
        return None
    c = adapter.cond(obs)
    return model.encode(grid, cond=c if c is not None else None)
