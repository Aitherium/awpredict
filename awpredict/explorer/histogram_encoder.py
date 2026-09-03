"""histogram_encoder: a deterministic cold-start encoder — the map before training.

The curiosity machinery (landmark map + policies) is encoder-agnostic: it
operates purely on latents. A trained engine (``core.lewm.LeWorldModel`` etc.)
supplies those latents from a checkpoint; this module supplies them with NO
weights at all — the category histogram of the token grid, normalized.

Why this is not a toy: the filesystem explorer's central observation is that
directories with similar TYPE-COMPOSITION are the same "kind of place". The
treemap the adapter renders already encodes that composition, so a histogram of
it is the honest zero-training stand-in for the trained encoder — directories
with similar content collapse to similar latents, and the dominant-type peek
that colors each child box makes descent visible in the parent's histogram.

What it cannot do (be honest about this): there is no learned structure, so the
``predictive`` and ``planner`` policies have no forward model to imagine with,
and the surprise signal is meaningless. Replace this with a trained engine the
moment one exists for your environment — the rest of the machinery does not
change. The blog's 23MB result used the trained engine; this is the bootstrap.
"""
from __future__ import annotations

from typing import Any, Optional

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on numpy-less installs
    _NP_AVAILABLE = False
    np = None  # type: ignore[assignment]


class HistogramEncoder:
    """Token grid -> normalized category-histogram latent (deterministic).

    ``encode(grid, cond=None)`` ignores ``cond`` (there is nothing to condition
    on without weights) but accepts it so the call shape matches a trained
    engine. The latent is the category histogram normalized to sum 1, padded to
    ``dim`` — two grids whose composition matches land at cosine ~1.
    """

    ok = True

    def __init__(self, n_categories: int = 16, dim: int = 64) -> None:
        self.n_categories = int(n_categories)
        self.dim = max(self.n_categories, int(dim))

    def encode(self, grid: Any, cond: Optional[Any] = None) -> Optional[Any]:
        """[H, W] int category ids -> [dim] float32 histogram latent, or None."""
        if not _NP_AVAILABLE:
            self.ok = False
            return None
        arr = np.asarray(grid)
        if arr.ndim == 0 or arr.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        ids = arr.reshape(-1).astype(np.int64)
        hist = np.bincount(ids, minlength=self.n_categories).astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist = hist / total
        out = np.zeros(self.dim, dtype=np.float32)
        out[: self.n_categories] = hist
        return out
