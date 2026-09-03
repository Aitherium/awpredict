"""landmark: the persistent map — where the agent has been, what is related,
and where the unmapped frontier is.

The world model encodes ONE observation at a time. An explorer needs something
that ACCUMULATES: a growing set of LANDMARK latents — the distinct regions of
the environment observed so far (an online, growing self-organizing map in
latent space). It is environment-agnostic: it operates purely on latents ``z``
produced by any adapter, so it serves a filesystem, a website, or a camera
identically.

    localize(z) -> (idx, dist)   nearest landmark and its distance
    integrate(z) -> idx          match z (EMA-update) or ADD a new landmark
    novelty(z)   -> float        distance to nearest landmark = FRONTIER signal
    coverage()   -> dict         #landmarks + recent match-rate (-> 1 when done)

Three signals fall out for free:

* **Novelty** — high distance to the nearest landmark means unmapped territory.
  That is the frontier the curiosity policy climbs.
* **Coverage** — the rate at which new observations match existing landmarks.
  As it approaches 1, you are done exploring.
* **Compactness** — thousands of directories collapse into a few dozen kinds of
  place. Which is what a map *is*.

``calibrate`` sets ``match_radius`` between the same-region and different-region
distance regimes from a few labelled pairs — latents live at a scale set by the
engine's regularization, so a fixed radius is meaningless across engines.

Pure numpy (no torch): accepts a latent as a list / np array / torch tensor.
Numpy is OPTIONAL at import time; with it missing every method degrades loudly
(returns the "no map" value) rather than raising.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on numpy-less installs
    _NP_AVAILABLE = False
    np = None  # type: ignore[assignment]


def _to_vec(z: Any) -> Any:
    """A latent as float32 1-D ndarray. Accepts lists, ndarrays, torch tensors."""
    if hasattr(z, "detach"):
        z = z.detach().cpu().numpy()
    return np.asarray(z, dtype=np.float32).reshape(-1)


class MapMemory:
    """Persistent latent map as a growing set of landmark regions.

    ``match_radius`` is the squared-distance threshold below which an
    observation is deemed "already mapped" (matches a landmark) vs a new region.
    Calibrate it once with ``calibrate()`` from a few same-region vs
    different-region pairs, or pass it explicitly.
    """

    def __init__(self, match_radius: float = 1000.0, ema: float = 0.2,
                 max_landmarks: int = 4096, window: int = 32) -> None:
        self.match_radius = float(match_radius)
        self.ema = float(ema)
        self.max_landmarks = int(max_landmarks)
        self.window = int(window)
        self.landmarks: List[Any] = []
        self.counts: List[int] = []
        self._recent_hits: List[int] = []          # 1=matched existing, 0=new region
        self.ok = True

    # -- core ops ----------------------------------------------------------------
    def _dist2(self, z: Any) -> Any:
        if not _NP_AVAILABLE or not self.landmarks:
            return np.empty(0, dtype=np.float32) if _NP_AVAILABLE else []
        stacked = np.stack(self.landmarks)         # [K, D]
        return ((stacked - z[None, :]) ** 2).sum(1)  # [K]

    def localize(self, z: Any) -> Tuple[int, float]:
        """Nearest landmark index + squared distance. (-1, inf) when the map is
        empty or degraded."""
        if not _NP_AVAILABLE:
            self.ok = False
            return -1, float("inf")
        z = _to_vec(z)
        d2 = self._dist2(z)
        if not _NP_AVAILABLE or len(d2) == 0:
            return -1, float("inf")
        i = int(d2.argmin())
        return i, float(d2[i])

    def novelty(self, z: Any) -> float:
        """Distance to the nearest landmark = frontier signal (inf when the map
        is empty). High means "this is unmapped, go here"."""
        return self.localize(z)[1]

    def integrate(self, z: Any) -> int:
        """Match z to a landmark (EMA-update) or add a new one. Returns the
        landmark index; records whether it was a hit (mapped) or a miss (new
        frontier region) in the coverage window."""
        if not _NP_AVAILABLE:
            self.ok = False
            return -1
        z = _to_vec(z)
        i, d2 = self.localize(z)
        if i >= 0 and d2 <= self.match_radius:
            self.landmarks[i] = (1 - self.ema) * self.landmarks[i] + self.ema * z
            self.counts[i] += 1
            hit = 1
        else:
            if len(self.landmarks) < self.max_landmarks:
                self.landmarks.append(z.copy())
                self.counts.append(1)
                i = len(self.landmarks) - 1
            else:                                   # map full -> fold into nearest
                self.landmarks[i] = (1 - self.ema) * self.landmarks[i] + self.ema * z
                self.counts[i] += 1
            hit = 0
        self._recent_hits.append(hit)
        if len(self._recent_hits) > self.window:
            self._recent_hits = self._recent_hits[-self.window:]
        return i

    def coverage(self) -> Dict[str, Any]:
        """How mapped is the space: #landmarks and the recent match-rate. A high,
        stable match-rate = the accessible space is largely mapped (explore
        less); a low match-rate = still discovering new regions (keep going)."""
        if not _NP_AVAILABLE:
            self.ok = False
            return {"landmarks": 0, "match_rate": 0.0, "total_visits": 0}
        mr = (sum(self._recent_hits) / len(self._recent_hits)) if self._recent_hits else 0.0
        return {"landmarks": len(self.landmarks), "match_rate": round(mr, 3),
                "total_visits": int(sum(self.counts))}

    # -- calibration + Sensorium summary ----------------------------------------
    def calibrate(self, same_pairs: List[Tuple[Any, Any]],
                  diff_pairs: List[Tuple[Any, Any]], margin: float = 0.5) -> float:
        """Set match_radius between the same-region and different-region distance
        regimes. Picks a threshold ``margin`` of the way from the max same-region
        distance toward the min different-region distance (falls back to a
        midpoint if the regimes overlap)."""
        if not _NP_AVAILABLE:
            self.ok = False
            return self.match_radius
        same = [float(((_to_vec(a) - _to_vec(b)) ** 2).sum()) for a, b in same_pairs]
        diff = [float(((_to_vec(a) - _to_vec(b)) ** 2).sum()) for a, b in diff_pairs]
        hi_same = max(same) if same else 0.0
        lo_diff = min(diff) if diff else hi_same * 4 + 1.0
        if lo_diff > hi_same:
            self.match_radius = hi_same + margin * (lo_diff - hi_same)
        else:
            self.match_radius = 0.5 * (hi_same + lo_diff)
        return self.match_radius

    def sensorium(self, z: Any) -> Any:
        """A compact map summary for a cond slot: [novelty_norm, coverage_rate,
        n_landmarks_norm]. Feed it as part of a FiLM cond vector so predictions
        are conditioned on the accumulated map, not just the current frame."""
        if not _NP_AVAILABLE:
            self.ok = False
            return None
        nov = self.novelty(z)
        cov = self.coverage()
        nov_n = 0.0 if nov == float("inf") else float(np.tanh(nov / max(self.match_radius, 1e-6)))
        return np.array([nov_n, cov["match_rate"],
                         float(np.tanh(cov["landmarks"] / 64.0))], dtype=np.float32)
