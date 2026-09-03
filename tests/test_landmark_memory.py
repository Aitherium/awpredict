"""MapMemory: the growing landmark map — localize / integrate / novelty / coverage."""

import numpy as np
import pytest
from awpredict.memory import MapMemory

np.random.seed(7)


def _clusters(k: int = 4, per: int = 6, dim: int = 16) -> list:
    """k well-separated clusters of per points in dim-d space."""
    out = []
    for c in range(k):
        center = np.random.randn(dim) * 8.0
        out.extend(center + np.random.randn(per, dim) * 0.4)
    return out


def test_integrate_merges_near_and_splits_far():
    mem = MapMemory(match_radius=25.0, ema=0.2)
    a, b = np.zeros(8), np.ones(8) * 10.0        # 100 apart -> different regions
    i1 = mem.integrate(a)
    i2 = mem.integrate(a + 0.01)                 # near a -> SAME landmark
    i3 = mem.integrate(b)
    assert i1 == i2
    assert i3 != i1
    assert len(mem.landmarks) == 2


def test_novelty_is_frontier_signal():
    mem = MapMemory(match_radius=1.0)
    z0 = np.zeros(8)
    mem.integrate(z0)
    near = mem.novelty(z0 + 0.1)
    far = mem.novelty(np.ones(8) * 5.0)
    assert near < far
    assert mem.novelty(np.zeros(8)) < float("inf")


def test_empty_map_novelty_is_inf():
    mem = MapMemory()
    assert mem.localize(np.zeros(4)) == (-1, float("inf"))
    assert mem.novelty(np.zeros(4)) == float("inf")


def test_coverage_rises_as_space_exhausts():
    mem = MapMemory(match_radius=25.0)
    z = np.zeros(8)
    for _ in range(10):
        mem.integrate(z + np.random.randn(8) * 0.5)   # new region, then hits
    c1 = mem.coverage()
    for _ in range(20):
        mem.integrate(z + np.random.randn(8) * 0.5)
    c2 = mem.coverage()
    assert c1["landmarks"] >= 1
    assert c2["match_rate"] >= c1["match_rate"]       # hits dominate as it maps


def test_ema_updates_landmark_toward_family():
    mem = MapMemory(match_radius=25.0, ema=0.5)
    z0 = np.zeros(8)
    idx = mem.integrate(z0)
    drift = z0 + np.ones(8) * 1.0                      # dist2 = 8 < 25 -> EMA merge
    mem.integrate(drift)
    assert np.allclose(mem.landmarks[idx], np.ones(8) * 0.5, atol=1e-5)  # halfway
    assert mem.counts[idx] == 2


def test_calibrate_separates_regimes():
    mem = MapMemory()
    same = [(np.zeros(8), np.zeros(8) + 0.1), (np.ones(8), np.ones(8) + 0.1)]
    diff = [(np.zeros(8), np.ones(8) * 10.0)]
    r = mem.calibrate(same, diff, margin=0.5)
    assert 0.08 < r < 10.0 * 10.0 * 8                  # between same and diff regimes


def test_accepts_lists_and_torch_tensors():
    mem = MapMemory(match_radius=1.0)
    idx = mem.integrate([0.0] * 4)                     # plain list
    assert idx == 0
    class FakeTensor:                                  # .detach().cpu().numpy() shape
        def __init__(self, v):
            self._v = np.asarray(v, dtype=np.float32)
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self._v
    assert mem.novelty(FakeTensor([1.0, 0, 0, 0])) < float("inf")


@pytest.mark.parametrize("dim", [8, 32])
def test_cluster_separation(dim):
    """Points from k distinct clusters settle into ~k landmarks."""
    # within-cluster dist2 ~ 0.32*dim, between-cluster ~ 128*dim
    mem = MapMemory(match_radius=2.0 * dim)
    for z in _clusters(dim=dim):
        mem.integrate(z)
    assert len(mem.landmarks) <= 6
    assert len(mem.landmarks) >= 3
