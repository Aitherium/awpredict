"""Value-guided planning (_ValueHead/_FsAdapter/value()/train_value_step) —
behavior contract.

The value head shipped with this package from day one, but carried zero
test coverage until now -- it was ported in from a solving agent's fork with
a commit message asserting "value head proven non-vacuously" against a live
checkpoint, and that claim was never captured as a repeatable test. Every
claim this file makes is asserted, not just exercised:

  * OFF by default (ARC_WM_VALUE_HEAD unset, ARC_WM_VALUE_LAMBDA=0): no head
    is built, .value() and .train_value_step() are honestly None -- the
    model is structurally identical to a build with no value-head code at
    all.
  * ON (ARC_WM_VALUE_HEAD=1): a head is built, .value(grid) returns a float,
    and .train_value_step() actually reduces loss on a FITTABLE synthetic
    target over a few steps -- a trainer that only proves its plumbing runs,
    never that anything was learned, is a worse bug than one that crashes
    (see RESULTS.md for a case where exactly that happened elsewhere in this
    program).
  * the encoder is frozen during value training: train_value_step must not
    move the encoder's own parameters (it optimises the value head only, on
    a *detached* latent per its own docstring).

CPU-only, tiny grid (16 -> Gh=4), runs in seconds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = REPO_ROOT / "packages" / "world-model"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

BASE_ENV = {
    "ARC_WM_DEVICE": "cpu", "ARC_WORLD_ENGINE": "1", "ARC_WM_GRID": "16",
    "ARC_WM_SPATIAL": "1", "ARC_WM_DCH": "8", "ARC_WM_BATCH": "8",
    "ARC_WM_FREEZE_ENCODER": "0",
}

torch = pytest.importorskip("torch")


def _fresh_model(monkeypatch, *, value_head_on: bool):
    for k in list(os.environ):
        if k.startswith(("ARC_WM_", "AITHER_WM_")):
            monkeypatch.delenv(k, raising=False)
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ARC_WM_VALUE_HEAD", "1" if value_head_on else "0")
    import importlib

    import world_model.core.lewm as lewm_mod
    importlib.reload(lewm_mod)
    m = lewm_mod.LeWorldModel(device="cpu", seed=7)
    assert m.ok, "torch present but model failed to build"
    return m


def _random_grid(gs: int = 16):
    import numpy as np
    rng = np.random.default_rng(3)
    return rng.integers(0, 10, size=(gs, gs)).tolist()


def test_value_head_off_by_default(monkeypatch):
    m = _fresh_model(monkeypatch, value_head_on=False)
    assert m.value_head is None
    grid = _random_grid()
    assert m.value(grid) is None
    ids = [[[0] * 16 for _ in range(16)]]
    assert m.train_value_step(ids, [1.0]) is None


def test_value_head_on_returns_a_float(monkeypatch):
    m = _fresh_model(monkeypatch, value_head_on=True)
    assert m.value_head is not None
    grid = _random_grid()
    v = m.value(grid)
    assert isinstance(v, float)
    assert v == v  # not NaN


def test_train_value_step_is_not_vacuous(monkeypatch):
    """A trainer that always reports success without actually fitting is a
    real failure class this program has hit before (see RESULTS.md): this
    constructs a target the head CAN fit (a constant return) and asserts
    loss actually drops, not just that the call returns a dict shape."""
    import numpy as np
    m = _fresh_model(monkeypatch, value_head_on=True)
    rng = np.random.default_rng(11)
    n = 8
    ids_batch = rng.integers(0, 10, size=(n, 16, 16))
    returns_batch = np.full(n, 5.0, dtype="float32")

    first = m.train_value_step(ids_batch, returns_batch)
    assert first is not None
    assert set(first) >= {"value_loss", "value_mean", "return_mean", "explained_var"}

    last = first
    for _ in range(30):
        last = m.train_value_step(ids_batch, returns_batch)

    assert last["value_loss"] < first["value_loss"], (
        f"value head did not fit a constant-return target over 30 steps "
        f"({first['value_loss']} -> {last['value_loss']}) -- plumbing ran, "
        f"nothing learned")
    assert last["value_mean"] == pytest.approx(5.0, abs=1.0), (
        "after fitting a constant target the head's mean prediction should "
        "land near it, not just report a lower loss number")


def test_train_value_step_freezes_the_encoder(monkeypatch):
    import numpy as np
    m = _fresh_model(monkeypatch, value_head_on=True)
    before = [p.detach().clone() for p in m.encoder.parameters()]

    rng = np.random.default_rng(5)
    ids_batch = rng.integers(0, 10, size=(4, 16, 16))
    returns_batch = np.full(4, 2.0, dtype="float32")
    for _ in range(5):
        m.train_value_step(ids_batch, returns_batch)

    after = list(m.encoder.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a), (
            "train_value_step moved an encoder parameter -- it must train "
            "the value head only, on a detached latent")
