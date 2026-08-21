"""OnlineLRController behavior contract (slice 7, WM_ONLINE_LR_ENABLED).

Asserts the flag-off no-op, the direction and clamping of both control laws,
and restore() — so a regression fails a test instead of silently destabilizing
a training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

torch = pytest.importorskip("torch")

from world_model.training.online import OnlineLRController  # noqa: E402


class _FakeModel:
    def __init__(self, lr=0.001, sig_lambda=0.05):
        self.lr = lr
        self.sig_lambda = sig_lambda
        p = torch.nn.Parameter(torch.zeros(2))
        self._opt = torch.optim.Adam([p], lr=lr)


def _lr(model):
    return model._opt.param_groups[0]["lr"]


def test_flag_off_is_noop(monkeypatch):
    monkeypatch.setenv("WM_ONLINE_LR_ENABLED", "0")
    m = _FakeModel()
    ctl = OnlineLRController(m)
    out = ctl.after_step({"mse": 5.0}, torch.zeros(8, 4))
    assert out is None
    assert _lr(m) == 0.001
    assert m.sig_lambda == 0.05


def test_surprise_raises_lr_and_calm_lowers_it():
    m = _FakeModel()
    ctl = OnlineLRController(m, enabled=True)
    ctl.after_step({"mse": 1.0})           # seeds the EMA at 1.0
    out = ctl.after_step({"mse": 3.0})     # 3x the EMA -> surprise
    assert out["lr"] > 0.001, "surprise must raise the LR"
    # long calm stretch: relative surprise < 1 -> lr under base
    for _ in range(50):
        out = ctl.after_step({"mse": 0.01})
    assert out["lr"] < 0.001, "sustained calm must cool the LR below base"


def test_lr_clamped_both_sides():
    m = _FakeModel()
    ctl = OnlineLRController(m, enabled=True)
    ctl.after_step({"mse": 1.0})
    out = ctl.after_step({"mse": 1e9})     # absurd spike
    assert out["lr"] <= 0.001 * ctl.lr_max_x + 1e-12
    for _ in range(200):
        out = ctl.after_step({"mse": 1e-9})
    assert out["lr"] >= 0.001 * ctl.lr_min_x - 1e-12


def test_homeostat_direction_and_clamp():
    m = _FakeModel(sig_lambda=0.05)
    ctl = OnlineLRController(m, enabled=True)
    collapsing = torch.zeros(16, 8) + 0.01 * torch.randn(16, 8)   # var ~1e-4
    ctl.after_step({"mse": 1.0}, collapsing)
    assert m.sig_lambda > 0.05, "low variance (collapse) must RAISE sig_lambda"
    over = 10.0 * torch.randn(64, 8)                              # var ~100
    for _ in range(400):
        ctl.after_step({"mse": 1.0}, over)
    assert m.sig_lambda >= 0.1 * 0.05 - 1e-12, "sig_lambda clamped at 0.1x base"
    assert m.sig_lambda < 0.05, "high variance must LOWER sig_lambda"


def test_restore_resets_both():
    m = _FakeModel()
    ctl = OnlineLRController(m, enabled=True)
    ctl.after_step({"mse": 1.0})
    ctl.after_step({"mse": 9.0}, torch.zeros(8, 4))
    assert _lr(m) != 0.001 or m.sig_lambda != 0.05
    ctl.restore()
    assert _lr(m) == 0.001
    assert m.sig_lambda == 0.05


def test_none_metrics_and_nan_ignored():
    m = _FakeModel()
    ctl = OnlineLRController(m, enabled=True)
    assert ctl.after_step(None) is None
    assert ctl.after_step({"mse": float("nan")}) is None
    assert _lr(m) == 0.001
