"""C-JEPA object-level latent masking (ARC_WM_OBJECT_MASK) — behavior contract.

Every claim the slice-5 change makes is asserted here so a regression fails a
test instead of silently degrading a gated experiment:
  * flag OFF (default): no mask token exists, no masking path runs — the model
    is structurally identical to before the change;
  * flag ON: objects are segmented as 4-connected non-background components,
    frac selects a strict subset (never all, never none, for frac=0.5 & K=2),
    train_step runs, and the mask token actually receives gradient (an inert
    masking path is the slice-2 vacuous-probe failure class all over again);
  * the identity anchor: z_t handed to the loss terms is NOT the masked tensor
    (_apply_object_mask returns a new tensor and leaves its input intact).

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


def _fresh_model(monkeypatch, mask_frac: str):
    for k in list(os.environ):
        if k.startswith(("ARC_WM_", "AITHER_WM_")):
            monkeypatch.delenv(k, raising=False)
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("ARC_WM_OBJECT_MASK", mask_frac)
    import importlib

    import awpredict.core.lewm as lewm_mod
    importlib.reload(lewm_mod)
    m = lewm_mod.LeWorldModel(device="cpu", seed=7)
    assert m.ok, "torch present but model failed to build"
    return m


def _two_object_grid():
    g = [[0] * 16 for _ in range(16)]
    g[0][0] = g[0][1] = g[1][0] = 3          # L tromino, top-left
    g[12][12] = g[12][13] = g[13][12] = g[13][13] = 5  # 2x2 square, bottom-right
    return g


def test_flag_off_no_token(monkeypatch):
    m = _fresh_model(monkeypatch, "0")
    assert not hasattr(m.predictor, "obj_mask_token"), (
        "flag OFF must leave the predictor structurally unchanged"
    )


def test_segmentation_masks_subset_of_objects(monkeypatch):
    m = _fresh_model(monkeypatch, "0.5")
    ids = torch.tensor([_two_object_grid()], dtype=torch.long)
    mask = m._object_masks(ids)
    assert tuple(mask.shape) == (1, 1, 4, 4)
    # frac 0.5 of K=2 objects -> exactly ONE object masked; each object here
    # fits inside one 4x4 latent cell, so exactly one latent cell is masked.
    assert int(mask.sum().item()) == 1, (
        f"expected exactly 1 masked latent cell, got {int(mask.sum().item())} "
        "(0 = masking inert, >1 = masked both objects or bled across cells)"
    )


def test_empty_grid_masks_nothing(monkeypatch):
    m = _fresh_model(monkeypatch, "0.5")
    ids = torch.zeros(1, 16, 16, dtype=torch.long)
    assert float(m._object_masks(ids).sum()) == 0.0


def test_apply_mask_preserves_input_identity_anchor(monkeypatch):
    m = _fresh_model(monkeypatch, "0.5")
    ids = torch.tensor([_two_object_grid()], dtype=torch.long)
    z = m._encode_ids(ids)
    z_before = z.detach().clone()
    z_in = m._apply_object_mask(z, ids)
    assert not torch.equal(z_in, z), "masking changed nothing — inert path"
    assert torch.equal(z, z_before), (
        "identity anchor violated: _apply_object_mask mutated its input; "
        "d_id / SIGReg / IDM would silently train on the corrupted latent"
    )


def test_train_step_updates_mask_token(monkeypatch):
    m = _fresh_model(monkeypatch, "0.5")
    tok0 = m.predictor.obj_mask_token.detach().clone()
    g = _two_object_grid()
    for i in range(40):
        g2 = [row[:] for row in g]
        g2[i % 16][(i * 3) % 16] = (i % 5) + 1
        m.observe(g, f"ACTION{1 + i % 4}", g2, game="t")
        g = g2
    out = None
    for _ in range(5):
        out = m.train_step()
    assert out is not None and "loss" in out
    moved = float((m.predictor.obj_mask_token.detach() - tok0).abs().max())
    assert moved > 0, (
        "mask token never updated after 5 train steps — the masked input is "
        "not reaching the predictor (inert-feature class)"
    )
