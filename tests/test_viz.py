"""Dream visualizer -- report shape, the copy-last yardstick, the HTML, the CLI.

The copy-last baseline is the claim this module exists to make honestly, so it is
pinned by hand on a sequence small enough to compute on paper. Engines here are
pure-numpy fakes unless a test is marked ``needs_torch``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from awpredict import cli, viz, viz_demo

_HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


class _AddEngine:
    """latent = the observation vector; predict(z, a) = z + a (exact dynamics)."""

    ok = True

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = bias

    def encode(self, obs, cond=None):
        return np.asarray(obs, dtype=float)

    def predict(self, z, action, **kwargs):
        return np.asarray(z, dtype=float) + np.asarray(action, dtype=float) + self.bias

    def observe(self, *a, **k):
        return None

    def surprise(self, *a, **k):
        return None

    def train_step(self, *a, **k):
        return None

    def plan(self, *a, **k):
        return None

    def save(self, path, *a, **k):
        return False

    def load(self, path, *a, **k):
        return False


class _GridEngine(_AddEngine):
    """A 1x4 'grid' whose single lit cell moves right; decode is exact."""

    def encode(self, obs, cond=None):
        return np.asarray(obs, dtype=float).reshape(-1)

    def predict(self, z, action, **kwargs):
        return np.roll(np.asarray(z, dtype=float), 1)

    def decode(self, z):
        return np.rint(np.asarray(z)).astype(int).reshape(1, -1).tolist()


class _DeadEngine(_AddEngine):
    def predict(self, z, action, **kwargs):
        return None


# Hand-made: x moves 0 -> 1 -> 3. Copy-last one-step MSE per dim (2 dims):
#   t0->t1: ((1-0)^2 + 0)/2 = 0.5 ; t1->t2: ((3-1)^2 + 0)/2 = 2.0 ; mean 1.25.
# Copy-last for the open-loop rollout compares the SEED to every later state:
#   r0 vs r1 = 0.5 ; r0 vs r2 = 9/2 = 4.5.
OBS = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]
ACTS = [[1.0, 0.0], [2.0, 0.0]]


def test_copy_last_baseline_is_computed_by_hand():
    r = viz.dream_report(_AddEngine(), OBS, ACTS, steps=2)
    assert r["one_step"]["copy_last_err"] == [0.5, 2.0]
    assert r["one_step"]["copy_last_mean"] == pytest.approx(1.25)
    assert r["rollout"]["copy_last_err"] == [0.5, 4.5]
    # the exact engine has zero error, so it genuinely beats copy-last
    assert r["one_step"]["model_err"] == [0.0, 0.0]
    assert r["rollout"]["model_err"] == [0.0, 0.0]
    assert r["one_step"]["beats_copy_last"] and r["rollout"]["beats_copy_last"]


def test_a_worse_model_is_never_reported_as_beating_copy_last():
    r = viz.dream_report(_AddEngine(bias=5.0), OBS, ACTS, steps=2)
    assert r["one_step"]["model_mean"] > r["one_step"]["copy_last_mean"]
    assert r["one_step"]["beats_copy_last"] is False
    assert r["one_step"]["steps_beaten"] == 0
    assert "does NOT beat copy-last" in viz.render_html(r)


def test_report_shape_for_a_vector_engine():
    r = viz.dream_report(_AddEngine(), OBS, ACTS, steps=6)
    assert r["ok"] and r["kind"] == "step" and r["latent_dim"] == 2
    assert r["rollout_steps"] == 2          # capped at the two actions given
    assert any("capped" in n for n in r["notes"])
    assert len(r["pca"]["real"]) == 3 and len(r["pca"]["imagined"]) == 3
    assert r["frames"] is None and r["pixel"] is None
    hm = r["heatmap"]
    assert hm["dims_total"] == 2 and len(hm["imagined"]) == 3
    import json
    json.dumps(r)                            # the report is plain data


def test_decodable_engine_gets_a_filmstrip_and_cell_errors():
    obs = [[[1, 0, 0, 0]], [[0, 1, 0, 0]], [[0, 0, 1, 0]], [[0, 0, 0, 1]]]
    r = viz.dream_report(_GridEngine(), obs, ["r", "r", "r"], steps=3)
    assert r["ok"] and r["heatmap"] is None
    assert r["frames"]["imagined"] == obs[1:]
    assert r["pixel"]["rollout"]["model_err"] == [0.0, 0.0, 0.0]
    # copy-last: the lit cell has moved, so 2 of 4 cells differ at every horizon
    assert r["pixel"]["rollout"]["copy_last_err"] == [0.5, 0.5, 0.5]
    page = viz.render_html(r)
    assert 'id="filmstrip"' in page and "canvas" in page
    assert ARC_FIRST_COLOUR in page          # palette inlined, no CDN
    assert "http://" not in page and "https://" not in page


ARC_FIRST_COLOUR = viz.ARC_PALETTE[0]


def test_html_carries_heatmap_and_both_error_numbers():
    r = viz.dream_report(_AddEngine(), OBS, ACTS, steps=2)
    page = viz.render_html(r)
    assert 'id="heatmap"' in page and 'id="pca"' in page
    assert 'class="num err-model"' in page and 'class="num err-copy"' in page
    assert "1.25" in page                    # the copy-last mean, printed
    assert "width=device-width" in page      # phone-readable viewport


def test_no_rollout_is_not_ok():
    r = viz.dream_report(_DeadEngine(), OBS, ACTS, steps=2)
    assert r["ok"] is False and r["rollout_steps"] == 0
    assert "no rollout" in viz.render_html(r)


def test_degraded_engine_is_not_ok():
    eng = _AddEngine()
    eng.ok = False
    r = viz.dream_report(eng, OBS, ACTS)
    assert r["ok"] is False and "degraded" in r["reason"]


def test_pca_projects_a_line_onto_one_axis():
    rows = [np.array([i, 2.0 * i, 0.0]) for i in range(5)]
    p = viz.pca_2d(rows)
    assert p["explained"][0] == pytest.approx(1.0)
    xs = [pt[0] for pt in p["points"]]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True)


# -- CLI exit codes -------------------------------------------------------------

def _fake_demo(engine):
    def build(steps=6):
        return engine, OBS, ACTS, {"world": "hand-made"}
    return build


def test_cli_writes_html_and_exits_0(tmp_path, monkeypatch):
    monkeypatch.setitem(viz_demo.DEMOS, "mlp", _fake_demo(_AddEngine()))
    out = tmp_path / "d.html"
    assert cli.main(["viz", "--engine", "mlp", "--demo", "--out", str(out)]) == 0
    assert 'id="heatmap"' in out.read_text(encoding="utf-8")


def test_cli_exits_1_when_the_engine_cannot_roll_out(tmp_path, monkeypatch):
    monkeypatch.setitem(viz_demo.DEMOS, "mlp", _fake_demo(_DeadEngine()))
    out = tmp_path / "d.html"
    assert cli.main(["viz", "--engine", "mlp", "--demo", "--out", str(out)]) == 1
    assert not out.exists()


def test_cli_exits_2_when_it_cannot_run(tmp_path, monkeypatch):
    def unavailable(steps=6):
        raise viz.VizUnavailableError("torch is not installed")
    monkeypatch.setitem(viz_demo.DEMOS, "mlp", unavailable)
    out = tmp_path / "d.html"
    assert cli.main(["viz", "--engine", "mlp", "--demo", "--out", str(out)]) == 2
    assert not out.exists()
    assert cli.main(["viz", "--engine", "mlp"]) == 2        # no data source


@pytest.mark.skipif(_HAS_TORCH, reason="torch is installed; the real demo can run")
def test_cli_real_demo_exits_2_without_torch(tmp_path):
    assert cli.main(["viz", "--engine", "concept", "--demo",
                     "--out", str(tmp_path / "d.html")]) == 2


# -- real engines (torch) -------------------------------------------------------

@needs_torch
def test_mlp_demo_dreams_and_beats_copy_last(tmp_path):
    out = tmp_path / "mlp.html"
    assert cli.main(["viz", "--engine", "mlp", "--demo", "--out", str(out)]) == 0
    model, obs, acts, _ = viz_demo.mlp_demo()
    r = viz.dream_report(model, obs, acts)
    assert r["ok"] and r["rollout_steps"] == 6 and r["latent_dim"] == 768
    assert r["one_step"]["beats_copy_last"]


@needs_torch
def test_concept_demo_rolls_out_a_heatmap():
    model, obs, acts, _ = viz_demo.concept_demo(train_steps=20)
    r = viz.dream_report(model, obs, acts, steps=4)
    assert r["ok"] and r["kind"] == "sequence" and r["rollout_steps"] == 4
    assert r["heatmap"] is not None and r["frames"] is None


@needs_torch
def test_lewm_decodes_a_filmstrip():
    model, obs, acts, _ = viz_demo.lewm_demo(steps=3, train_steps=5)
    r = viz.dream_report(model, obs, acts, steps=3)
    assert r["ok"] and r["rollout_steps"] == 3
    assert len(r["frames"]["imagined"]) == 3
    assert len(r["frames"]["imagined"][0]) == 16       # a 16x16 decoded grid
    assert r["pixel"]["rollout"]["steps_measured"] == 3
    assert 'id="filmstrip"' in viz.render_html(r)
