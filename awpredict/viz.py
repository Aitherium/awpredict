"""Dream visualizer: see what a world model imagines, next to what really happened.

``dream_report(model, observations, actions, steps=6)`` drives any engine that
satisfies :class:`awpredict.contracts.WorldModel` two ways along a real sequence:

  * **imagination** -- encode the FIRST observation, then feed the model its own
    predictions for ``steps`` actions (open loop, never shown the real frames);
  * **one-step** -- teacher-forced: predict ``t+1`` from the REAL latent at ``t``.

Every error is printed beside the COPY-LAST baseline (predict that nothing
changes: next = current; for the open-loop rollout, every imagined state = the
seed). That baseline is the honest yardstick -- a world model that cannot beat
it has learned nothing usable -- so the report states plainly whether the model
beats it, and never implies a win it did not measure.

``render_html(report)`` turns the report into one self-contained, dark HTML page
(inline CSS/SVG/JS, no network) readable on a 390px-wide phone: the error
tables, the latent trajectory (real vs imagined) projected to 2D with a numpy
PCA, and either a filmstrip of decoded frames (engines with ``decode``, e.g.
LeWorldModel, drawn in the 16-colour ARC palette) or a heatmap of the imagined
latent vectors per step.

Engine shapes, found by duck typing (no engine is imported here):

  * step engines -- ``encode(obs) -> z`` and ``predict(z, action) -> z'``.
    An engine whose ``predict`` returns something other than a latent (the MLP
    engine returns a ``(state_hash, reward, done)`` tuple) exposes
    ``predict_latent(z, action)``, which is used instead.
  * sequence engines -- ``encode(sequence) -> [M, dim]`` and
    ``predict(history[M, dim]) -> next[dim]`` (ConceptWorldModel, recognised by
    ``concept_ids``). Each ``observation`` is one row of the sequence; each step
    is one concept, and ``actions`` are ignored.

numpy is required; torch is not imported here (engines bring their own).
"""

from __future__ import annotations

import html
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import numpy as np
    _NP_OK = True
except ImportError:  # pragma: no cover - numpy ships with the dev extra
    np = None  # type: ignore[assignment]
    _NP_OK = False

#: The ARC-AGI-3 16-colour palette (index 0 = white, index 5 = black).
ARC_PALETTE = (
    "#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
    "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
    "#FF851B", "#921231", "#4FCC30", "#A356D6",
)

#: Widest slice of a latent the heatmap draws; wider latents show their first dims.
HEATMAP_MAX_DIMS = 64


class VizUnavailableError(RuntimeError):
    """The visualizer could not run at all (numpy or torch missing)."""


# -- measurement helpers ----------------------------------------------------

def _vec(x: Any) -> Optional["np.ndarray"]:
    """A latent (torch tensor, list, ndarray) -> flat float64 array, or None."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    try:
        a = np.asarray(x, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.size == 0 or not np.all(np.isfinite(a)):
        return None
    return a


def _mse(a: Optional["np.ndarray"], b: Optional["np.ndarray"]) -> Optional[float]:
    """Mean squared difference per dimension; None when either side is missing."""
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(np.mean((a - b) ** 2))


def _cell_error(pred: Any, real: Any) -> Optional[float]:
    """Fraction of cells that differ, over the REAL grid's extent."""
    if pred is None or real is None:
        return None
    try:
        r = np.asarray(real, dtype=np.int64)
        p = np.asarray(pred, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if r.ndim != 2 or p.ndim != 2 or p.shape[0] < r.shape[0] or p.shape[1] < r.shape[1]:
        return None
    return float(np.mean(p[:r.shape[0], :r.shape[1]] != r))


def _compare(model_err: Sequence[Optional[float]],
             copy_err: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Model vs copy-last over the steps where BOTH were measured.

    ``beats_copy_last`` is True only when the model's mean error is strictly
    lower; a tie or a loss is reported as not beating it.
    """
    pairs = [(m, c) for m, c in zip(model_err, copy_err) if m is not None and c is not None]
    if not pairs:
        return {"model_err": list(model_err), "copy_last_err": list(copy_err),
                "model_mean": None, "copy_last_mean": None, "beats_copy_last": False,
                "steps_beaten": 0, "steps_measured": 0}
    m_mean = sum(p[0] for p in pairs) / len(pairs)
    c_mean = sum(p[1] for p in pairs) / len(pairs)
    return {"model_err": list(model_err), "copy_last_err": list(copy_err),
            "model_mean": m_mean, "copy_last_mean": c_mean,
            "beats_copy_last": m_mean < c_mean,
            "steps_beaten": sum(1 for m, c in pairs if m < c),
            "steps_measured": len(pairs)}


def pca_2d(rows: Sequence["np.ndarray"]) -> Dict[str, Any]:
    """Project row vectors to 2D with an SVD PCA. Returns points + explained ratios."""
    x = np.stack([np.asarray(r, dtype=np.float64) for r in rows])
    xc = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] < 2 or not np.any(xc):
        return {"points": [[0.0, 0.0] for _ in rows], "explained": [0.0, 0.0]}
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(2, vt.shape[0])
    proj = xc @ vt[:k].T
    if k < 2:
        proj = np.hstack([proj, np.zeros((proj.shape[0], 2 - k))])
    var = s ** 2
    total = float(var.sum()) or 1.0
    explained = [float(v / total) for v in var[:2]] + [0.0] * (2 - min(2, var.size))
    return {"points": proj.tolist(), "explained": explained}


# -- engine probes ------------------------------------------------------------

class _Probe:
    """How one engine is driven: real latents, and one imagined step."""

    kind = "step"

    def __init__(self, model: Any) -> None:
        self.model = model
        latent_fn = getattr(model, "predict_latent", None)
        self._predict: Callable[..., Any] = latent_fn if callable(latent_fn) else model.predict

    def real_latents(self, observations: Sequence[Any]) -> List[Any]:
        return [self.model.encode(o) for o in observations]

    def step(self, history: List[Any], action: Any) -> Any:
        return self._predict(history[-1], action)


class _SequenceProbe(_Probe):
    """Engines whose latent is a whole sequence of concepts (ConceptWorldModel)."""

    kind = "sequence"

    def real_latents(self, observations: Sequence[Any]) -> List[Any]:
        z = self.model.encode(np.asarray(observations, dtype=np.float32))
        if z is None:
            return []
        return [z[i] for i in range(int(z.shape[0]))]

    def step(self, history: List[Any], action: Any) -> Any:
        del action
        rows = [_vec(h) for h in history]
        if any(r is None for r in rows):
            return None
        return self.model.predict(np.stack(rows).astype(np.float32))


def _probe_for(model: Any) -> _Probe:
    if callable(getattr(model, "concept_ids", None)):
        return _SequenceProbe(model)
    return _Probe(model)


def _decoder(model: Any) -> Optional[Callable[[Any], Any]]:
    fn = getattr(model, "decode", None)
    return fn if callable(fn) else None


def _as_grid(x: Any) -> Optional[List[List[int]]]:
    """A decoded frame -> list of int rows, or None if it is not a 2D grid."""
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    return a.tolist() if a.ndim == 2 and a.size else None


# -- the report ---------------------------------------------------------------

def dream_report(model: Any, observations: Sequence[Any], actions: Sequence[Any],
                 steps: int = 6) -> Dict[str, Any]:
    """Roll ``model`` forward in imagination and score it against copy-last.

    Args:
        model: any engine satisfying ``awpredict.contracts.WorldModel``.
        observations: the REAL sequence, ``observations[0]`` is the seed.
            For a sequence engine, one row of the embedding sequence each.
        actions: ``actions[t]`` takes ``observations[t]`` to ``observations[t+1]``.
            Ignored by sequence engines.
        steps: how many imagined steps to roll from the seed.

    Returns:
        A JSON-serialisable dict. ``ok`` is False (with ``reason``) when the
        engine produced no rollout at all; the error blocks are always present.

    Raises:
        VizUnavailableError: numpy is not installed.
    """
    if not _NP_OK:
        raise VizUnavailableError("numpy is not installed")
    probe = _probe_for(model)
    report: Dict[str, Any] = {
        "engine": type(getattr(model, "_model", model)).__name__,
        "kind": probe.kind, "steps_requested": int(steps), "rollout_steps": 0,
        "latent_dim": None, "ok": False, "reason": "", "notes": [],
        "one_step": _compare([], []), "rollout": _compare([], []),
        "pixel": None, "pca": None, "frames": None, "heatmap": None,
    }
    if not getattr(model, "ok", False):
        report["reason"] = "engine is degraded (ok == False): torch missing or not built"
        return report
    if len(observations) < 2:
        report["reason"] = "need at least two observations (a seed and a real next step)"
        return report

    real_raw = probe.real_latents(observations)
    real = [_vec(z) for z in real_raw]
    if not real or real[0] is None:
        report["reason"] = "encode() returned no latent for the seed observation"
        return report
    dim = int(real[0].size)
    report["latent_dim"] = dim
    acts = list(actions or [])

    def action_at(t: int) -> Any:
        return acts[t] if t < len(acts) else None

    # -- one-step (teacher-forced) along the real sequence --------------------
    one_raw: List[Any] = []
    one_err: List[Optional[float]] = []
    one_copy: List[Optional[float]] = []
    for t in range(len(real) - 1):
        if probe.kind == "step" and t >= len(acts):
            break
        pred = probe.step(real_raw[:t + 1], action_at(t))
        one_raw.append(pred)
        one_err.append(_mse(_vec(pred), real[t + 1]))
        one_copy.append(_mse(real[t], real[t + 1]))
    report["one_step"] = _compare(one_err, one_copy)

    # -- open-loop imagination from the seed ----------------------------------
    n = int(steps)
    if probe.kind == "step" and len(acts) < n:
        report["notes"].append(f"only {len(acts)} actions given; rollout capped there")
        n = len(acts)
    history = [real_raw[0]]
    imagined: List["np.ndarray"] = []
    for k in range(n):
        z = probe.step(history, action_at(k))
        v = _vec(z)
        if v is None:
            report["notes"].append(f"rollout stopped at step {k + 1}: predict returned no latent")
            break
        if v.size != dim:
            report["notes"].append(
                f"rollout stopped at step {k + 1}: prediction has {v.size} dims, "
                f"the latent has {dim}")
            break
        history.append(z)
        imagined.append(v)
    report["rollout_steps"] = len(imagined)
    roll_err = [_mse(v, real[k + 1]) if k + 1 < len(real) else None
                for k, v in enumerate(imagined)]
    roll_copy = [_mse(real[0], real[k + 1]) if k + 1 < len(real) else None
                 for k in range(len(imagined))]
    report["rollout"] = _compare(roll_err, roll_copy)
    if not imagined:
        report["reason"] = "the engine produced no imagined step"
        return report
    report["ok"] = True

    # -- latent trajectory, real vs imagined, in one shared 2D projection -----
    real_shown = [r for r in real[:len(imagined) + 1] if r is not None and r.size == dim]
    pts = pca_2d(real_shown + [real[0]] + imagined)
    split = len(real_shown)
    report["pca"] = {"real": pts["points"][:split], "imagined": pts["points"][split:],
                     "explained": pts["explained"]}

    # -- what the model SEES: decoded frames, or the raw vectors ---------------
    decode = _decoder(model)
    seed_frame = _as_grid(decode(real_raw[0])) if decode is not None else None
    if decode is not None and seed_frame is not None:
        report["frames"] = {
            "real": [_as_grid(o) for o in observations[:len(imagined) + 1]],
            "reconstructed_seed": seed_frame,
            "imagined": [_as_grid(decode(z)) for z in history[1:]],
            "one_step": [_as_grid(decode(z)) for z in one_raw[:len(imagined)]],
        }
        fr = report["frames"]
        roll_px = [_cell_error(g, observations[k + 1]) if k + 1 < len(observations) else None
                   for k, g in enumerate(fr["imagined"])]
        roll_px_copy = [_cell_error(observations[0], observations[k + 1])
                        if k + 1 < len(observations) else None
                        for k in range(len(fr["imagined"]))]
        one_px = [_cell_error(_as_grid(decode(z)), observations[t + 1])
                  for t, z in enumerate(one_raw)]
        one_px_copy = [_cell_error(observations[t], observations[t + 1])
                       for t in range(len(one_raw))]
        report["pixel"] = {"one_step": _compare(one_px, one_px_copy),
                           "rollout": _compare(roll_px, roll_px_copy)}
    else:
        shown = min(dim, HEATMAP_MAX_DIMS)
        report["heatmap"] = {
            "dims_shown": shown, "dims_total": dim,
            "real": [r[:shown].tolist() for r in real_shown],
            "imagined": [real[0][:shown].tolist()] + [v[:shown].tolist() for v in imagined],
        }
    return report


# -- HTML -----------------------------------------------------------------------

_CSS = """
*{box-sizing:border-box}body{margin:0;background:#0d1117;color:#e6edf3;
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:14px}
h1{font-size:1.25rem;margin:.2rem 0}h2{font-size:1.05rem;margin:1.4rem 0 .5rem;color:#9fd8ff}
.sub{color:#8b949e;font-size:.85rem}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px;margin:10px 0}
.verdict{font-weight:600}.win{color:#4fcc30}.loss{color:#f93c31}
.nums{display:flex;flex-wrap:wrap;gap:8px}.num{flex:1 1 140px;background:#0d1117;
border-radius:8px;padding:8px}.num b{display:block;font-size:1.1rem}
.err-model b{color:#ffdc00}.err-copy b{color:#88d8f1}
table{border-collapse:collapse;width:100%;font-size:.82rem}
td,th{padding:3px 6px;text-align:right;border-bottom:1px solid #21262d}
th:first-child,td:first-child{text-align:left}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.strip{display:flex;gap:6px;padding-bottom:4px}
.frame{flex:0 0 auto;text-align:center;font-size:.72rem;color:#8b949e}
canvas.px{width:72px;height:72px;image-rendering:pixelated;border:1px solid #30363d;
border-radius:4px;display:block;margin-bottom:2px}
.rowlab{font-size:.8rem;color:#c9d1d9;margin:8px 0 4px}
svg{width:100%;height:auto;display:block}
.legend span{display:inline-block;margin-right:12px;font-size:.8rem}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
ul{padding-left:18px}
"""

_JS = """
(function(){var P=%s;document.querySelectorAll('canvas.px').forEach(function(c){
var g=JSON.parse(c.getAttribute('data-grid'));if(!g||!g.length)return;
var h=g.length,w=g[0].length;c.width=w;c.height=h;var x=c.getContext('2d');
for(var i=0;i<h;i++)for(var j=0;j<w;j++){x.fillStyle=P[g[i][j]&15]||'#000';
x.fillRect(j,i,1,1);}});})();
"""


def _fmt(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.4g}"


def _verdict_html(title: str, block: Dict[str, Any], unit: str) -> str:
    """One card: model error vs copy-last, and a verdict that never overclaims."""
    if not block.get("steps_measured"):
        line = '<span class="verdict loss">not measured (no comparable steps)</span>'
    elif block["beats_copy_last"]:
        line = (f'<span class="verdict win">beats copy-last</span> on average '
                f'({block["steps_beaten"]}/{block["steps_measured"]} steps lower)')
    else:
        line = (f'<span class="verdict loss">does NOT beat copy-last</span> '
                f'({block["steps_beaten"]}/{block["steps_measured"]} steps lower)')
    rows = "".join(
        f"<tr><td>t+{i + 1}</td><td>{_fmt(m)}</td><td>{_fmt(c)}</td></tr>"
        for i, (m, c) in enumerate(zip(block["model_err"], block["copy_last_err"])))
    return (f'<div class="card"><div class="rowlab">{html.escape(title)}</div>'
            f'<div class="nums"><div class="num err-model">model {html.escape(unit)}'
            f'<b>{_fmt(block["model_mean"])}</b></div>'
            f'<div class="num err-copy">copy-last {html.escape(unit)}'
            f'<b>{_fmt(block["copy_last_mean"])}</b></div></div><p>{line}</p>'
            f'<div class="scroll"><table><tr><th>step</th><th>model</th><th>copy-last</th>'
            f'</tr>{rows}</table></div></div>')


def _pca_svg(pca: Dict[str, Any]) -> str:
    real, imag = pca["real"], pca["imagined"]
    pts = real + imag
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w, h, pad = 360.0, 240.0, 24.0
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0

    def sx(v: float) -> float:
        return pad + (v - min(xs)) / span_x * (w - 2 * pad)

    def sy(v: float) -> float:
        return h - pad - (v - min(ys)) / span_y * (h - 2 * pad)

    def path(seq: List[List[float]], colour: str, dash: str) -> str:
        if not seq:
            return ""
        d = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in seq)
        dots = "".join(
            f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="4" fill="{colour}"/>'
            f'<text x="{sx(p[0]) + 6:.1f}" y="{sy(p[1]) - 6:.1f}" fill="{colour}" '
            f'font-size="11">{i}</text>' for i, p in enumerate(seq))
        return (f'<polyline points="{d}" fill="none" stroke="{colour}" stroke-width="2" '
                f'stroke-dasharray="{dash}"/>{dots}')

    ex = pca["explained"]
    return (f'<div class="card" id="pca"><div class="legend">'
            f'<span><i class="dot" style="background:#88d8f1"></i>real</span>'
            f'<span><i class="dot" style="background:#ffdc00"></i>imagined</span></div>'
            f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="latent PCA">'
            f'<rect width="{w:.0f}" height="{h:.0f}" fill="#0d1117" rx="8"/>'
            f'{path(real, "#88d8f1", "")}{path(imag, "#ffdc00", "5 4")}</svg>'
            f'<div class="sub">PC1 {ex[0] * 100:.1f}% + PC2 {ex[1] * 100:.1f}% of the variance '
            f'across these points; labels are step numbers, 0 = the seed.</div></div>')


def _canvas(grid: Optional[List[List[int]]], label: str) -> str:
    data = html.escape(json.dumps(grid if grid is not None else []), quote=True)
    return (f'<div class="frame"><canvas class="px" width="1" height="1" '
            f'data-grid="{data}"></canvas>{html.escape(label)}</div>')


def _filmstrip(frames: Dict[str, Any]) -> str:
    def row(title: str, grids: List[Any], start: int) -> str:
        cells = "".join(_canvas(g, f"t{start + i}") for i, g in enumerate(grids))
        return (f'<div class="rowlab">{html.escape(title)}</div>'
                f'<div class="scroll"><div class="strip">{cells}</div></div>')

    return ('<div class="card" id="filmstrip">'
            + row("real frames", frames["real"], 0)
            + row("imagined (open loop from t0, its own predictions fed back)",
                  [frames["reconstructed_seed"]] + frames["imagined"], 0)
            + row("one-step (teacher-forced from the real t-1)", frames["one_step"], 1)
            + '<div class="sub">The imagined row starts with the seed as the decoder '
              'reconstructs it.</div></div>')


def _heat_svg(rows: List[List[float]], vmax: float, title: str) -> str:
    if not rows:
        return ""
    n_r, n_c = len(rows), len(rows[0])
    cell, row_h, lab = 8.0, 22.0, 40.0     # tall rows: 64 dims must still read at 390px
    w, h = lab + n_c * cell, n_r * row_h
    rects = []
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            a = max(-1.0, min(1.0, v / vmax)) if vmax else 0.0
            colour = (f"rgb({int(255 * a)},{int(60 * a)},{int(40 * a)})" if a >= 0 else
                      f"rgb({int(-30 * a)},{int(-120 * a)},{int(-255 * a)})")
            rects.append(f'<rect x="{lab + j * cell:.0f}" y="{i * row_h:.0f}" '
                         f'width="{cell:.0f}" height="{row_h:.0f}" fill="{colour}"/>')
        rects.append(f'<text x="2" y="{i * row_h + row_h - 6:.0f}" fill="#c9d1d9" '
                     f'font-size="16">t{i}</text>')
    return (f'<div class="rowlab">{html.escape(title)}</div>'
            f'<svg viewBox="0 0 {w:.0f} {h:.0f}">{"".join(rects)}</svg>')


def _heatmap(hm: Dict[str, Any]) -> str:
    allv = [abs(v) for r in hm["real"] + hm["imagined"] for v in r]
    vmax = max(allv) if allv else 1.0
    return ('<div class="card" id="heatmap">'
            + _heat_svg(hm["imagined"], vmax, "imagined latent per step (open loop)")
            + _heat_svg(hm["real"], vmax, "real latent per step")
            + f'<div class="sub">First {hm["dims_shown"]} of {hm["dims_total"]} latent dims; '
              f'red positive, blue negative, shared scale |v| &le; {_fmt(vmax)}. '
              'This engine cannot decode to a grid, so its imagination is shown as '
              'the raw vectors.</div></div>')


def render_html(report: Dict[str, Any], title: Optional[str] = None) -> str:
    """Render a :func:`dream_report` dict as one self-contained dark HTML page."""
    engine = html.escape(str(report.get("engine", "?")))
    head = html.escape(title or f"awpredict dream: {report.get('engine', '?')}")
    meta = report.get("meta") or {}
    meta_html = "".join(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>"
                        for k, v in meta.items())
    parts = [f'<h1>{head}</h1><div class="sub">engine {engine} &middot; '
             f'{report.get("rollout_steps", 0)} imagined steps of '
             f'{report.get("steps_requested", 0)} requested &middot; latent dim '
             f'{report.get("latent_dim")}</div>']
    if meta_html:
        parts.append(f'<div class="card"><ul>{meta_html}</ul></div>')
    if not report.get("ok"):
        parts.append(f'<div class="card"><span class="verdict loss">no rollout</span>: '
                     f'{html.escape(str(report.get("reason", "")))}</div>')
    parts.append("<h2>Prediction error vs copy-last</h2>"
                 '<div class="sub">Copy-last predicts that nothing changes. A model that '
                 'does not beat it has not learned the dynamics.</div>')
    parts.append(_verdict_html("one-step (teacher-forced), latent MSE",
                               report["one_step"], "MSE"))
    parts.append(_verdict_html("imagination (open loop from the seed), latent MSE",
                               report["rollout"], "MSE"))
    px = report.get("pixel")
    if px:
        parts.append(_verdict_html("one-step, decoded cells wrong", px["one_step"],
                                   "cell error"))
        parts.append(_verdict_html("imagination, decoded cells wrong", px["rollout"],
                                   "cell error"))
    if report.get("frames"):
        parts.append("<h2>Dream filmstrip</h2>" + _filmstrip(report["frames"]))
    if report.get("heatmap"):
        parts.append("<h2>Imagined vectors</h2>" + _heatmap(report["heatmap"]))
    if report.get("pca"):
        parts.append("<h2>Latent trajectory (PCA)</h2>" + _pca_svg(report["pca"]))
    if report.get("notes"):
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in report["notes"])
        parts.append(f'<h2>Notes</h2><div class="card"><ul>{notes}</ul></div>')
    js = _JS % json.dumps(list(ARC_PALETTE)) if report.get("frames") else ""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{head}</title><style>{_CSS}</style></head><body><main>'
            + "".join(parts) + f'</main><script>{js}</script></body></html>')
