"""The explorer protocol: budgeted exploration with planted needles, gated like
the published result — curiosity must beat the blind baselines by >= 1.15x on
coverage, and win on external needle-reach. The encoder here is the cold-start
HistogramEncoder, so this tests the MAP + POLICIES machinery, not a trained
encoder (the trained-engine claim is a probe-script measurement).

Discipline carried over from the audit the blog post documents:
  * needles are NEVER planted in cache/vendor trees (they classify as caches);
  * every policy shares the SAME observation stream and budget;
  * random is seeded, so the comparison is reproducible.
"""

import os

import numpy as np
from awpredict.adapters.obs_adapter import StructuredAdapter
from awpredict.explorer import FsCartographer, observe_dir
from awpredict.explorer.fs_cartographer import _CACHE_DIRS
from awpredict.explorer.histogram_encoder import HistogramEncoder
from awpredict.memory import MapMemory

KINDS = {"code": ".py", "data": ".json", "media": ".png", "doc": ".md",
         "config": ".yaml"}
OTHER = {"code": "data", "data": "doc", "media": "doc", "doc": "data",
         "config": "code"}
MASS_DIRS = 160                # the homogeneous long tail (see build_tree)


def _write(d: str, name: str, n_bytes: int) -> None:
    with open(os.path.join(d, name), "wb") as fh:
        fh.write(b"\0" * n_bytes)


def _leaf_files(leaf: str, kind: str, variant: int, rng) -> None:
    """One leaf dir. Variant 0 is dominant-heavy (4 dominant + 1 other), variant
    1 is other-heavy (1 dominant + 4 other) — a real structural difference, so
    cross-variant dirs are further apart than same-variant ones. File sizes are
    >= 2x the 1000-byte dir-block weight the adapter's peek uses, so a dir's
    dominant type never ties with its subdir blocks (ties resolve by scandir
    order, which is filesystem-dependent — a tie would make the map's structure
    machine-dependent)."""
    if variant == 0:
        for i in range(4):
            _write(leaf, f"{kind}{i}.{KINDS[kind]}", int(rng.integers(2000, 6000)))
        _write(leaf, f"o.{KINDS[OTHER[kind]]}", int(rng.integers(1000, 2500)))
    else:
        _write(leaf, f"{kind}0.{KINDS[kind]}", int(rng.integers(2000, 4000)))
        for i in range(4):
            _write(leaf, f"o{i}.{KINDS[OTHER[kind]]}", int(rng.integers(2000, 5000)))


def build_tree(root: str, seed: int = 0) -> dict:
    """A tree shaped like a real repo, with the mechanism the gate depends on:

    * files at EVERY level (a dir's children are files AND subdirs) — without
      this the observation is undecodable and the map collapses (the blog's own
      "decodable observation" lesson);
    * 5 'kinds of place', each = a top dir with 2 structurally-distinct variant
      subdirs (dominant-heavy vs other-heavy), each with 3 leaves;
    * a HOMOGENEOUS MASS: 160 near-identical code dirs that merge into ONE
      landmark — blind search keeps re-visiting them, curiosity moves on;
    * a RARE deep region (secret, doc-heavy parents -> pure-secret leaves) that
      holds the planted needles;
    * a vendored cache tree (node_modules) that is not project content and can
      never hold a needle.
    """
    rng = np.random.default_rng(seed)
    planted = []
    for kind, ext in KINDS.items():
        base = os.path.join(root, kind)
        os.makedirs(base)
        _write(base, f"top.{ext}", 5000)
        _write(base, f"top2.{ext}", 5000)
        for j in (0, 1):
            sub = os.path.join(base, f"d{j}")
            os.makedirs(sub)
            _write(sub, f"v{j}.{KINDS[kind]}", 5000)         # > dir-block weight
            _write(sub, f"v{j}o.{KINDS[OTHER[kind]]}", 2000)
            for k in range(3):
                leaf = os.path.join(sub, f"l{k}")
                os.makedirs(leaf)
                _leaf_files(leaf, kind, j, rng)
    # the rare deep region: doc-heavy parents, a PURE-secret leaf with the needles
    sb = os.path.join(root, "secret")
    os.makedirs(sb)
    for j in (0, 1):
        dj = os.path.join(sb, f"d{j}")
        os.makedirs(dj)
        _write(dj, "s.pem", 2000)
        _write(dj, "s.docx", 4000)                           # doc-heavy mix
    deep = os.path.join(sb, "d1", "deep")
    os.makedirs(deep)
    for name in ("token.pem", "key.pem"):
        with open(os.path.join(deep, name), "wb") as fh:
            fh.write(b"\1" * 64)
        planted.append(os.path.join(deep, name))
    # the MASS: near-identical code dirs -> one landmark
    mass = os.path.join(root, "lib")
    os.makedirs(mass)
    _write(mass, "top.py", 5000)
    for j in range(MASS_DIRS):
        md = os.path.join(mass, f"m{j}")
        os.makedirs(md)
        for i in range(4):
            _write(md, f"c{i}.py", int(rng.integers(2000, 4000)))
        _write(md, "d0.json", int(rng.integers(800, 1500)))
    # vendored cache tree: never a needle target
    vendor = os.path.join(root, "node_modules")
    os.makedirs(vendor)
    _write(vendor, "m.bin", 5000)
    return {"planted": planted}


class FakeForwardModel:
    """Cold-start engine stand-in: histogram encoder + an identity forward model
    ('nothing changes'), which is exactly what an untrained engine predicts."""

    ok = True

    def __init__(self, enc: HistogramEncoder) -> None:
        self._enc = enc

    def encode(self, grid, cond=None):
        return self._enc.encode(grid, cond)

    def predict(self, z, action, **kwargs):
        return z


class PurposeCond:
    """A reading stub for the cond seam: 16-dim vector, purpose one-hot in
    dims 0..12 (path -> its top-level kind), depth-likelihood dim 14, novelty
    dim 15. Enough for the llm ablation and the planner's purpose histogram."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.order = {k: i for i, k in enumerate(KINDS)}

    def __call__(self, path):
        rel = os.path.relpath(path, self.root).split(os.sep)
        kind = rel[0] if rel[0] in self.order else "unknown"
        v = np.zeros(16, dtype=np.float32)
        if kind in self.order:
            v[self.order[kind]] = 1.0
        v[14] = 0.5 if len(rel) > 1 else 0.1            # deeper = likelier depth
        v[15] = 0.3
        return v


def _setup(tmp_path, budget=60):
    root = os.path.join(str(tmp_path), "t")
    os.makedirs(root)
    info = build_tree(root, seed=3)
    enc = HistogramEncoder(n_categories=16, dim=64)
    adapter = StructuredAdapter(grid=32)
    model = FakeForwardModel(enc)

    # match_radius from the 18th percentile of REAL pairwise dir distances over
    # a breadth-first sample (the fork's own recipe): the closest pairs are
    # same-region dirs (which must merge) and the 18th pctl sits inside that
    # band — far below the cross-region distances that must stay separate.
    # A fixed radius is meaningless across trees; this self-tunes to the tree.
    sample = []
    q = [root]
    while q and len(sample) < 60:
        d = q.pop(0)
        if os.path.basename(d) in _CACHE_DIRS:
            continue
        sample.append(d)
        try:
            q.extend(e.path for e in os.scandir(d) if e.is_dir())
        except OSError:
            continue
    vecs = []
    for d in sample:
        z = enc.encode(adapter.token_grid(observe_dir(d)))
        if z is not None:
            vecs.append(z)
    dists = [float(((vecs[i] - vecs[j]) ** 2).sum())
             for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    radius = float(np.percentile(dists, 18)) if dists else 10000.0
    mem = MapMemory(match_radius=radius, ema=0.15)
    return root, info, enc, adapter, mem, model


def _run(root, mem, model, adapter, policy, budget, seed=0):
    m = MapMemory(match_radius=mem.match_radius, ema=0.15)
    carto = FsCartographer(model, adapter, m)
    return carto.explore(root, budget, policy=policy, seed=seed)


def _reach(result, info):
    """(reach_rate, median_actions) over the planted needles, censored at budget."""
    found = {p: None for p in info["planted"]}
    for i, (path, _idx, _nov) in enumerate(result["visited"]):
        for p in list(found):
            if found[p] is None and os.path.abspath(p).startswith(os.path.abspath(path)):
                found[p] = i + 1
    times = [t if t is not None else result["budget"] + 1 for t in found.values()]
    return sum(1 for t in times if t <= result["budget"]) / len(times), sorted(times)


def test_curious_beats_blind_baselines_on_coverage(tmp_path):
    """The gate from the published protocol: >= 1.15x the blind-search coverage
    on the SAME budget and observation stream."""
    root, info, _enc, adapter, mem, model = _setup(tmp_path)
    budget = 60
    curious = _run(root, mem, model, adapter, "curious", budget)
    random = [_run(root, mem, model, adapter, "random", budget, seed=s)
              for s in range(5)]
    bfs = _run(root, mem, model, adapter, "bfs", budget)
    rand_lm = np.mean([r["n_landmarks"] for r in random])
    assert curious["n_landmarks"] >= 1.15 * rand_lm, (
        f"curious {curious['n_landmarks']} vs random {rand_lm:.1f}")
    assert curious["n_landmarks"] >= 1.15 * bfs["n_landmarks"], (
        f"curious {curious['n_landmarks']} vs bfs {bfs['n_landmarks']}")


def test_curious_wins_needle_reach(tmp_path):
    """External ground truth the representation cannot game: planted needles in
    the rarest region. Curiosity must reach them at least as often as random."""
    root, info, _enc, adapter, mem, model = _setup(tmp_path)
    budget = 60
    curious = _run(root, mem, model, adapter, "curious", budget)
    random = _run(root, mem, model, adapter, "random", budget, seed=1)
    c_rate, c_times = _reach(curious, info)
    r_rate, r_times = _reach(random, info)
    assert len(info["planted"]) == 2
    assert c_rate >= r_rate, f"curious reach {c_rate} vs random {r_rate}"
    assert c_times[0] <= r_times[0], f"first needle {c_times[0]} vs random {r_times[0]}"


def test_needles_never_plant_in_vendor_trees(tmp_path):
    """The audit rule: vendored toolchain dirs classify as caches and cannot
    hold needles — nothing (agent or human) should crawl there."""
    root, info, _enc, adapter, mem, model = _setup(tmp_path)
    for p in info["planted"]:
        assert "node_modules" not in p.split(os.sep)
    # and the classifier agrees about the vendor tree
    from awpredict.explorer.fs_cartographer import _classify
    assert _classify("node_modules", True) == "cache"
    assert _classify("site-packages", True) == "cache"
    assert _classify("src", True) == "dir"


def test_cache_dirs_observed_but_never_descended(tmp_path):
    """Vendor/tool-state trees paint a 'cache' block in the parent frame (the
    structure signal survives) but are NOT in _subdirs — the explorer never
    crawls them. Measured on this monorepo: .claude/.github depth-3 children
    starved a BFS walk so lib/clients sat at queue position ~175, unreached in
    a 400-step walk."""
    root = tmp_path / "t"
    root.mkdir()
    (root / "node_modules").mkdir()
    (root / "src").mkdir()
    obs = observe_dir(str(root))
    kids = {c["type"] for c in obs["children"]}
    assert "cache" in kids                          # observed: treemap signal
    subs = [os.path.basename(p) for _, p in obs["_subdirs"]]
    assert "node_modules" not in subs               # never descended
    assert "src" in subs


def test_unknown_extension_drops_not_coerces(tmp_path):
    """An unparseable entry is a dropped label, never a default class — the
    doctrine that fixed the collapsed training buffer."""
    root, _info, _enc, adapter, _mem, model = _setup(tmp_path)
    weird = os.path.join(root, "weird.zzzz")
    with open(weird, "wb") as fh:
        fh.write(b"x" * 10)
    obs = observe_dir(root)
    kids = {c["type"] for c in obs["children"]}
    assert "unknown" in kids


def test_predictive_and_planner_policies_run(tmp_path):
    """The imagination-based policies must complete on the cold-start engine
    (identity forward model) and produce a map, never raise."""
    root, info, _enc, adapter, mem, model = _setup(tmp_path)
    for policy in ("predictive", "planner"):
        res = _run(root, mem, model, adapter, policy, 40)
        assert res["n_visited"] > 0
        assert res["n_landmarks"] >= 1


def test_llm_ablation_runs_with_cond(tmp_path):
    """The no-world-model ablation (semantic frontier from the reading) runs on
    the cond seam and records the WM-independent purpose yardstick."""
    root, _info, _enc, adapter, mem, model = _setup(tmp_path)
    cond = PurposeCond(root)
    m = MapMemory(match_radius=mem.match_radius, ema=0.15)
    carto = FsCartographer(model, adapter, m, cond_fn=cond)
    res = carto.explore(root, 40, policy="llm")
    assert res["n_visited"] > 0
    assert res["n_purposes"] >= 2                       # visited >1 kind of place


def test_determinism(tmp_path):
    root, _info, _enc, adapter, mem, model = _setup(tmp_path)
    a = _run(root, mem, model, adapter, "curious", 40)
    b = _run(root, mem, model, adapter, "curious", 40)
    assert a["visited"] == b["visited"]
    assert a["n_landmarks"] == b["n_landmarks"]
