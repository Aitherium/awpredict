"""fs_cartographer: map a filesystem with the world model — the 23MB explorer.

The whole cartographer core, pointed at a directory tree:

    directory --observe--> {type-composition tree} --StructuredAdapter--> token grid
              --engine encode--> latent z --MapMemory--> landmark + novelty

Each directory becomes a latent capturing its TYPE-COMPOSITION (the treemap of
file categories by size), so directories with similar content collapse to the
SAME landmark and structurally-distinct ones become NEW landmarks. The map is
therefore COMPACT — thousands of directories become a handful of "kinds of
place" — and the novelty signal (distance to nearest landmark) marks the
frontier: the structurally-new regions to explore.

Policies (``explore(policy=...)``):

* ``curious`` — expand the frontier dir whose PARENT was most novel (go deeper
  where it is interesting). The winner of the measured 23MB-vs-LLM result.
* ``predictive`` — plan which branch to open BEFORE opening it: imagine each
  child's latent with the engine's forward model, expand the most-novel
  prediction. Needs a trained engine; cold-start encoders have no forward model.
* ``prospector`` — content-seeking: go where RARE PURPOSES concentrate
  (alpha*kind_rarity + beta*purpose_rarity*depth_likely + gamma*novelty).
  Beat plain curiosity on needle-reach in the measured audit run.
* ``planner`` — the world-model-native mechanism: descend toward the centroid
  of RARE-purpose landmarks (cold-starts as ``predictive``).
* ``bfs`` / ``random`` — the blind baselines every gate is scored against.
* ``llm`` — the ablation: no world model at all, just the cond reading's
  purpose vector (the semantic frontier). The bar the world model must clear.

The engine seam: ``model`` must conform to ``awpredict.contracts.WorldModel``
(``encode(obs, cond=)``, ``predict(z, action, **kwargs)``) or at least expose
those two methods. ``action_fn(cx, cy)`` renders a "click this treemap box"
action; the default is the ``ACTION6(cx,cy)`` vocabulary the ARC-trained engine
(the one the blog result was measured with) understands — override for engines
with a different action space. ``cond_fn(abs_path)`` supplies the LLM's READING
of a directory (a 16-dim vector: 13 purpose dims, depth-likelihood, novelty,
...) — the train/inference seam from the blog: a model trained with a cond must
be run with it, or it is a different model.

Extracted from the Aitherium ARC-AGI-3 agent fork (``agents/fs_cartographer.py``
+ ``agents/map_memory.py``), which is the implementation the published
23-megabyte-world-model result was measured with. The FastContext rival policy
(needs a live LLM endpoint) is intentionally not ported; the probe tool drives
that comparison externally.
"""
from __future__ import annotations

import heapq
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on numpy-less installs
    _NP_AVAILABLE = False
    np = None  # type: ignore[assignment]

# extension -> type vocab (the treemap's 8 content categories + friends)
_EXT: Dict[str, set] = {
    "code": {".py", ".js", ".ts", ".tsx", ".jsx", ".c", ".cpp", ".h", ".hpp", ".java",
             ".go", ".rs", ".sh", ".ps1", ".rb", ".php", ".lua", ".sql", ".css", ".html"},
    "data": {".json", ".jsonl", ".csv", ".tsv", ".parquet", ".db", ".sqlite", ".xml",
             ".npy", ".npz"},
    "media": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".mp4", ".mov", ".avi",
              ".mp3", ".wav", ".flac", ".svg", ".ico"},
    "doc": {".md", ".txt", ".pdf", ".docx", ".doc", ".rtf", ".rst"},
    "archive": {".zip", ".tar", ".gz", ".7z", ".rar", ".bz2", ".xz"},
    "config": {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env", ".lock"},
    "log": {".log", ".out", ".err"},
    "model": {".safetensors", ".ckpt", ".pt", ".pth", ".onnx", ".gguf", ".bin", ".pkl"},
    "secret": {".pem", ".key", ".crt", ".p12"},
    "binary": {".exe", ".dll", ".so", ".dylib", ".o", ".a", ".wasm"},
}
_EXT_LOOKUP: Dict[str, str] = {e: t for t, exts in _EXT.items() for e in exts}

# Vendor/cache/tool-state trees: rendered flat, never peeked, NEVER descended
# into, and never needle targets. site-packages / venvs are the python analogue
# of node_modules — vendored mass, not project content; .claude/.vscode/...
# are tool state. Without this, a needle rule plants targets inside e.g. a
# bundled runtime's certificate store: rare-kind by the letter, buried under
# thousands of near-identical vendor dirs with no structural gradient to climb.
# Descending into them is worse than wasted budget: measured on this monorepo,
# .AITHEROS/.claude/.github depth-3 children flooded a BFS queue so that
# lib/clients — the actual content — sat at queue position ~175 and was never
# reached in a 400-step walk. They are OBSERVED (painted "cache" in the parent
# treemap, so the structure signal survives) but never enqueued.
_CACHE_DIRS = {"__pycache__", ".cache", "node_modules", ".git", ".ruff_cache",
               ".pytest_cache", "site-packages", ".venv", "venv", ".tox",
               ".mypy_cache", ".next", "dist", "build",
               ".claude", ".vscode", ".idea", ".AITHEROS", ".AITHERIUM",
               ".RESEARCH", ".agent"}

# The default descend-action vocabulary — what the ARC-trained engine expects.
def _default_action_fn(cx: int, cy: int) -> str:
    return f"ACTION6({cx},{cy})"


def _classify(name: str, is_dir: bool) -> str:
    """A path name -> a type id. A directory is 'cache' when its name is in
    _CACHE_DIRS, else 'dir'; a file is its extension's category, with anything
    unrecognized kept as 'unknown' — dropped, never coerced."""
    if is_dir:
        return "cache" if name in _CACHE_DIRS else "dir"
    ext = os.path.splitext(name)[1].lower()
    return _EXT_LOOKUP.get(ext, "unknown")


def _peek(path: str, max_entries: int = 200) -> Tuple[str, float]:
    """One extra (cheap) scandir level for a subdir: its DOMINANT content class
    (size-weighted) and total direct byte-size. This is the fix the adversarial
    reviews demanded: without it every subdir paints as a flat 'dir' block of
    constant size, the parent frame carries ZERO per-child content, and
    (parent, click) -> child is not a function — the forward model collapses to
    the marginal-mean treemap. Coloring the block by dominant content and
    sizing it by real area makes descent a decodable (predictive) function.
    Cache dirs are NOT peeked (kept flat) to bound cost and cycles."""
    from collections import defaultdict
    by_type: Dict[str, float] = defaultdict(float)
    size_acc = 0.0
    try:
        # sorted: scandir order is filesystem-dependent (NTFS index order), and
        # a tie between content weight and dir-block weight resolves by scan
        # order — an unsorted scan makes the map's structure machine-dependent.
        with os.scandir(path) as it:
            for i, e in enumerate(sorted(it, key=lambda x: x.name)):
                if i >= max_entries:
                    break
                try:
                    if e.is_dir():
                        t, s = _classify(e.name, True), 1000.0
                    else:
                        t, s = _classify(e.name, False), float(max(1, e.stat().st_size))
                except OSError:
                    continue
                by_type[t] += s
                size_acc += s
    except OSError:
        return "dir", 1.0          # unreadable dir -> the flat default, loudly
    if not by_type:
        return "dir", 1.0
    dom = max(by_type.items(), key=lambda kv: kv[1])[0]
    return dom, max(1.0, size_acc)


def observe_dir(path: str, max_entries: int = 400) -> Dict[str, Any]:
    """A directory -> a shallow StructuredAdapter tree: children are files
    (type+size) and subdirs (colored by their DOMINANT content class and sized
    by their real byte-area via one cheap peek). ``_subdirs`` is a list of
    (child_index, abs_path): child_index is the position in ``children``, so a
    descend into that subdir clicks the treemap box for exactly that leaf."""
    children: List[Dict[str, Any]] = []
    subdirs: List[Tuple[int, str]] = []
    try:
        # sorted, as in _peek: scandir order is filesystem-dependent and the
        # (child_index -> click box) mapping must be reproducible across runs.
        with os.scandir(path) as it:
            for i, e in enumerate(sorted(it, key=lambda x: x.name)):
                if i >= max_entries:
                    break
                try:
                    is_dir = e.is_dir()
                    if is_dir:
                        name = e.name
                        if name in _CACHE_DIRS:
                            dom, sz = "cache", 1000.0
                        else:
                            dom, sz = _peek(e.path)
                            subdirs.append((len(children), e.path))
                        children.append({"type": dom, "size": max(1.0, sz)})
                    else:
                        sz = e.stat().st_size
                        children.append({"type": _classify(e.name, False),
                                         "size": max(1, sz)})
                except OSError:
                    continue
    except OSError:
        # an unreadable directory observes as an empty directory, not a crash
        return {"type": "dir", "size": 1, "children": [], "_subdirs": []}
    return {"type": "dir", "size": 1, "children": children, "_subdirs": subdirs}


class FsCartographer:
    """Maps a filesystem with an engine + landmark memory + a policy.

    ``cond_fn(abs_path) -> [D_c] | None`` supplies the LLM's READING of a
    directory. It must be passed whenever the engine's checkpoint was trained
    with a cond, because the model then lives in a different space than a
    cond-free call produces. None (the default) = the cond-free behaviour.
    """

    def __init__(self, model: Any, adapter: Any, mem: Any,
                 cond_fn: Optional[Callable[[str], Optional[Any]]] = None,
                 action_fn: Optional[Callable[[int, int], Any]] = None,
                 prospect_weights: Tuple[float, float, float] = (1.0, 1.0, 0.5)) -> None:
        self.model = model
        self.adapter = adapter
        self.mem = mem
        self.cond_fn = cond_fn
        self.action_fn = action_fn or _default_action_fn
        self._pw = tuple(float(w) for w in prospect_weights)

    # -- seams ------------------------------------------------------------------
    def _cond(self, path: Optional[str]) -> Optional[Any]:
        if self.cond_fn is None or not path:
            return None
        try:
            return self.cond_fn(path)
        except Exception:  # noqa: BLE001 - a reading failure is a missing reading
            return None

    def _encode(self, obs: Dict[str, Any], path: Optional[str] = None) -> Any:
        grid = self.adapter.token_grid(obs)
        if grid is None:
            return None
        return self.model.encode(grid, cond=self._cond(path))

    def _child_action(self, box: Tuple[int, int, int, int], grid_size: int) -> Any:
        r0, r1, c0, c1 = box
        cr = min(grid_size - 1, max(0, (r0 + r1) // 2))     # ROW  -> cy
        cc = min(grid_size - 1, max(0, (c0 + c1) // 2))     # COL  -> cx
        return self.action_fn(cc, cr)

    def _predicted_child_latents(self, obs: Dict[str, Any], z_parent: Any) -> Dict[str, Any]:
        """For each unopened child, IMAGINE its latent with the engine's forward
        model — z_pred = predict(z_parent, click(child box), ctx=child's reading).
        No child directory is opened: the reading comes from its name/path (+ a
        shallow peek), all visible from the parent. Returns {child_abs_path: z_pred},
        skipping children the engine cannot imagine (degrade loudly, per child)."""
        grid_size = self.adapter.grid if hasattr(self.adapter, "grid") else 64
        grid, boxes = self.adapter.token_grid_with_boxes(obs)
        if grid is None:
            return {}
        out: Dict[str, Any] = {}
        for cidx, sp in obs.get("_subdirs", []):
            if cidx not in boxes:                    # collapsed box -> no click location
                continue
            try:
                z_pred = self.model.predict(z_parent, self._child_action(boxes[cidx], grid_size),
                                            ctx=self._cond(sp))
            except Exception:  # noqa: BLE001 - engine refuses this action -> skip child
                z_pred = None
            if z_pred is not None:
                out[sp] = z_pred
        return out

    def _rare_goal_centroid(self, lm_purpose: Dict[int, Dict[int, int]],
                            purpose_freq: Dict[int, int]) -> Any:
        """The centroid of landmark latents whose dominant purpose is RARE
        (bottom third of visit counts so far): where 'more of the scarce stuff'
        lives in latent space. None until the map has enough structure (cold
        start -> caller falls back to predicted-novelty)."""
        if not _NP_AVAILABLE or len(self.mem.landmarks) < 5 or len(purpose_freq) < 2:
            return None
        counts = sorted(purpose_freq.values())
        thresh = counts[max(0, len(counts) // 3 - 1)]
        rare = {p for p, c in purpose_freq.items() if c <= thresh}
        if not rare:
            return None
        vecs = []
        for idx, hist in lm_purpose.items():
            if hist and idx < len(self.mem.landmarks):
                if max(hist, key=hist.get) in rare:
                    vecs.append(self.mem.landmarks[idx])
        if not vecs:
            return None
        return np.mean(np.stack(vecs), axis=0)

    def _purpose_novelty(self, path: str) -> Tuple[int, float]:
        """(purpose_id, novelty) straight from the cond reading — NO world model.
        purpose = argmax of the cond's first 13 purpose dims; novelty = its own
        0-1 score. (-1, 0.0) when there is no reading."""
        c = self._cond(path)
        if c is None or len(c) < 16:
            return -1, 0.0
        head = list(c[:13])
        if not any(head):
            return -1, float(c[15]) if len(c) > 15 else 0.0
        return max(range(13), key=lambda i: head[i]), float(c[15])

    # -- the walk ---------------------------------------------------------------
    def explore(self, root: str, budget: int, policy: str = "curious",
                seed: int = 0) -> Dict[str, Any]:
        """Walk from ``root``, encoding each directory into the map. Returns the
        per-step record + coverage. See the module docstring for the policies."""
        import random as _random
        rng = _random.Random(seed)
        visited: List[Tuple[str, int, float]] = []   # (path, landmark_idx, novelty)
        seen_paths = set()
        counter = 0
        # frontier entries: (priority, tiebreak, path, parent_novelty); a heap for
        # the scored policies, a plain list for bfs (FIFO) and random.
        frontier: List[Any] = []
        heapq.heappush(frontier, (0.0, counter, root, float("inf")))
        seen_paths.add(root)

        seen_purposes: set = set()          # llm ablation: visited purpose ids
        kind_freq: Dict[str, int] = {}      # prospector: online peeked-kind histogram
        purpose_freq: Dict[int, int] = {}   # prospector/planner: visited purposes
        lm_purpose: Dict[int, Dict[int, int]] = {}   # planner: landmark -> purpose hist
        # A SECOND, world-model-INDEPENDENT coverage yardstick: distinct LLM
        # purposes visited. Landmark count is measured with the engine's own
        # encoder, which could flatter engine-based policies; distinct purposes
        # is scored by the cond alone, so the ablation is not graded by the
        # thing it is competing against.
        visited_purposes: set = set()
        while frontier and len(visited) < budget:
            if policy in ("curious", "predictive", "prospector", "planner"):
                _negprio, _, path, _pnov = heapq.heappop(frontier)
            elif policy == "random":
                path = frontier.pop(rng.randrange(len(frontier)))[2]
            elif policy == "llm":
                # THE ABLATION: a steelmanned LLM-ONLY policy that uses NO world
                # model — no encoder, no latent, no map novelty. It prefers a
                # frontier dir whose predicted PURPOSE we have not visited yet,
                # tie-broken by the reading's own novelty. This is what a
                # competent engineer writes WITHOUT any of this machinery, and it
                # is the bar the world model has to clear.
                best_i, best_key = 0, None
                for i, ent in enumerate(frontier):
                    pur, nov = self._purpose_novelty(ent[2])
                    key = (0 if (pur < 0 or pur not in seen_purposes) else 1, -nov)
                    if best_key is None or key < best_key:
                        best_key, best_i = key, i
                path = frontier.pop(best_i)[2]
            else:
                path = frontier.pop(0)[2]
            if policy == "llm":
                p_, _ = self._purpose_novelty(path)
                if p_ >= 0:
                    seen_purposes.add(p_)
            obs = observe_dir(path)
            z = self._encode(obs, path)
            if z is None:
                continue
            nov = self.mem.novelty(z)
            idx = self.mem.integrate(z)
            visited.append((path, idx, nov))
            if self.cond_fn is not None:     # WM-INDEPENDENT diversity yardstick
                p_v, _ = self._purpose_novelty(path)
                if p_v >= 0:
                    visited_purposes.add(p_v)
            # prospector/planner: track the ONLINE purpose histogram per landmark
            goal = None
            if policy in ("prospector", "planner"):
                p_c, _ = self._purpose_novelty(path)
                if p_c >= 0:
                    purpose_freq[p_c] = purpose_freq.get(p_c, 0) + 1
                    lm_purpose.setdefault(idx, {})[p_c] = lm_purpose.setdefault(
                        idx, {}).get(p_c, 0) + 1
                if policy == "planner":
                    goal = self._rare_goal_centroid(lm_purpose, purpose_freq)
            # predictive/planner: imagine each child's latent ONCE, unopened.
            pred_lat = (self._predicted_child_latents(obs, z)
                        if policy in ("predictive", "planner") else {})
            pred_nov = ({sp: self.mem.novelty(zp) for sp, zp in pred_lat.items()}
                        if policy == "predictive" else {})
            for _cidx, sub in obs.get("_subdirs", []):
                if sub in seen_paths:
                    continue
                seen_paths.add(sub)
                counter += 1
                if policy == "predictive":
                    pn = pred_nov.get(sub, 0.0)
                    pr = -(pn if pn != float("inf") else 1e18)
                    heapq.heappush(frontier, (pr, counter, sub, pn))
                elif policy == "curious":
                    # priority = this dir's novelty (descend where the parent was
                    # interesting); negate for a max-heap; inf (new region) first.
                    pr = -(nov if nov != float("inf") else 1e18)
                    heapq.heappush(frontier, (pr, counter, sub, nov))
                elif policy == "prospector":
                    kind = (obs["children"][_cidx]["type"]
                            if _cidx < len(obs["children"]) else "dir")
                    kind_freq[kind] = kind_freq.get(kind, 0) + 1
                    kr = 1.0 / (1.0 + kind_freq[kind])
                    pp, dl = 0.0, 0.0
                    c = self._cond(sub)
                    if c is not None and len(c) >= 15:
                        head = list(c[:13])
                        if any(head):
                            pi = max(range(13), key=lambda i: head[i])
                            pp = 1.0 / (1.0 + purpose_freq.get(pi, 0))
                            dl = float(c[14])
                    novn = 3.0 if nov == float("inf") else min(
                        nov / (self.mem.match_radius + 1e-6), 3.0)
                    score = (self._pw[0] * kr + self._pw[1] * pp * dl
                             + self._pw[2] * novn)
                    heapq.heappush(frontier, (-score, counter, sub, nov))
                elif policy == "planner":
                    # ONE positive scale (units of match_radius; smaller = better)
                    # so cold-start and goal-phase entries stay comparable.
                    rad = self.mem.match_radius + 1e-6
                    zp = pred_lat.get(sub)
                    if zp is None:
                        pr = 1e18                        # unimaginable -> last resort
                    elif goal is not None:
                        zv = zp.detach().cpu().numpy() if hasattr(zp, "detach") else zp
                        pr = float(((np.asarray(zv, dtype=np.float32).reshape(-1)
                                     - goal) ** 2).sum()) / rad
                    else:                                 # cold start = predictive
                        pn = self.mem.novelty(zp)
                        pn = 1e9 if pn == float("inf") else pn
                        pr = 1.0 / (1.0 + pn / rad)
                    heapq.heappush(frontier, (pr, counter, sub, nov))
                else:
                    frontier.append((0.0, counter, sub, 0.0))

        return {"visited": visited, "coverage": self.mem.coverage(),
                "n_visited": len(visited), "n_landmarks": len(self.mem.landmarks),
                "n_purposes": len(visited_purposes), "policy": policy,
                "budget": budget}


def summarize(carto: FsCartographer, res: Dict[str, Any], root: str) -> str:
    """A compact human summary of an explore() result: the landmarks and the
    directories each one gathered. Returns the text (caller prints/logs it)."""
    from collections import defaultdict
    by_lm = defaultdict(list)
    for path, idx, _nov in res["visited"]:
        by_lm[idx].append(os.path.relpath(path, root))
    lines = [f"visited {res['n_visited']} dirs -> {res['n_landmarks']} MAP LANDMARKS "
             f"(distinct 'kinds of place')  coverage={res['coverage']}",
             "  each landmark = a cluster of structurally-similar directories:"]
    for idx in sorted(by_lm, key=lambda k: -len(by_lm[k]))[:8]:
        ex = by_lm[idx][:3]
        lines.append(f"    landmark {idx:>2} ({len(by_lm[idx])} dirs): {', '.join(ex)}")
    return "\n".join(lines)
