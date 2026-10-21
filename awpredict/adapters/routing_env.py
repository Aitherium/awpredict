"""RoutingEnv - mixture-of-experts routing as an environment a world model can learn.

A MoE model picks ``k`` experts per layer per token. A host-offloaded runtime keeps only some of
each layer's experts resident (the *warm set*); every selected expert outside it is an upload on
the critical path. That is a sequential decision problem with a cheap, exact simulator - the
routing log itself:

* **observe** - the last ubatch's per-layer expert sets (a tuple of frozensets, one per layer;
  ``multi_hot`` flattens it into a fixed-length vector for vector engines);
* **actions** - candidate warm sets for the next ubatch (``WarmSetAction``: a name plus one
  frozenset of ``keep`` experts per layer), built from the history seen so far;
* **step** - reveal the next ubatch; reward = share of its selected experts the action kept
  resident (recall), ``info`` carries hits/needed per layer.

``step`` never peeks: candidate actions are built from ubatches already revealed, so an engine
trained on these transitions learns a policy that is deployable, not one that saw the answer.

The trace is read from an AXRL routing log (``read_axrl``; the little-endian format below) or
passed in as ``RoutingStep`` records. Dependency-free: no numpy, no torch.

    file header 32 B : "AXRL" u16 version=1 u16 n_layer u16 n_expert u16 n_expert_used
                       u64 unix_ms u8[12]
    record header 20 B: u16 0xA17E u16 layer u16 n_tokens u16 n_used u16 tag u16 0
                       u32 ubatch u32 t_ms
    record body       : n_tokens*n_used u16 expert ids

A writer that rotates keeps ``<path>.1`` (older) beside ``<path>``; ``axrl_files`` returns both,
oldest first. A trailing partial record is dropped.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

_HDR = struct.Struct("<4sHHHHQ12x")
_REC = struct.Struct("<HHHHHHII")
_REC_MAGIC = 0xA17E

Experts = Tuple[FrozenSet[int], ...]  # one set per layer


@dataclass(frozen=True)
class AxrlHeader:
    n_layer: int
    n_expert: int
    n_expert_used: int
    unix_ms: int


@dataclass
class RoutingStep:
    """One ubatch: ``layers[l]`` = distinct experts layer ``l`` selected (absent = not logged)."""

    ubatch: int
    tag: int = 0
    layers: Dict[int, FrozenSet[int]] = field(default_factory=dict)
    stream: int = 0  # file index; a new stream = a restarted process


@dataclass(frozen=True)
class WarmSetAction:
    """A candidate warm set: one frozenset of resident experts per layer."""

    name: str
    sets: Experts


@dataclass(frozen=True)
class RoutingState:
    index: int  # index of the LAST revealed step (-1 = nothing revealed yet)


def axrl_files(path: "str | os.PathLike[str]") -> List[str]:
    p = os.fspath(path)
    return [x for x in (p + ".1", p) if os.path.exists(x)]


def read_axrl(path: "str | os.PathLike[str]") -> Tuple[AxrlHeader, List[RoutingStep]]:
    """Read ``[<path>.1, <path>]``. Raises ValueError on a file that is not AXRL v1."""
    header: Optional[AxrlHeader] = None
    steps: List[RoutingStep] = []
    for stream, f in enumerate(axrl_files(path)):
        with open(f, "rb") as fh:
            raw = fh.read(_HDR.size)
            if len(raw) < _HDR.size:
                continue  # created, nothing written yet
            magic, ver, n_layer, n_expert, k, unix_ms = _HDR.unpack(raw)
            if magic != b"AXRL" or ver != 1:
                raise ValueError(f"{f}: not an AXRL v1 routing log")
            header = header or AxrlHeader(n_layer, n_expert, k, unix_ms)
            cur: Optional[RoutingStep] = None
            while True:
                rh = fh.read(_REC.size)
                if len(rh) < _REC.size:
                    break
                rmagic, layer, n_tok, n_used, tag, _pad, ubatch, _t = _REC.unpack(rh)
                if rmagic != _REC_MAGIC:
                    raise ValueError(f"{f}: lost record sync at {fh.tell() - _REC.size}")
                body = fh.read(2 * n_tok * n_used)
                if len(body) < 2 * n_tok * n_used:
                    break
                ids = frozenset(struct.unpack(f"<{n_tok * n_used}H", body))
                if cur is None or cur.ubatch != ubatch:
                    if cur is not None:
                        steps.append(cur)
                    cur = RoutingStep(ubatch, tag, {}, stream)
                cur.layers[layer] = cur.layers.get(layer, frozenset()) | ids
            if cur is not None:
                steps.append(cur)
    if header is None:
        raise ValueError(f"no readable AXRL log at {os.fspath(path)}")
    return header, steps


def multi_hot(obs: Experts, n_expert: int) -> List[float]:
    """Flatten an observation into ``n_layer * n_expert`` 0/1 floats for vector engines."""
    out = [0.0] * (len(obs) * n_expert)
    for layer, s in enumerate(obs):
        for e in s:
            out[layer * n_expert + e] = 1.0
    return out


class RoutingEnv:
    """EnvironmentAdapter over a routing trace. See the module docstring."""

    domain = "moe_routing"

    def __init__(self, steps: Sequence[RoutingStep], n_layer: int, n_expert: int, keep: int):
        if not 0 < keep <= n_expert:
            raise ValueError(f"keep must be in 1..{n_expert}, got {keep}")
        self.steps = list(steps)
        self.n_layer, self.n_expert, self.keep = n_layer, n_expert, keep
        self.reset()

    @classmethod
    def from_axrl(cls, path: "str | os.PathLike[str]", keep_fraction: float = 0.5) -> "RoutingEnv":
        h, steps = read_axrl(path)
        keep = max(1, min(h.n_expert, round(keep_fraction * h.n_expert)))
        return cls(steps, h.n_layer, h.n_expert, keep)

    # -- protocol -----------------------------------------------------------------------------
    def reset(self) -> RoutingState:
        self._freq = [[0] * self.n_expert for _ in range(self.n_layer)]
        self._last = [[-1] * self.n_expert for _ in range(self.n_layer)]
        self.state = RoutingState(-1)
        if self.steps:
            self._reveal(0)
        return self.state

    def observe(self, env_state: Any = None) -> Experts:
        st = self.state if env_state is None else env_state
        idx = st.index if isinstance(st, RoutingState) else int(st)
        if idx < 0:
            return tuple(frozenset() for _ in range(self.n_layer))
        layers = self.steps[idx].layers
        return tuple(layers.get(layer, frozenset()) for layer in range(self.n_layer))

    def actions(self) -> List[WarmSetAction]:
        """Candidate warm sets from the revealed history: LRU order, frequency, and their blend."""
        lru, freq, blend = [], [], []
        for layer in range(self.n_layer):
            last, cnt = self._last[layer], self._freq[layer]
            by_lru = sorted(range(self.n_expert), key=lambda e: (-last[e], e))
            by_freq = sorted(range(self.n_expert), key=lambda e: (-cnt[e], e))
            rank_l = {e: i for i, e in enumerate(by_lru)}
            rank_f = {e: i for i, e in enumerate(by_freq)}
            by_blend = sorted(range(self.n_expert), key=lambda e: (rank_l[e] + rank_f[e], e))
            lru.append(frozenset(by_lru[: self.keep]))
            freq.append(frozenset(by_freq[: self.keep]))
            blend.append(frozenset(by_blend[: self.keep]))
        return [
            WarmSetAction("last_ubatch", tuple(lru)),
            WarmSetAction("frequency", tuple(freq)),
            WarmSetAction("blend", tuple(blend)),
        ]

    def step(self, action: Any) -> Tuple[RoutingState, float, bool, Dict[str, Any]]:
        """Reveal the next ubatch and score ``action`` (a WarmSetAction or per-layer sets)."""
        nxt = self.state.index + 1
        if nxt >= len(self.steps):
            return self.state, 0.0, True, {"reason": "trace exhausted"}
        sets = action.sets if isinstance(action, WarmSetAction) else tuple(action)
        if len(sets) != self.n_layer:
            raise ValueError(f"action has {len(sets)} layers, env has {self.n_layer}")
        target = self.steps[nxt]
        hits = needed = 0
        per_layer: Dict[int, Tuple[int, int]] = {}
        for layer, used in target.layers.items():
            h = len(used & frozenset(sets[layer]))
            per_layer[layer] = (h, len(used))
            hits += h
            needed += len(used)
        self._reveal(nxt)
        reward = hits / needed if needed else 0.0
        done = nxt >= len(self.steps) - 1
        info = {
            "hits": hits,
            "needed": needed,
            "misses": needed - hits,
            "per_layer": per_layer,
            "ubatch": target.ubatch,
            "tag": target.tag,
            "action": getattr(action, "name", "custom"),
        }
        return self.state, reward, done, info

    # -- internals ----------------------------------------------------------------------------
    def _reveal(self, idx: int) -> None:
        st = self.steps[idx]
        if idx > 0 and st.stream != self.steps[idx - 1].stream:
            self._last = [[-1] * self.n_expert for _ in range(self.n_layer)]  # cache restarted
        for layer, used in st.layers.items():
            for e in used:
                self._freq[layer][e] += 1
                self._last[layer][e] = idx
        self.state = RoutingState(idx)
