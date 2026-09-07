"""World-model contracts: the two protocols every engine and environment obeys.

These are runtime-checkable Protocols, written to match what the shipped
engines ACTUALLY expose today (awpredict.core.lewm.LeWorldModel, the JEPA
engine behind an ARC-AGI-3 solving agent, and awpredict.core.mlp.MLPWorldModel,
a lighter embedding-MLP transition model). They are the seam the rest of the
program plugs into: adapters map an environment into (observation, action)
space, an engine learns the dynamics, and every consumer (a solving agent,
code-ranking, sandboxed exploration, an evolutionary scheduler) talks to
this surface only.

Rules of the contract:
  * Degrade loudly, never silently: an engine that cannot operate (torch
    missing, checkpoint unreadable) exposes ``ok == False`` and returns
    None/[] from methods — it must never raise into a caller's turn loop, and
    it must never fabricate a prediction.
  * ``surprise`` is the universal signal: prediction error in latent space,
    normalized so consumers can threshold it — a fitness signal for
    evolutionary selection, a violation-of-expectation gate for safe
    exploration, an anomaly feed for a knowledge graph.
  * Checkpoint promotion decisions NEVER come from an engine's own training
    buffer — only from a held-out evaluation set.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable


@runtime_checkable
class WorldModel(Protocol):
    """A learned latent dynamics model: encode → predict → plan, with surprise.

    ``obs`` is whatever the paired EnvironmentAdapter's ``observe`` returns —
    an ARC grid (list of lists / ndarray), an embedding vector, etc. Engines
    document which observation family they accept; the adapter guarantees it.
    """

    ok: bool

    def observe(self, obs: Any, action: Any, next_obs: Any, *args: Any,
                **kwargs: Any) -> Any:
        """Buffer one transition for training. Returns engine-specific status."""

    def encode(self, obs: Any, cond: Any = None) -> Any:
        """Observation → latent z, or None when degraded."""

    def predict(self, z: Any, action: Any, **kwargs: Any) -> Any:
        """Latent + action → predicted next latent, or None when degraded."""

    def surprise(self, obs: Any, action: Any, next_obs: Any, *args: Any,
                 **kwargs: Any) -> Optional[float]:
        """Prediction error for the observed transition; None when degraded."""

    def train_step(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """One (or a few) optimization steps over the buffer; loss dict or None."""

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        """Search action space in latent imagination toward a goal."""

    def save(self, path: str, *args: Any, **kwargs: Any) -> bool:
        """Checkpoint to disk; True on success (atomic where the engine supports it)."""

    def load(self, path: str, *args: Any, **kwargs: Any) -> Any:
        """Load a checkpoint; engine-specific status. Must not raise on missing file."""


@runtime_checkable
class EnvironmentAdapter(Protocol):
    """Maps one environment family into a WorldModel's observation/action space.

    An adapter is the ONLY thing that knows a domain's shape. Enrolling an
    agent in a new environment means writing (or selecting) an adapter —
    nothing in an engine changes.
    """

    #: short domain tag carried on transitions (e.g. "arc", "code", "sandbox")
    domain: str

    def observe(self, env_state: Any) -> Any:
        """Convert raw environment state into the engine's observation format."""

    def actions(self) -> Sequence[Any]:
        """The discrete action vocabulary for this environment (or a sample of it)."""

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute an action for real: (next_env_state, reward, done, info).

        Exploration safety is the adapter's duty: a sandbox adapter executes
        inside the sandbox; a code adapter performs read-only probes unless
        explicitly configured otherwise.
        """


def conforms(obj: Any, proto: type) -> List[str]:
    """Return the members of ``proto`` that ``obj`` is missing (empty == conforms).

    Protocol ``isinstance`` checks only see attribute presence; this helper
    names what is absent so a failing conformance check says WHY.
    """
    missing = []
    for name in getattr(proto, "__protocol_attrs__", set()):
        if not hasattr(obj, name):
            missing.append(name)
    return sorted(missing)


# =============================================================================
# The PROBLEM contract — what turns "an environment" into "a thing being solved"
# =============================================================================
#
# An EnvironmentAdapter tells you what the world does. It does not tell you
# whether you are winning. The ARC-AGI-3 solving lane learned that the hard
# way: every layer of the solver (memory, hypothesis council, MCTS, the value
# head, the JEPA world model) became real only once a SCORER existed that could
# refuse — a held-out level count, an A/B against the bare model, a latent
# health band. Before that, a run that learned nothing and a run that explored
# honestly and found nothing printed the same line.
#
# So a Problem is three things, and all three are data, not code:
#   * an adapter reference  — the world (observe / actions / step)
#   * a scorer reference    — the judge, which may REFUSE to judge
#   * a budget              — the hard stop that makes a run a run
#
# and every solving run emits the SAME four records regardless of domain:
# Transition (what happened), Attempt (how this try went), Fact (what was
# learned, scoped), Outcome (the verdict that a learner consumes). One record
# shape per role means a loop over ARC, over a kernel harness, over a fleet
# defect and over a browser can be replayed, compared and trained on by one
# consumer. Six rival shapes existed in the monorepo when this was written;
# these are their union, kept small.
#
# Rules (asserted by tests/test_problem_contracts.py):
#   * A scorer returns Score OR Refusal — NEVER None, NEVER 0.0 for "could not
#     judge". `check_score()` names anything else as a defect. A silent zero is
#     the worst failure this contract can have: the loop keeps running and
#     optimises a number nobody produced.
#   * Score.value is finite. NaN/inf is a programming error and raises.
#   * `baseline` on an Outcome is required to be SET before `kept` means anything;
#     the first honest scorer for a learned model is "beats the self-updating
#     lookup on the rows that are actually novel" — never an aggregate.
#   * memory_scope is awm-shaped: exactly three ':'-separated segments,
#     `tenant:user:project`, `*` for "not narrowed". A Fact never crosses it.


@dataclass(frozen=True)
class Score:
    """One measurement a scorer stands behind.

    ``strict_ok`` mirrors the harness `--strict` exit-code contract: False means
    the candidate was WRONG (not merely slow), and a loop must treat the trial as
    a revert regardless of ``value``.
    """

    metric: str
    value: float
    minimize: bool = False
    strict_ok: bool = True
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("Score.metric must be a non-empty name")
        v = float(self.value)
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"Score.value must be finite, got {self.value!r}")
        object.__setattr__(self, "value", v)

    def better_than(self, baseline: Optional[float]) -> bool:
        """Is this score an improvement over ``baseline``? A wrong candidate never is."""
        if not self.strict_ok:
            return False
        if baseline is None:
            return True
        return self.value < baseline if self.minimize else self.value > baseline


@dataclass(frozen=True)
class Refusal:
    """A scorer's honest 'I could not judge this'. Carries WHY; never a number."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("Refusal.reason must say why the scorer could not judge")


@runtime_checkable
class Scorer(Protocol):
    """Judges one episode. May refuse. Must never return nothing."""

    metric: str
    minimize: bool

    def score(self, episode: Any) -> "Score | Refusal":
        """Return a Score, or a Refusal naming why no score is possible."""


def check_score(result: Any) -> List[str]:
    """Name what is wrong with a scorer's return value (empty == acceptable).

    The point of this helper is the None/0.0 case: a scorer that returns nothing
    is not a scorer that scored zero, and a loop that cannot tell the two apart
    will ratchet on silence.
    """
    problems: List[str] = []
    if result is None:
        problems.append("scorer returned None — a refusal must be a Refusal(reason=...)")
    elif isinstance(result, (int, float)) and not isinstance(result, bool):
        problems.append("scorer returned a bare number — wrap it in Score(metric=..., value=...)")
    elif not isinstance(result, (Score, Refusal)):
        problems.append(f"scorer returned {type(result).__name__}; expected Score or Refusal")
    return problems


@dataclass(frozen=True)
class Budget:
    """The hard stop. Zero means 'not bounded on this axis'; at least one axis must bind."""

    steps: int = 80
    seconds: float = 0.0
    tokens: int = 0

    def problems(self) -> List[str]:
        out: List[str] = []
        if self.steps < 0 or self.seconds < 0 or self.tokens < 0:
            out.append("budget axes must be >= 0")
        if self.steps == 0 and self.seconds == 0 and self.tokens == 0:
            out.append("budget binds on no axis — a run with no stop is not a run")
        return out


def _looks_like_ref(ref: str) -> bool:
    mod, sep, name = (ref or "").partition(":")
    return bool(sep) and bool(mod) and bool(name) and " " not in ref


def scope_problems(scope: str) -> List[str]:
    """awm scope rule: exactly three ':' segments, none empty; '*' = not narrowed."""
    parts = (scope or "").split(":")
    if len(parts) != 3:
        return [f"memory_scope must be tenant:user:project (3 segments), got {scope!r}"]
    if any(not p for p in parts):
        return [f"memory_scope has an empty segment: {scope!r}"]
    return []


@dataclass
class ProblemSpec:
    """A problem, as data: the world, the judge, the stop, and where memory goes.

    ``adapter_ref`` / ``scorer_ref`` are ``"package.module:ClassName"`` strings.
    Loading them is the RUNNER's job (and the runner's import boundary); this
    module only says whether the spec is well-formed. ``planner`` is explicit on
    purpose: the ARC solver's MCTS/value-head planner was gated behind a CHAIN of
    three env flags, and setting the obvious one alone was a silent no-op.
    """

    problem_id: str
    domain: str
    adapter_ref: str
    scorer_ref: str
    budget: Budget = field(default_factory=Budget)
    memory_scope: str = "platform:*:*"
    planner: str = "none"  # "none" | "mcts" | "cem" | "council" | engine-specific
    success: Optional[str] = None
    adapter_kwargs: Dict[str, Any] = field(default_factory=dict)
    scorer_kwargs: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.problem_id:
            out.append("problem_id is required")
        if not self.domain:
            out.append("domain is required")
        for label, ref in (("adapter_ref", self.adapter_ref), ("scorer_ref", self.scorer_ref)):
            if not _looks_like_ref(ref):
                out.append(f"{label} must be 'package.module:ClassName', got {ref!r}")
        out.extend(self.budget.problems())
        out.extend(scope_problems(self.memory_scope))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ProblemSpec":
        data = dict(raw)
        budget = data.pop("budget", None) or {}
        if isinstance(budget, Budget):
            b = budget
        else:
            b = Budget(**{k: budget[k] for k in ("steps", "seconds", "tokens") if k in budget})
        return cls(budget=b, **data)


@dataclass
class Transition:
    """One (obs, action, next_obs) the world produced. The unit every learner eats."""

    domain: str
    episode: str
    step: int
    obs: Any
    action: Any
    next_obs: Any
    reward: float = 0.0
    done: bool = False
    surprise: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Attempt:
    """How one try at one problem went — the ARC attempt ledger, made domain-neutral."""

    domain: str
    problem_id: str
    episode: str
    steps: int
    score: Optional[Score] = None
    refusal: Optional[Refusal] = None
    first_effect_step: Optional[int] = None
    opening_actions: List[Any] = field(default_factory=list)
    trajectory: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Fact:
    """Something learned, pinned to a scope it may never leave."""

    memory_scope: str
    relation: str
    fact: str
    confidence: float = 0.7
    source_episode: str = ""

    def problems(self) -> List[str]:
        out = scope_problems(self.memory_scope)
        if not (0.0 <= float(self.confidence) <= 1.0):
            out.append(f"confidence must be in [0,1], got {self.confidence!r}")
        if not self.relation or not self.fact:
            out.append("relation and fact are both required")
        return out

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Outcome:
    """The verdict a learner consumes. One per episode, whatever the domain."""

    problem_id: str
    domain: str
    episode: str
    score: Optional[Score] = None
    refusal: Optional[Refusal] = None
    baseline: Optional[float] = None
    kept: Optional[bool] = None
    transitions_n: int = 0
    facts_n: int = 0
    duration_ms: float = 0.0
    model_used: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        out: List[str] = []
        if self.score is None and self.refusal is None:
            out.append("an Outcome carries a Score or a Refusal — never neither")
        if self.score is not None and self.refusal is not None:
            out.append("an Outcome carries a Score or a Refusal — never both")
        if self.kept is not None and self.baseline is None:
            out.append("kept is meaningless without a baseline — set baseline first")
        return out

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
