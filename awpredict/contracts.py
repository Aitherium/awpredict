"""World-model contracts: the protocols every engine, environment and renderer obeys.

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
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)


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


#: Class-namespace names that are machinery, not protocol members. The union of
#: CPython's pre-3.12 _get_protocol_attrs exclusions and the names 3.12/3.13
#: added to class namespaces, so the fallback agrees with __protocol_attrs__.
_NOT_MEMBERS = frozenset({
    "__abstractmethods__", "__annotations__", "__weakref__", "_is_protocol",
    "_is_runtime_protocol", "__dict__", "__args__", "__slots__", "__next_in_mro__",
    "__parameters__", "__origin__", "__orig_bases__", "__extra__", "__tree_hash__",
    "__doc__", "__subclasshook__", "__init__", "__new__", "__module__",
    "_MutableMapping__marker", "_gorg", "__protocol_attrs__",
    "__non_callable_proto_members__", "__type_params__", "__qualname__",
    "__firstlineno__", "__static_attributes__", "__annotate__",
    "__annotate_func__", "__annotations_cache__",
})


def protocol_members(proto: type) -> set:
    """The member names a Protocol requires, on every supported Python.

    ``__protocol_attrs__`` exists only on Python 3.12+. Reading it with a
    ``set()`` default made :func:`conforms` FAIL OPEN on 3.10/3.11 -- it
    iterated nothing and approved every object -- while the package declares
    ``requires-python >= 3.10``. Measured 2026-10-07 on the public mirror's
    first CI run: 3.12 named ``['actions', 'step']`` for a half-built adapter,
    3.10 returned ``[]``. Below 3.12 the members are derived the way CPython's
    own pre-3.12 ``_get_protocol_attrs`` did; a parity test pins the two
    against each other on any 3.12+ interpreter.
    """
    attrs = getattr(proto, "__protocol_attrs__", None)
    if attrs is not None:
        return set(attrs)
    return _derive_protocol_members(proto)


def _derive_protocol_members(proto: type) -> set:
    """The pre-3.12 derivation, split out so a test can pin it to the stdlib."""
    names = set()
    for base in proto.__mro__[:-1]:  # without object
        if base.__name__ in ("Protocol", "Generic"):
            continue
        annotations = base.__dict__.get("__annotations__", {})
        for name in list(base.__dict__) + list(annotations):
            if not name.startswith("_abc_") and name not in _NOT_MEMBERS:
                names.add(name)
    return names


def conforms(obj: Any, proto: type) -> List[str]:
    """Return the members of ``proto`` that ``obj`` is missing (empty == conforms).

    Protocol ``isinstance`` checks only see attribute presence; this helper
    names what is absent so a failing conformance check says WHY.
    """
    return sorted(name for name in protocol_members(proto) if not hasattr(obj, name))


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


# =============================================================================
# The GENERATIVE contract — a world model that RENDERS, not only predicts
# =============================================================================
#
# Everything above learns dynamics over whatever observation space an adapter
# hands it: an ARC grid, an embedding vector. That is enough to predict, and
# not enough to SHOW anything. A Genie-class system is asked for something
# else — a sentence in, a real-time steerable frame with its own audio out —
# and a latent predictor answers none of it.
#
# What such a system needs is small, and deliberately NOT a subtype of
# ``WorldModel``: a video or image backend has no honest latent to
# ``encode``, and a contract that made it fake one would be a contract that
# rewards lying. A backend implements whichever protocols it actually
# satisfies; the two compose at the RECORD level — an
# ``EnvironmentAdapter`` supplies the action, the generative model supplies
# the next ``Frame``, and the pair becomes one ``Transition``.
#
# Status, stated rather than implied: as of 2026-10-27 NOTHING implements
# ``GenerativeWorldModel`` in this package OR in the monorepo, and Genie has no
# public API. This is a specified target. The backends that will implement it
# belong in ``AitherOS/lib/worldgen`` (engine, still, video, replay, genie) and
# do not exist yet; when they do they obey the same rule as every engine here:
# ``ok == False`` plus a ``Refusal`` when they cannot render — never a
# fabricated frame, never a silent black one.


@dataclass(frozen=True)
class Frame:
    """One control-plane record for a rendered frame.

    Payloads do NOT travel in this object. ``image_ref`` / ``audio_ref`` /
    ``depth_ref`` name where the bytes are: a side-channel id on a WebSocket,
    a path, or a URL. Inlining a 4 MB frame into JSON is how a stream becomes
    a stall, and base64 in a dict is indistinguishable from a bug when it
    fails. An empty ref means "no payload of this kind", never "zero bytes".
    """

    frame_id: int
    ts: float
    width: int
    height: int
    world_id: str = ""
    codec: str = "image/jpeg"
    image_ref: str = ""
    audio_ref: str = ""
    audio_codec: str = ""
    depth_ref: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        """What makes this frame unusable downstream (empty == usable)."""
        out: List[str] = []
        if self.frame_id < 0:
            out.append("frame_id must be >= 0")
        if self.width <= 0 or self.height <= 0:
            out.append(f"frame geometry must be positive, got {self.width}x{self.height}")
        if not self.image_ref and not self.audio_ref:
            out.append("a Frame carries an image_ref or an audio_ref — never neither")
        return out


@dataclass(frozen=True)
class Action:
    """A control input to a continuous 3D world.

    Axes are normalized to [-1, 1] so a keyboard, a gamepad and a policy all
    speak one vocabulary; ``kind`` is the open-ended part (``move``, ``look``,
    ``interact``, ``idle``, or an engine-specific verb).
    """

    kind: str = "idle"
    forward: float = 0.0
    strafe: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    target: str = ""
    text: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Steer:
    """A natural-language edit to the world, in Genie's sense.

    ``ttl_frames`` bounds how long the edit holds: 0 means "until replaced",
    which is what an interactive session wants and what a replay must record,
    because an unbounded edit makes two otherwise identical runs differ.
    """

    text: str
    ttl_frames: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        """What makes this edit meaningless (empty == usable)."""
        out: List[str] = []
        if not self.text.strip():
            out.append("Steer.text must say what to change")
        if self.ttl_frames < 0:
            out.append("Steer.ttl_frames must be >= 0")
        return out

    def __post_init__(self) -> None:
        issues = self.problems()
        if issues:
            raise ValueError("; ".join(issues))


@dataclass(frozen=True)
class StateDigest:
    """The engine's authoritative state, sent alongside a frame.

    This is the half a world model does not have: collision, position,
    objectives. Keeping it on the wire is what lets the engine stay the
    symbolic twin when the model's object permanence drifts.
    """

    ts: float
    world_id: str = ""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    heading: float = 0.0
    entities: Tuple[Dict[str, Any], ...] = ()
    objectives: Tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GenerativeWorldModel(Protocol):
    """A model that renders a world: render → steer → audio.

    Standalone on purpose (see the section comment above). A backend that ALSO
    learns latent dynamics implements ``WorldModel`` as well; the contracts do
    not nest.

    ``mode`` says which half of the frame the model owns, so a consumer never
    has to guess:

      * ``"engine"`` — the model contributes nothing; the engine renders.
      * ``"backdrop"`` — the model fills what the camera cannot reach.
      * ``"frame"``  — the model owns the whole frame.
      * ``"replay"`` — recorded frames; deterministic by construction.
    """

    ok: bool
    mode: str

    def render(self, state: "StateDigest", action: "Action",
               *args: Any, **kwargs: Any) -> Optional["Frame"]:
        """Produce the next frame for this state and action, or None when degraded."""

    def steer(self, text: str, *args: Any, **kwargs: Any) -> bool:
        """Accept a natural-language edit. True when the backend took it.

        A backend that cannot honour a prompt returns False and changes
        nothing — it never silently renders the old world and reports success.
        """

    def audio(self, *args: Any, **kwargs: Any) -> Optional["Frame"]:
        """The audio track for the current world, or None when this backend has none."""


# =============================================================================
# The MOTION contract — animation as bone rotations, never as pixels
# =============================================================================
#
# A pixel-generating world model cannot animate a character. It has no skeleton, no
# collision, no continuity between frames: the silhouette drifts, and a game needs the
# body to be the same body after the camera turns around.
#
# So motion is the one thing a generative model should NOT own frame-to-frame. What it can
# do is PROPOSE motion — a track of bone rotations with explicit times — which an engine
# retargets onto a rig it already has and plays through its own mixer at full framerate,
# with physics, damage and saves intact.
#
# Same rule as everywhere else here: a track is either present and valid or the backend says
# so. There is no "approximate skeleton" that quietly animates the wrong bone.


@dataclass(frozen=True)
class BoneKey:
    """One bone's rotation at one instant. Quaternion is ``(x, y, z, w)``."""

    bone: str
    t: float
    quat: Tuple[float, float, float, float]

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.bone:
            out.append("BoneKey.bone is required")
        if len(self.quat) != 4:
            out.append(f"BoneKey.quat must be (x, y, z, w), got {len(self.quat)} components")
        elif not all(isinstance(v, (int, float)) for v in self.quat):
            out.append("BoneKey.quat components must be numbers")
        if self.t < 0:
            out.append("BoneKey.t must be >= 0")
        return out


@dataclass(frozen=True)
class PoseTrack:
    """A whole animation: which bones, and their keys over time.

    ``bones`` is declared so a consumer can retarget by name BEFORE trusting any key — an
    engine that maps bones at playback time and discovers a typo at frame 40 is an engine
    that plays the previous pose and looks fine.
    """

    track_id: str
    duration: float
    bones: Tuple[str, ...]
    keys: Tuple[BoneKey, ...]
    fps: float = 30.0
    loop: bool = False
    expressions: Tuple["ExpressionKey", ...] = ()
    expression_names: Tuple[str, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.track_id:
            out.append("PoseTrack.track_id is required")
        if self.duration <= 0:
            out.append("PoseTrack.duration must be > 0")
        if self.fps <= 0:
            out.append("PoseTrack.fps must be > 0")
        if not self.bones:
            out.append("PoseTrack declares no bones")
        declared = set(self.bones)
        for key in self.keys:
            out.extend(f"key {key.bone}@{key.t}: {p}" for p in key.problems())
            if key.bone and key.bone not in declared:
                out.append(f"key names undeclared bone {key.bone!r}")
            if self.duration > 0 and key.t > self.duration:
                out.append(f"key {key.bone}@{key.t} is past the track's duration")
        faces = set(self.expression_names)
        for key in self.expressions:
            out.extend(f"expression {key.expression}@{key.t}: {p}" for p in key.problems())
            if key.expression and key.expression not in faces:
                out.append(f"expression key names undeclared expression {key.expression!r}")
            if self.duration > 0 and key.t > self.duration:
                out.append(f"expression {key.expression}@{key.t} is past the track's duration")
        return out

    def bone_keys(self, bone: str) -> Tuple[BoneKey, ...]:
        """Every key for one bone, in time order."""
        return tuple(sorted((k for k in self.keys if k.bone == bone), key=lambda k: k.t))

    def expression_keys(self, expression: str) -> Tuple["ExpressionKey", ...]:
        """Every key for one expression, in time order."""
        return tuple(sorted((k for k in self.expressions if k.expression == expression),
                            key=lambda k: k.t))


@runtime_checkable
class GenerativeMotion(Protocol):
    """A source of animation: ask for an intent, get bone rotations or nothing.

    Deliberately not a ``GenerativeWorldModel``: a pose track is not a frame. The pair
    compose at the engine, which renders the body it always had with motion a model
    proposed.
    """

    ok: bool

    def sample(self, state: "StateDigest", intent: str,
               *args: Any, **kwargs: Any) -> Optional["PoseTrack"]:
        """A track for this intent, or None when this source has none."""

    def names(self) -> Sequence[str]:
        """The intents this source can actually produce."""


@dataclass(frozen=True)
class ExpressionKey:
    """One facial expression at one instant, as a morph weight in [0, 1].

    Expressions are blendshape weights, not bones, because that is what a rigged face
    actually exposes (VRM: ``happy``/``angry``/``sad``/``relaxed``/``surprised``, the
    visemes ``aa ih ou ee oh``, ``blink``). Names are NOT a fixed enum here — a rig may
    declare its own — so a track declares the ones it uses and a key must name one of them.
    """

    expression: str
    t: float
    weight: float

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.expression:
            out.append("ExpressionKey.expression is required")
        if self.t < 0:
            out.append("ExpressionKey.t must be >= 0")
        if not (0.0 <= float(self.weight) <= 1.0):
            out.append(f"ExpressionKey.weight must be in [0,1], got {self.weight!r}")
        return out


@dataclass(frozen=True)
class SequenceStep:
    """One beat of a sequence: which motion, when it starts, how long it runs."""

    intent: str
    start: float
    duration: float
    blend_in: float = 0.0
    expression: str = ""

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.intent:
            out.append("SequenceStep.intent is required")
        if self.start < 0:
            out.append("SequenceStep.start must be >= 0")
        if self.duration <= 0:
            out.append("SequenceStep.duration must be > 0")
        if self.blend_in < 0:
            out.append("SequenceStep.blend_in must be >= 0")
        if self.blend_in > max(self.duration, 0):
            out.append("SequenceStep.blend_in cannot exceed its duration")
        return out

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass(frozen=True)
class SequenceEvent:
    """A cue at a time: sound, a line of dialogue, damage, a spawn.

    The engine fires these on its own clock; a sequence that carries its cues can be
    replayed exactly, which is why they live in the data and not in a callback list.
    """

    t: float
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def problems(self) -> List[str]:
        out: List[str] = []
        if self.t < 0:
            out.append("SequenceEvent.t must be >= 0")
        if not self.kind:
            out.append("SequenceEvent.kind is required")
        return out


@dataclass(frozen=True)
class MotionSequence:
    """Several motions in order, with the cues that belong with them.

    A sequence is data, not a player: the engine asks ``step_at(t)`` which motion is active
    and ``events_between(a, b)`` what fired, and plays the body it already has.
    """

    sequence_id: str
    steps: Tuple[SequenceStep, ...]
    events: Tuple[SequenceEvent, ...] = ()
    loop: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max((s.end for s in self.steps), default=0.0)

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.sequence_id:
            out.append("MotionSequence.sequence_id is required")
        if not self.steps:
            out.append("MotionSequence has no steps")
        for index, step in enumerate(self.steps):
            out.extend(f"step {index}: {p}" for p in step.problems())
        for index, event in enumerate(self.events):
            out.extend(f"event {index}: {p}" for p in event.problems())
        if self.duration > 0 and any(e.t > self.duration for e in self.events):
            out.append("a SequenceEvent fires past the sequence's duration")
        return out

    def step_at(self, t: float) -> Optional[SequenceStep]:
        """The step active at ``t``, or None between steps. Later steps win an overlap."""
        active = [s for s in self.steps if s.start <= t < s.end]
        return max(active, key=lambda s: s.start) if active else None

    def events_between(self, start: float, end: float) -> Tuple[SequenceEvent, ...]:
        """Cues in ``[start, end)``, in time order — the window a frame actually covers."""
        return tuple(sorted((e for e in self.events if start <= e.t < end), key=lambda e: e.t))

