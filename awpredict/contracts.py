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
