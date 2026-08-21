"""
CodeWorld Environment Adapter
==============================

Maps a codebase environment into the WorldModel's observation/action space.

The adapter observes landmarks (from a code-landmark indexer) and code chunks
(from a code graph), encodes them into a fixed 768-dim embedding, and exposes
a vocabulary of focus actions (e.g. "focus on landmark X", "traverse to chunk Y").

Step execution is a loud NotImplementedError: code-world stepping (editing,
running code) is future work; this adapter is observe/rank-only today.

CONTRACT:
  Conforms to awpredict.contracts.EnvironmentAdapter — stdlib + optional
  numpy only, no dependency on any host framework.

USAGE:
    adapter = CodeWorldAdapter()
    obs = adapter.observe(env_state)  # env_state: {"landmarks": [...], "chunks": [...]}
    actions = adapter.actions()  # List of action ids: "focus_landmark_X", "traverse_chunk_Y"
    before, after = adapter.observe(old_state), adapter.observe(new_state)
    sig = adapter.transition_signature(old_state, action, new_state)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("code_world_adapter")

# Constants from mlp.py (synchronized)
_STATE_DIM = 768
_ACTION_DIM = 128


# ============================================================================
# STATE EMBEDDING (shared idiom with mlp.py)
# ============================================================================

def _hash_state_embedding(text: str) -> List[float]:
    """Deterministic hash-based pseudo-embedding for a state.

    Uses the same SHA256→float scaling as MLPWorldModel._hash_state_embedding
    to ensure embedding parity across adapters and engines. This is NOT a learned
    embedding; it is a deterministic hash that:
      - Produces the same output for the same input (repeatable)
      - Distributes uniformly across [-1, 1] without learned structure
      - Never requires training or model parameters

    The MLPWorldModel learns on top of these embeddings, so a domain-specific
    learned encoder (e.g., code-specific embeddings) would improve prediction
    fidelity, but this default hash-based path is domain-agnostic and auditable.
    """
    h = hashlib.sha256((text or "empty").encode("utf-8")).digest()
    result = []
    for i in range(_STATE_DIM):
        byte_val = h[i % len(h)]
        result.append((byte_val / 127.5) - 1.0)
    return result


# ============================================================================
# STATE DESCRIPTION (deterministic extraction for embedding)
# ============================================================================

def _landmark_descriptor(landmark: Dict[str, Any]) -> str:
    """Extract a deterministic text descriptor from a landmark dict.

    Landmark shape (from prospector LandmarkNode):
      {
        "id": str,
        "name": str,
        "rank": int,
        "purpose": str,
        "files": List[str],
        "tools": List[str],
        "skills": List[str],
        "intent_hints": Dict[str, float],
      }

    Returns a compact, normalized string for hashing.
    """
    name = str(landmark.get("name", "")).strip()
    purpose = str(landmark.get("purpose", "")).strip()
    rank = int(landmark.get("rank", 0) or 0)
    files_count = len(landmark.get("files", []) or [])

    # Deterministic: rank and file count are normalized into buckets to reduce
    # spurious changes on small variations (e.g. rank 42 vs 43).
    rank_bucket = (rank // 10) * 10  # 0-9→0, 10-19→10, etc.
    file_bucket = (files_count // 5) * 5

    parts = [name, purpose, str(rank_bucket), str(file_bucket)]
    return " | ".join(p for p in parts if p)


def _feature_hash_embedding(features: List[str]) -> List[float]:
    """Signed feature hashing: each feature adds ±1 to one of 768 buckets.

    Unlike the whole-string hash above, this is SMOOTH: states sharing features
    (call edges, name tokens) share embedding components, so a rename (few
    features change) stays close while a call-graph change (edge features
    change) moves away. That property is what lets a world model GRADE
    surprise on code transitions instead of answering only same/different —
    measured 2026-08-01: the whole-string variant scored a logic change as
    IDENTICAL (call edges were reduced to a count bucket) and a rename as
    unrelated, i.e. exactly backwards.
    """
    vec = [0.0] * _STATE_DIM
    for feat in features:
        h = hashlib.sha256(feat.encode("utf-8", "replace")).digest()
        idx = int.from_bytes(h[:4], "big") % _STATE_DIM
        sign = 1.0 if h[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _name_tokens(name: str) -> List[str]:
    """snake_case + camelCase token split, lowercased."""
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "").replace("_", " ")
    return [t.lower() for t in parts.split() if t]


def _landmark_features(landmark: Dict[str, Any]) -> List[str]:
    feats = [f"lm-name:{t}" for t in _name_tokens(str(landmark.get("name", "")))]
    for tok in _name_tokens(str(landmark.get("purpose", "")))[:24]:
        feats.append(f"lm-purpose:{tok}")
    rank = int(landmark.get("rank", 0) or 0)
    feats.append(f"lm-rank:{(rank // 10) * 10}")
    for f in sorted(landmark.get("files", []) or [])[:32]:
        feats.append(f"lm-file:{f}")
    return feats


def _chunk_features(chunk: Dict[str, Any]) -> List[str]:
    feats = [f"ch-name:{t}" for t in _name_tokens(str(chunk.get("name", "")))]
    feats.append(f"ch-type:{chunk.get('chunk_type', '')}")
    complexity = int(chunk.get("complexity", 0) or 0)
    feats.append(f"ch-cx:{(complexity // 5) * 5}")
    # EVERY edge is a feature — the call graph is the semantics of the chunk.
    for callee in sorted(chunk.get("calls", []) or [])[:64]:
        feats.append(f"ch-call:{callee}")
    for caller in sorted(chunk.get("called_by", []) or [])[:64]:
        feats.append(f"ch-caller:{caller}")
    return feats


def _state_features(env_state: Dict[str, Any]) -> List[str]:
    feats: List[str] = []
    for lm in env_state.get("landmarks", []) or []:
        if isinstance(lm, dict):
            feats.extend(_landmark_features(lm))
    for ch in env_state.get("chunks", []) or []:
        if isinstance(ch, dict):
            feats.extend(_chunk_features(ch))
    return feats


def _chunk_descriptor(chunk: Dict[str, Any]) -> str:
    """Extract a deterministic text descriptor from a code chunk dict.

    Chunk shape (CodeGraph CodeChunk):
      {
        "name": str,
        "chunk_type": str,
        "calls": List[str],
        "called_by": List[str],
        "complexity": int,
        ...
      }

    Returns a compact, normalized string for hashing.
    """
    name = str(chunk.get("name", "")).strip()
    chunk_type = str(chunk.get("chunk_type", "")).strip()
    complexity = int(chunk.get("complexity", 0) or 0)
    calls_count = len(chunk.get("calls", []) or [])
    called_by_count = len(chunk.get("called_by", []) or [])

    complexity_bucket = (complexity // 5) * 5  # Normalize to 0, 5, 10, ...
    call_bucket = (calls_count // 3) * 3
    caller_bucket = (called_by_count // 3) * 3

    parts = [name, chunk_type, str(complexity_bucket), str(call_bucket), str(caller_bucket)]
    return " | ".join(p for p in parts if p)


def _state_description(env_state: Dict[str, Any]) -> str:
    """Convert environment state to a deterministic description string.

    Combines landmark and chunk descriptors into one hashable text. The state
    description is fed to _hash_state_embedding to produce the 768-dim obs vector.

    Determinism rule: same state always produces the same description, regardless
    of ordering within lists (we sort).
    """
    landmarks = env_state.get("landmarks", []) or []
    chunks = env_state.get("chunks", []) or []

    landmark_descs = sorted(
        _landmark_descriptor(lm) for lm in landmarks if isinstance(lm, dict)
    )
    chunk_descs = sorted(
        _chunk_descriptor(ch) for ch in chunks if isinstance(ch, dict)
    )

    # Sort before joining to make the description independent of input order
    all_descs = landmark_descs + chunk_descs
    if not all_descs:
        return "empty_state"

    return " ; ".join(all_descs)


# ============================================================================
# ACTION VOCABULARY
# ============================================================================

@dataclass
class CodeWorldAction:
    """A discrete action in code-world navigation."""
    action_id: str  # "focus_landmark_<id>", "traverse_chunk_<id>"
    action_type: str  # "focus_landmark", "traverse_chunk"
    target_id: str  # landmark id or chunk id
    target_name: str  # human-readable name


class CodeWorldAdapter:
    """EnvironmentAdapter for code-world environments.

    Maps landmarks and chunks into observation space and exposes a discrete
    action vocabulary. Stepping is a loud NotImplementedError: editing/running
    code arrives with the ADK sandbox pack.

    Attributes:
        domain: "code"
        _last_env_state: Cached last observed state (for actions() to be context-aware)
        _last_obs: Cached last observation vector
    """

    domain = "code"

    def __init__(self) -> None:
        self._last_env_state: Optional[Dict[str, Any]] = None
        self._last_obs: Optional[List[float]] = None

    def observe(self, env_state: Dict[str, Any]) -> List[float]:
        """Convert raw environment state into the observation vector.

        Args:
            env_state: dict with optional "landmarks" and "chunks" keys
              landmarks: List[{id, name, purpose, rank, files, ...}]
              chunks: List[{name, chunk_type, calls, called_by, complexity, ...}]

        Returns:
            List[float] of length 768 (deterministic hash-based embedding)
        """
        if not isinstance(env_state, dict):
            env_state = {}

        self._last_env_state = env_state
        features = _state_features(env_state)
        if not features:
            self._last_obs = _hash_state_embedding("empty_state")
        else:
            self._last_obs = _feature_hash_embedding(features)
        return self._last_obs

    def actions(self) -> Sequence[str]:
        """Return the discrete action vocabulary for the last observed state.

        Actions are context-aware: they are derived from landmarks and chunks
        in the last env_state passed to observe(). If observe() was never called,
        returns an empty list.

        Action ids:
          - "focus_landmark_<landmark_id>": Focus attention on a landmark
          - "traverse_chunk_<chunk_id>": Navigate to a code chunk

        Returns:
            List of action id strings
        """
        if not self._last_env_state:
            return []

        actions = []

        landmarks = self._last_env_state.get("landmarks", []) or []
        for lm in landmarks:
            if isinstance(lm, dict):
                lm_id = lm.get("id", "")
                if lm_id:
                    actions.append(f"focus_landmark_{lm_id}")

        chunks = self._last_env_state.get("chunks", []) or []
        for ch in chunks:
            if isinstance(ch, dict):
                ch_id = ch.get("id", "")
                if ch_id:
                    actions.append(f"traverse_chunk_{ch_id}")

        return actions

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Execute an action in code-world.

        RAISES NotImplementedError with a LOUD message: stepping (editing,
        running code, navigating the runtime) arrives with the ADK sandbox pack.
        This adapter is observe/rank-only in slice 2.

        Exploration safety is deferred: sandboxed code execution requires
        careful isolation that is not part of this slice.
        """
        raise NotImplementedError(
            "🚨 CodeWorld.step() not implemented in slice 2 — "
            "code-world stepping (execute, edit, navigate) arrives with the ADK sandbox pack. "
            "This adapter is observe + rank only. "
            "Use awpredict.surprise() for anomaly detection, "
            "not for real code execution yet."
        )


# ============================================================================
# TRANSITION SIGNATURE (helper for WorldModel training)
# ============================================================================

def transition_signature(
    before_state: Dict[str, Any],
    action: str,
    after_state: Dict[str, Any],
) -> Tuple[List[float], str, List[float]]:
    """Build a (obs, action, next_obs) triple for WorldModel training.

    This helper is called to record observed transitions for training the
    world model. The signature combines:
      - before_obs: 768-dim observation of the codebase before the action
      - action: action id string (e.g. "focus_landmark_X")
      - after_obs: 768-dim observation after the action

    Args:
        before_state: Environment state dict before action
        action: Action id string
        after_state: Environment state dict after action

    Returns:
        (before_obs, action, after_obs) tuple ready for WorldModel.observe()
    """
    adapter = CodeWorldAdapter()
    before_obs = adapter.observe(before_state)
    after_obs = adapter.observe(after_state)
    return (before_obs, action, after_obs)
