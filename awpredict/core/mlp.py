"""
MLPWorldModel — Extracted Neural Transition Predictor
======================================================

Standalone extraction of a neural transition-prediction network from a
larger internal system, with no dependency on that system.

Architecture:
  Input:  concat(state_emb[768], action_emb[128]) = 896
  Hidden: Linear(896, 512) → ReLU → Linear(512, 770)
  Output: next_state[768], reward[1], done_logit[1]

Mode state machine:
  tabular  →  hybrid (after 200 transitions)  →  neural (after 1000 transitions)

Loss:
  0.6 * MSE(next_state) + 0.3 * MSE(reward) + 0.1 * BCE(done)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("awpredict.core.mlp")

_STATE_DIM = 768
_ACTION_DIM = 128
_INPUT_DIM = _STATE_DIM + _ACTION_DIM
_HIDDEN_DIM = 512
_OUTPUT_DIM = _STATE_DIM + 1 + 1

_HYBRID_THRESHOLD = 200
_NEURAL_THRESHOLD = 1000

_LEARNING_RATE = 1e-3
_MAX_TRAIN_STEPS = 500
_BATCH_SIZE = 64

_LOSS_NEXT_STATE = 0.6
_LOSS_REWARD = 0.3
_LOSS_DONE = 0.1

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class Transition:
    """A single observed state transition."""
    state_desc: str
    state_hash: int
    action: str
    next_state_desc: str
    next_state_hash: int
    reward: float
    done: bool
    domain: str = ""
    ts: float = 0.0


class ActionEncoder:
    """Encodes actions into fixed-size 128-dim vectors.

    Known actions get a learnable embedding. Unknown actions get a
    deterministic hash-based pseudo-embedding.
    """

    def __init__(
        self, dim: int = _ACTION_DIM, max_actions: int = 4096
    ) -> None:
        self._dim = dim
        self._max_actions = max_actions
        self._action_to_idx: Dict[str, int] = {}
        self._next_idx = 0
        self._embedding: Optional[Any] = None

    def _ensure_embedding(self) -> None:
        if self._embedding is not None:
            return
        if not _TORCH_AVAILABLE or nn is None:
            return
        self._embedding = nn.Embedding(self._max_actions, self._dim)
        nn.init.normal_(self._embedding.weight, std=0.1)

    def register_action(self, action_str: str) -> int:
        """Register an action and return its index."""
        if action_str in self._action_to_idx:
            return self._action_to_idx[action_str]
        if self._next_idx >= self._max_actions:
            return -1
        idx = self._next_idx
        self._action_to_idx[action_str] = idx
        self._next_idx += 1
        return idx

    def encode(self, action: Any) -> Any:
        """Encode an action to a dim-dimensional tensor or list."""
        action_str = self._normalize_action(action)

        if not _TORCH_AVAILABLE:
            return self._hash_encode(action_str)

        self._ensure_embedding()

        idx = self._action_to_idx.get(action_str, -1)
        if idx < 0:
            idx = self.register_action(action_str)

        if idx >= 0 and self._embedding is not None:
            idx_tensor = torch.tensor([idx], dtype=torch.long)
            return self._embedding(idx_tensor).squeeze(0)

        return self._hash_encode_tensor(action_str)

    def _hash_encode_tensor(self, action_str: str) -> Any:
        """Deterministic hash-based encoding as a torch tensor."""
        raw = self._hash_encode(action_str)
        return torch.tensor(raw, dtype=torch.float32)

    def _hash_encode(self, action_str: str) -> List[float]:
        """Deterministic hash-based encoding as a list of floats."""
        h = hashlib.sha256(action_str.encode("utf-8")).digest()
        result = []
        for i in range(self._dim):
            byte_val = h[i % len(h)]
            result.append((byte_val / 127.5) - 1.0)
        return result

    @staticmethod
    def _normalize_action(action: Any) -> str:
        """Convert any action type to a string key."""
        if isinstance(action, str):
            return action
        if isinstance(action, (int, float, bool)):
            return str(action)
        if isinstance(action, tuple):
            return str(action)
        return str(action)


class TransitionEncoder:
    """Small MLP that predicts (next_state_emb, reward, done) from
    (state_emb, action_emb).

    Architecture:
      Linear(896, 512) -> ReLU -> Linear(512, 770)
      Output: next_state[768] + reward[1] + done_logit[1]
    """

    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._available = _TORCH_AVAILABLE

    @property
    def is_available(self) -> bool:
        return self._available and self._model is not None

    def build(self) -> bool:
        """Initialize the PyTorch model. Returns True on success."""
        if not _TORCH_AVAILABLE or nn is None:
            log.debug("PyTorch not available")
            return False

        self._model = nn.Sequential(
            nn.Linear(_INPUT_DIM, _HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(_HIDDEN_DIM, _OUTPUT_DIM),
        )
        self._available = True
        param_count = sum(p.numel() for p in self._model.parameters())
        log.info(f"TransitionEncoder built: {param_count} params")
        return True

    def predict(
        self, state_emb: Any, action_emb: Any
    ) -> Tuple[Any, float, float]:
        """Predict next state embedding, reward, and done logit.

        Args:
            state_emb: Tensor of shape (768,)
            action_emb: Tensor of shape (128,)

        Returns:
            (next_state_emb[768], reward_scalar, done_logit)
        """
        if not self.is_available or self._model is None:
            raise RuntimeError("TransitionEncoder not built")

        with torch.no_grad():
            x = torch.cat(
                [state_emb, action_emb], dim=-1
            ).unsqueeze(0)
            out = self._model(x).squeeze(0)
            next_state = out[:_STATE_DIM]
            reward = out[_STATE_DIM].item()
            done_logit = out[_STATE_DIM + 1].item()

        return next_state, reward, done_logit

    def train_step(
        self,
        state_embs: Any,
        action_embs: Any,
        target_next_states: Any,
        target_rewards: Any,
        target_dones: Any,
        optimizer: Any,
    ) -> float:
        """Single training step. Returns combined loss value."""
        x = torch.cat([state_embs, action_embs], dim=-1)
        out = self._model(x)

        pred_next = out[:, :_STATE_DIM]
        pred_reward = out[:, _STATE_DIM]
        pred_done = out[:, _STATE_DIM + 1]

        loss_next = nn.functional.mse_loss(pred_next, target_next_states)
        loss_reward = nn.functional.mse_loss(pred_reward, target_rewards)
        loss_done = nn.functional.binary_cross_entropy_with_logits(
            pred_done, target_dones
        )

        loss = (
            _LOSS_NEXT_STATE * loss_next
            + _LOSS_REWARD * loss_reward
            + _LOSS_DONE * loss_done
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def save(self, path: Path) -> None:
        """Save model weights to disk."""
        if self._model is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), str(path))
        except Exception as e:
            log.debug(f"Save model failed: {e}")

    def load(self, path: Path) -> bool:
        """Load model weights from disk."""
        if not path.exists():
            return False
        if not self.is_available:
            if not self.build():
                return False
        try:
            state = torch.load(
                str(path), map_location="cpu", weights_only=True
            )
            self._model.load_state_dict(state)
            return True
        except Exception as e:
            log.debug(f"Load model failed: {e}")
            return False


class MLPWorldModel:
    """Standalone MLP-based world model with tabular cold-start fallback.

    Provides neural transition prediction for unseen (state, action) pairs
    with a tabular fallback. Mode state machine: tabular → hybrid → neural.

    Public API:
      observe(state, action, next_state, reward, done)
      predict(state, action) -> (next_state, reward, done)
      train_step(...) / train_sync(...)
      save(path) / load(path)
    """

    def __init__(
        self,
        state_dim: int = _STATE_DIM,
        action_dim: int = _ACTION_DIM,
        hybrid_threshold: int = _HYBRID_THRESHOLD,
        neural_threshold: int = _NEURAL_THRESHOLD,
    ) -> None:
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hybrid_threshold = hybrid_threshold
        self._neural_threshold = neural_threshold

        self._transitions: Dict[Tuple[int, Any], list] = {}
        self._state_values: Dict[int, float] = {}

        self._encoder = TransitionEncoder()
        self._action_encoder = ActionEncoder(dim=action_dim)
        self._neural_available = False

        self._mode = "tabular"
        self._last_train_time = 0.0
        self._train_count = 0
        self._state_desc_cache: Dict[int, str] = {}
        self._transition_count = 0

    @property
    def mode(self) -> str:
        return self._mode

    # -- WorldModel contract surface (awpredict.contracts.WorldModel) --------
    # The extracted source predates the contract; these are thin, honest
    # aliases — no new modeling behavior.

    @property
    def ok(self) -> bool:
        """Tabular mode needs nothing; the engine is always operable."""
        return True

    def encode(self, obs: Any, cond: Any = None) -> Optional[List[float]]:
        """Observation → the embedding the neural path consumes.

        ``obs`` is a state DESCRIPTION string (or anything str()-able); the
        MLP engine's latent space IS the deterministic hash embedding.
        ``cond`` is accepted for contract parity and unused here.
        """
        del cond
        if obs is None:
            return None
        return self._hash_state_embedding(str(obs))

    def surprise(
        self,
        obs: int,
        action: Any,
        next_obs: int,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[float]:
        """Discrete prediction error: 0.0 = predicted this next state,
        1.0 = predicted a different one, None = no prediction yet (unseen
        state/action — the caller decides whether ignorance is surprising)."""
        del args, kwargs
        predicted = self.predict(obs, action)
        if predicted is None:
            return None
        return 0.0 if predicted[0] == next_obs else 1.0

    def train_step(
        self, transitions: Optional[List[Transition]] = None, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Contract alias over train_sync(); returns a status dict or None."""
        del kwargs
        trained = self.train_sync(transitions)
        if not trained:
            return None
        return {"trained": True, "train_count": self._train_count, "mode": self._mode}

    def plan(self, *args: Any, **kwargs: Any) -> None:
        """This engine has NO planner (the source used an external CEMPlanner).

        Loud by design: callers wanting latent planning use LeWorldModel or
        wrap this engine with a planner — a silent empty plan would read as
        "no good actions exist", which is a lie.
        """
        del args, kwargs
        log.warning(
            "MLPWorldModel.plan: no planner in this engine; use "
            "awpredict.core.lewm.LeWorldModel or an external planner"
        )
        return None

    def observe(
        self,
        state_hash: int,
        action: Any,
        next_state_hash: int,
        reward: float,
        done: bool,
        state_desc: str = "",
        next_state_desc: str = "",
    ) -> None:
        """Record an observed transition."""
        key = (state_hash, self._action_key(action))
        records = self._transitions.setdefault(key, [])
        for rec in records:
            if rec["next_state_hash"] == next_state_hash:
                rec["reward"] = (
                    (rec["reward"] * rec["count"] + reward) /
                    (rec["count"] + 1)
                )
                rec["count"] += 1
                return

        records.append({
            "next_state_hash": next_state_hash,
            "reward": reward,
            "done": done,
            "count": 1,
        })

        if state_desc:
            self._state_desc_cache[state_hash] = state_desc
        if next_state_desc:
            self._state_desc_cache[next_state_hash] = next_state_desc

        action_str = self._action_key_str(action)
        self._action_encoder.register_action(action_str)

        self._transition_count += 1
        self._check_mode_transition()

    def predict(
        self, state_hash: int, action: Any
    ) -> Optional[Tuple[int, float, bool]]:
        """Predict outcome of (state, action).

        In tabular mode: exact match only.
        In hybrid mode: try tabular first, then neural fallback.
        In neural mode: neural prediction for everything.
        """
        tabular = self._tabular_predict(state_hash, action)

        if self._mode == "tabular":
            return tabular

        if self._mode == "hybrid":
            if tabular is not None:
                return tabular
            return self._neural_predict(state_hash, action)

        neural = self._neural_predict(state_hash, action)
        return neural if neural is not None else tabular

    def _tabular_predict(
        self, state_hash: int, action: Any
    ) -> Optional[Tuple[int, float, bool]]:
        """Tabular prediction (exact match only)."""
        key = (state_hash, self._action_key(action))
        records = self._transitions.get(key)
        if not records:
            return None
        best = max(records, key=lambda r: r["count"])
        return (best["next_state_hash"], best["reward"], best["done"])

    def _neural_predict(
        self, state_hash: int, action: Any
    ) -> Optional[Tuple[int, float, bool]]:
        """Use the neural encoder to predict a transition."""
        if not self._neural_available or not self._encoder.is_available:
            return None

        state_desc = (
            self._state_desc_cache.get(state_hash, str(state_hash))
        )
        state_emb_list = self._hash_state_embedding(state_desc)
        if state_emb_list is None:
            return None

        action_emb = self._action_encoder.encode(action)
        if not isinstance(action_emb, list):
            action_emb_list = action_emb.detach().tolist()
        else:
            action_emb_list = action_emb

        try:
            state_emb = torch.tensor(
                state_emb_list, dtype=torch.float32
            )
            if not isinstance(action_emb, torch.Tensor):
                action_emb_t = torch.tensor(
                    action_emb_list, dtype=torch.float32
                )
            else:
                action_emb_t = action_emb

            next_state_emb, reward, done_logit = (
                self._encoder.predict(state_emb, action_emb_t)
            )

            import math
            done_prob = 1.0 / (1.0 + math.exp(-done_logit))
            done = done_prob > 0.5

            emb_bytes = next_state_emb.numpy().tobytes()
            predicted_hash = hash(emb_bytes)

            return (predicted_hash, float(reward), done)

        except Exception as e:
            log.debug(f"Neural predict failed: {e}")
            return None

    def train_sync(
        self, transitions: Optional[List[Transition]] = None
    ) -> bool:
        """Synchronous training on transitions.

        If transitions provided, uses those. Otherwise returns False.
        """
        if not self._encoder.is_available:
            if not self._encoder.build():
                return False

        if not _TORCH_AVAILABLE:
            return False

        if transitions is None or len(transitions) < 10:
            return False

        state_embs = []
        action_embs = []
        target_next_states = []
        target_rewards = []
        target_dones = []

        for t in transitions:
            s_emb = self._hash_state_embedding(t.state_desc)
            ns_emb = self._hash_state_embedding(t.next_state_desc)
            if s_emb is None or ns_emb is None:
                continue

            a_emb = self._action_encoder.encode(t.action)
            if not isinstance(a_emb, list):
                a_emb = a_emb.detach().tolist()

            state_embs.append(s_emb)
            action_embs.append(a_emb)
            target_next_states.append(ns_emb)
            target_rewards.append(t.reward)
            target_dones.append(1.0 if t.done else 0.0)

        if len(state_embs) < 10:
            return False

        state_t = torch.tensor(state_embs, dtype=torch.float32)
        action_t = torch.tensor(action_embs, dtype=torch.float32)
        next_t = torch.tensor(target_next_states, dtype=torch.float32)
        reward_t = torch.tensor(target_rewards, dtype=torch.float32)
        done_t = torch.tensor(target_dones, dtype=torch.float32)

        optimizer = torch.optim.Adam(
            self._encoder._model.parameters(), lr=_LEARNING_RATE
        )

        n = len(state_embs)
        losses = []

        for step in range(_MAX_TRAIN_STEPS):
            idx = torch.randint(0, n, (min(_BATCH_SIZE, n),))
            loss = self._encoder.train_step(
                state_t[idx], action_t[idx],
                next_t[idx], reward_t[idx], done_t[idx],
                optimizer,
            )
            losses.append(loss)
            if step > 50 and loss < 0.01:
                break

        self._neural_available = True
        self._last_train_time = time.time()
        self._train_count += 1

        return True

    def save(self, path: Path) -> None:
        """Save model weights to disk."""
        self._encoder.save(path)

    def load(self, path: Path) -> bool:
        """Load model weights from disk."""
        return self._encoder.load(path)

    def get_transition_count(self) -> int:
        return self._transition_count

    def record_state_value(self, state_hash: int, value: float) -> None:
        old = self._state_values.get(state_hash)
        if old is not None:
            self._state_values[state_hash] = 0.7 * old + 0.3 * value
        else:
            self._state_values[state_hash] = value

    def get_state_value(self, state_hash: int) -> Optional[float]:
        return self._state_values.get(state_hash)

    def get_status(self) -> Dict[str, Any]:
        """Get world model status for diagnostics."""
        return {
            "mode": self._mode,
            "transition_count": self._transition_count,
            "neural_available": self._neural_available,
            "train_count": self._train_count,
            "last_train_time": self._last_train_time,
            "hybrid_threshold": self._hybrid_threshold,
            "neural_threshold": self._neural_threshold,
            "tabular_entries": sum(
                len(v) for v in self._transitions.values()
            ),
            "known_actions": len(
                self._action_encoder._action_to_idx
            ),
            "cached_state_descs": len(self._state_desc_cache),
        }

    def _check_mode_transition(self) -> None:
        """Check if we should transition modes."""
        count = self._transition_count
        old_mode = self._mode

        if count >= self._neural_threshold and self._neural_available:
            self._mode = "neural"
        elif count >= self._hybrid_threshold and self._neural_available:
            self._mode = "hybrid"
        else:
            self._mode = "tabular"

        if self._mode != old_mode:
            log.info(
                f"Mode transition: {old_mode} → {self._mode} "
                f"(transitions={count})"
            )

    @staticmethod
    def _action_key(action: Any) -> Any:
        """Make action hashable for dict keys."""
        if isinstance(action, (str, int, float, tuple, bool, type(None))):
            return action
        try:
            return hash(action)
        except TypeError:
            return str(action)

    @staticmethod
    def _action_key_str(action: Any) -> str:
        """Convert action to string."""
        if isinstance(action, str):
            return action
        return str(action)

    @staticmethod
    def _hash_state_embedding(text: str) -> List[float]:
        """Deterministic hash-based pseudo-embedding for a state."""
        h = hashlib.sha256((text or "empty").encode("utf-8")).digest()
        result = []
        for i in range(_STATE_DIM):
            byte_val = h[i % len(h)]
            result.append((byte_val / 127.5) - 1.0)
        return result
