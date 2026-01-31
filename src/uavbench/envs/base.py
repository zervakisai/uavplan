from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np

from uavbench.scenarios.schema import ScenarioConfig


class UAVBenchEnv(gym.Env, ABC):
    """Base class for all UAVBench environments.

    Common responsibilities:
    - stores ScenarioConfig
    - seeding discipline via per-env np.random.Generator
    - counts steps (per episode)
    - logs trajectory + structured events
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: ScenarioConfig):
        super().__init__()
        self.config = config

        # Per-instance RNG. Must be the ONLY source of randomness in domain logic.
        self._rng: np.random.Generator = np.random.default_rng()

        # Episode bookkeeping
        self._step_count: int = 0
        self._trajectory: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []

    # ----------------- Gym API wrappers -----------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Common reset:
        - follow Gymnasium seeding discipline
        - reset per-episode bookkeeping
        - delegate domain-specific init to _reset_impl
        """
        super().reset(seed=seed)

        # If seed is provided, re-seed our per-env RNG deterministically.
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._trajectory = []
        self._events = []

        return self._reset_impl(options)

    def step(self, action: Any):
        """Common step:
        - call domain logic (_step_impl)
        - increment step counter exactly once
        - log transition (trajectory)
        """
        obs, reward, terminated, truncated, info = self._step_impl(action)

        # V&V guard: domain must return booleans, not truthy numpy scalars etc.
        terminated = bool(terminated)
        truncated = bool(truncated)

        # In Gymnasium, terminated/truncated should not both be True.
        # It's not strictly forbidden in all cases, but it's usually a design bug.
        if terminated and truncated:
            raise RuntimeError("Both terminated and truncated are True. Check domain termination logic.")

        self._step_count += 1

        # Log with Python-native types for stability (tests, JSON, metrics).
        self._trajectory.append(
            {
                "step": self._step_count,
                "action": action,
                "obs": np.asarray(obs),           # keep as ndarray for ML usage
                "reward": float(reward),          # Python float
                "terminated": terminated,
                "truncated": truncated,
                "info": dict(info) if info is not None else {},
            }
        )
        return obs, float(reward), terminated, truncated, info

    # ----------------- Abstract hooks -----------------

    @abstractmethod
    def _reset_impl(self, options: dict[str, Any] | None):
        """Domain-specific reset. Must return (obs, info)."""
        raise NotImplementedError

    @abstractmethod
    def _step_impl(self, action: Any):
        """Domain-specific step. Must return (obs, reward, terminated, truncated, info)."""
        raise NotImplementedError

    # ----------------- Helpers for metrics/analysis -----------------

    @property
    def step_count(self) -> int:
        """Number of completed transitions in the current episode."""
        return self._step_count

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """Full per-step log: action, obs, reward, termination flags, info."""
        # Return a shallow copy to reduce accidental external mutation.
        return list(self._trajectory)

    @property
    def events(self) -> list[dict[str, Any]]:
        """High-level events (collisions, violations, etc.)."""
        return list(self._events)

    def log_event(self, event_type: str, **payload: Any) -> None:
        """Structured event logging (for metrics/robustness/debugging)."""
        self._events.append(
            {
                "step": self._step_count,  # event is associated with current step index
                "type": str(event_type),
                "payload": payload,
            }
        )

