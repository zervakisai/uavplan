from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from uavbench.scenarios.schema import ScenarioConfig


class UAVBenchEnv(gym.Env, ABC):
    """Base class for all UAVBench environments.

    Responsibilities (framework-level, not domain-specific):
    - Store ScenarioConfig
    - Enforce Gymnasium seeding discipline via per-env np.random.Generator
    - Count steps (per episode)
    - Log trajectory + structured events (JSON-safe, metrics-friendly)
    - Provide a common dynamic-state contract for planners/viz
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
            self._rng = np.random.default_rng(int(seed))

        self._step_count = 0
        self._trajectory = []
        self._events = []

        obs, info = self._reset_impl(options)

        # V&V guards: make sure reset returns the Gymnasium tuple.
        if info is None:
            info = {}
        if not isinstance(info, Mapping):
            raise TypeError(f"_reset_impl must return (obs, info: Mapping), got info type={type(info)}")

        return obs, dict(info)

    def step(self, action: Any):
        """Common step:
        - call domain logic (_step_impl)
        - enforce boolean flags + basic invariants
        - increment step counter exactly once
        - log transition (trajectory)
        """
        obs, reward, terminated, truncated, info = self._step_impl(action)

        # V&V guard: domain must return booleans, not truthy numpy scalars etc.
        terminated = bool(terminated)
        truncated = bool(truncated)

        # Usually a design bug: both True at once (treat as hard fail for benchmark consistency).
        if terminated and truncated:
            raise RuntimeError(
                "Both terminated and truncated are True. Check domain termination logic."
            )

        # Normalize info
        if info is None:
            info = {}
        if not isinstance(info, Mapping):
            raise TypeError(f"_step_impl must return info as Mapping, got {type(info)}")

        # Step index: count completed transitions (1..N). Logging uses the *post-step* index.
        self._step_count += 1

        # Log with stable Python-native scalars where possible.
        self._trajectory.append(
            {
                "step": self._step_count,
                "action": action,
                # Keep obs as ndarray for ML usage; user can serialize externally if needed.
                "obs": np.asarray(obs),
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": dict(info),
            }
        )

        return obs, float(reward), terminated, truncated, dict(info)

    # ----------------- Abstract hooks -----------------

    @abstractmethod
    def _reset_impl(self, options: dict[str, Any] | None):
        """Domain-specific reset. Must return (obs, info)."""
        raise NotImplementedError

    @abstractmethod
    def _step_impl(self, action: Any):
        """Domain-specific step. Must return (obs, reward, terminated, truncated, info)."""
        raise NotImplementedError

    # ----------------- Contract for planners & visualization -----------------

    @abstractmethod
    def get_dynamic_state(self) -> dict[str, Any]:
        """Return a dict with all dynamic layers needed by planners/viz.

        Benchmark contract (keys should exist even if value is None):
        - fire_mask: np.ndarray[H,W] bool | None
        - burned_mask: np.ndarray[H,W] bool | None
        - smoke_mask: np.ndarray[H,W] float | None
        - traffic_positions: np.ndarray[N,2] | None   (y,x) or (row,col) convention must be consistent across project
        - moving_target_pos: np.ndarray[2] | None
        - intruder_positions: np.ndarray[M,2] | None
        - dynamic_nfz_mask: np.ndarray[H,W] bool | None
        """
        raise NotImplementedError

    # ----------------- Helpers for metrics/analysis -----------------

    @property
    def step_count(self) -> int:
        """Number of completed transitions in the current episode."""
        return self._step_count

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """Full per-step log: action, obs, reward, termination flags, info."""
        # Shallow copy to reduce accidental external mutation.
        return list(self._trajectory)

    @property
    def events(self) -> list[dict[str, Any]]:
        """High-level events (collisions, violations, replans, etc.)."""
        return list(self._events)

    def log_event(self, event_type: str, **payload: Any) -> None:
        """Structured event logging (for metrics/robustness/debugging).

        Convention:
        - event is associated with the *current* step index (post-step).
        - keep payload JSON-friendly where possible (cast numpy scalars to Python types upstream).
        """
        self._events.append(
            {
                "step": self._step_count,
                "type": str(event_type),
                "payload": payload,
            }
        )


