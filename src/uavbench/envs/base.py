from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from uavbench.scenarios.schema import ScenarioConfig


class RejectReason(str, Enum):
    """Why a proposed move was rejected.  Enforces EC-1.

    String values match the existing info["reject_reason"] convention
    for backward compatibility with downstream code and tests.
    """
    NONE = "none"
    BUILDING = "building"
    NO_FLY = "no_fly"
    FIRE = "fire"
    TRAFFIC = "traffic"
    RESTRICTION_ZONE = "restriction_zone"
    FORCED_BLOCK = "forced_block"
    OUT_OF_BOUNDS = "out_of_bounds"
    INTRUDER = "intruder"
    MOVING_TARGET = "moving_target"
    DYNAMIC_NFZ = "dynamic_nfz"
    TRAFFIC_CLOSURE = "traffic_closure"


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

        # Usually a design bug: both True at once.  Log and resolve to terminated.
        if terminated and truncated:
            import warnings
            warnings.warn(
                "Both terminated and truncated are True. "
                "Resolving to terminated=True, truncated=False for Gymnasium compliance.",
                RuntimeWarning,
                stacklevel=2,
            )
            truncated = False

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

    def log_event(self, event_type: str, _step_override: int | None = None, **payload: Any) -> None:
        """Structured event logging (for metrics/robustness/debugging).

        Convention:
        - By default, event["step"] = self._step_count (the *completed* step count,
          i.e. pre-increment when called from _step_impl before base.step() finishes).
        - Pass _step_override=N to record an authoritative, unambiguous step index.
          Use this when the caller already knows the authoritative step (e.g., the
          next-step index passed to _maybe_trigger_interdictions), so that
          event["step"] == N exactly as specified in the paper protocol (t1, t2).
        - Keep payload JSON-friendly where possible.
        """
        step = int(_step_override) if _step_override is not None else self._step_count
        self._events.append(
            {
                "step": step,
                "type": str(event_type),
                "payload": payload,
            }
        )


