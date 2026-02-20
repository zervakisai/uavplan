"""PlannerAdapter + ReplanPolicy: dynamic replanning with causal logging.

PlannerAdapter
    Wraps any BasePlanner with UpdateBus integration:
    - Receives obstacle updates via bus subscription
    - Maintains merged obstacle state
    - Delegates plan/replan to underlying planner
    - Logs every replan with unique ID, trigger reason, cost delta

ReplanPolicy
    Determines WHEN to replan based on configurable triggers:
    - CADENCE — every N steps (scheduled)
    - EVENT — immediately on hard conflict detection
    - RISK_SPIKE — when risk integral exceeds threshold
    - FORCED — injected deterministically for testing (≥2 per episode)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np

from uavbench.planners.base import BasePlanner, GridPos, PlanResult
from uavbench.updates.bus import EventType, UpdateEvent, UpdateBus
from uavbench.updates.conflict import ConflictDetector, Conflict


# ─────────────────────────────────────────────────────────────────────────────
# Replan trigger types
# ─────────────────────────────────────────────────────────────────────────────

class ReplanTrigger(str, Enum):
    """Why a replan was initiated."""
    CADENCE = "cadence"           # scheduled interval
    EVENT = "event"               # hard obstacle on path
    RISK_SPIKE = "risk_spike"     # risk integral exceeded threshold
    FORCED = "forced"             # injected by ForcedReplanScheduler
    INITIAL = "initial"           # first plan


# ─────────────────────────────────────────────────────────────────────────────
# Replan log entry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplanRecord:
    """Immutable record of one replan event."""
    replan_id: str
    step: int
    trigger: ReplanTrigger
    reason: str
    position: GridPos
    old_path_length: int
    new_path_length: int
    old_path_cost: float
    new_path_cost: float
    compute_time_ms: float
    expansions: int
    success: bool
    conflicts: list[Conflict] = field(default_factory=list)
    parent_event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "replan_id": self.replan_id,
            "step": self.step,
            "trigger": self.trigger.value,
            "reason": self.reason,
            "position": list(self.position),
            "old_path_length": self.old_path_length,
            "new_path_length": self.new_path_length,
            "old_path_cost": self.old_path_cost,
            "new_path_cost": self.new_path_cost,
            "compute_time_ms": self.compute_time_ms,
            "expansions": self.expansions,
            "success": self.success,
            "conflict_count": len(self.conflicts),
            "parent_event_id": self.parent_event_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ReplanPolicy — decides WHEN to replan
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReplanPolicyConfig:
    """Configuration for replan trigger logic."""
    cadence_interval: int = 10       # replan every N steps
    risk_threshold: float = 0.8      # risk level triggering replan
    risk_window: int = 5             # steps of high risk before trigger
    min_conflict_severity: float = 0.5  # minimum severity to trigger EVENT replan
    max_replans_per_episode: int = 50   # safety cap


class ReplanPolicy:
    """Evaluates whether a replan should be triggered this step.

    Checks in priority order:
      1. FORCED — forced injection schedule (guaranteed delivery)
      2. EVENT — hard conflict detected on current path
      3. RISK_SPIKE — sustained high-risk exposure
      4. CADENCE — scheduled interval elapsed
    """

    def __init__(self, config: ReplanPolicyConfig | None = None) -> None:
        self.cfg = config or ReplanPolicyConfig()
        self._steps_since_replan: int = 0
        self._high_risk_steps: int = 0
        self._total_replans: int = 0
        self._forced_steps: set[int] = set()

    def add_forced_replan(self, step: int) -> None:
        """Register a step at which a replan MUST occur."""
        self._forced_steps.add(step)

    def evaluate(
        self,
        step: int,
        conflicts: list[Conflict],
        risk_at_pos: float = 0.0,
    ) -> tuple[bool, ReplanTrigger, str]:
        """Decide whether to replan this step.

        Returns
        -------
        (should_replan, trigger, reason)
        """
        self._steps_since_replan += 1

        if self._total_replans >= self.cfg.max_replans_per_episode:
            return False, ReplanTrigger.CADENCE, "max_replans_reached"

        # 1. Forced
        if step in self._forced_steps:
            self._forced_steps.discard(step)
            return True, ReplanTrigger.FORCED, f"forced_at_step_{step}"

        # 2. Event — hard conflict on path
        hard_conflicts = [c for c in conflicts if c.severity >= self.cfg.min_conflict_severity]
        if hard_conflicts:
            nearest = hard_conflicts[0]
            return True, ReplanTrigger.EVENT, (
                f"{nearest.obstacle_type}_at_({nearest.position[0]},{nearest.position[1]})"
                f"_in_{nearest.distance_steps}_steps"
            )

        # 3. Risk spike
        if risk_at_pos >= self.cfg.risk_threshold:
            self._high_risk_steps += 1
            if self._high_risk_steps >= self.cfg.risk_window:
                self._high_risk_steps = 0
                return True, ReplanTrigger.RISK_SPIKE, f"risk_{risk_at_pos:.2f}_sustained"
        else:
            self._high_risk_steps = max(0, self._high_risk_steps - 1)

        # 4. Cadence
        if self._steps_since_replan >= self.cfg.cadence_interval:
            return True, ReplanTrigger.CADENCE, f"interval_{self.cfg.cadence_interval}"

        return False, ReplanTrigger.CADENCE, ""

    def record_replan(self) -> None:
        """Called after a replan is executed."""
        self._steps_since_replan = 0
        self._total_replans += 1


# ─────────────────────────────────────────────────────────────────────────────
# PlannerAdapter — wraps BasePlanner with bus integration
# ─────────────────────────────────────────────────────────────────────────────

class PlannerAdapter:
    """Wraps any BasePlanner with UpdateBus integration and causal logging.

    Responsibilities:
    - Subscribe to UpdateBus for obstacle/constraint/risk events
    - Maintain merged dynamic obstacle mask
    - Provide ``plan()`` and ``try_replan()`` with full logging
    - Track replan history as ``ReplanRecord`` list

    Parameters
    ----------
    planner : BasePlanner
        Underlying planner instance.
    bus : UpdateBus
        Event bus to subscribe to.
    conflict_detector : ConflictDetector
        For path-obstacle intersection checks.
    replan_policy : ReplanPolicy, optional
        Custom replan trigger policy.
    """

    def __init__(
        self,
        planner: BasePlanner,
        bus: UpdateBus,
        conflict_detector: ConflictDetector,
        replan_policy: ReplanPolicy | None = None,
    ) -> None:
        self.planner = planner
        self.bus = bus
        self.detector = conflict_detector
        self.policy = replan_policy or ReplanPolicy()
        self.H, self.W = planner.H, planner.W

        # Dynamic state — accumulated from bus events
        self._obstacle_mask = np.zeros((self.H, self.W), dtype=bool)
        self._nfz_mask = planner.no_fly.copy()
        self._risk_map = np.zeros((self.H, self.W), dtype=np.float32)
        self._vehicle_positions: list[tuple[int, int]] = []

        # Path state
        self._current_path: list[GridPos] = []
        self._current_goal: GridPos | None = None
        self._path_cost: float = 0.0

        # Replan history
        self._replan_log: list[ReplanRecord] = []

        # Subscribe to obstacle and constraint events
        self.bus.subscribe(EventType.OBSTACLE, self._on_obstacle)
        self.bus.subscribe(EventType.CONSTRAINT, self._on_constraint)
        self.bus.subscribe(EventType.RISK, self._on_risk)

    # ── Bus callbacks ─────────────────────────────────────────────

    def _on_obstacle(self, event: UpdateEvent) -> None:
        """Handle obstacle event (moving vehicle, vessel, etc.)."""
        if event.mask is not None:
            self._obstacle_mask |= event.mask.astype(bool)
        if event.position is not None:
            self._vehicle_positions.append(event.position)
            # Keep only last 20 vehicle positions
            if len(self._vehicle_positions) > 20:
                self._vehicle_positions = self._vehicle_positions[-20:]

    def _on_constraint(self, event: UpdateEvent) -> None:
        """Handle constraint event (NFZ activation, corridor block)."""
        if event.mask is not None:
            self._nfz_mask |= event.mask.astype(bool)

    def _on_risk(self, event: UpdateEvent) -> None:
        """Handle risk field update."""
        if event.mask is not None:
            # Risk mask is float; take element-wise max
            risk = event.mask.astype(np.float32)
            self._risk_map = np.maximum(self._risk_map, risk)

    # ── Dynamic state management ──────────────────────────────────

    def update_dynamic_state(
        self,
        *,
        obstacle_mask: np.ndarray | None = None,
        nfz_mask: np.ndarray | None = None,
        risk_map: np.ndarray | None = None,
        vehicle_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        """Directly set dynamic state (alternative to bus subscription)."""
        if obstacle_mask is not None:
            self._obstacle_mask = obstacle_mask.astype(bool)
        if nfz_mask is not None:
            self._nfz_mask = nfz_mask.astype(bool)
        if risk_map is not None:
            self._risk_map = risk_map.astype(np.float32)
        if vehicle_positions is not None:
            self._vehicle_positions = list(vehicle_positions)

    def get_merged_obstacles(self) -> np.ndarray:
        """Build merged obstacle mask from all dynamic layers."""
        return self.detector.merge_obstacles(
            extra_mask=self._obstacle_mask,
            vehicle_positions=self._vehicle_positions,
        )

    # ── Planning API ──────────────────────────────────────────────

    def plan(
        self,
        start: GridPos,
        goal: GridPos,
        cost_map: np.ndarray | None = None,
    ) -> PlanResult:
        """Initial plan with obstacle avoidance.

        Builds a merged obstacle map and plans on it.
        """
        self._current_goal = goal

        # Merge all dynamic obstacles into no_fly for the planner
        merged = self.planner.no_fly | self._obstacle_mask | self._nfz_mask
        # Temporarily swap no_fly
        original_nfz = self.planner.no_fly
        self.planner.no_fly = merged

        t0 = time.perf_counter()
        result = self.planner.plan(start, goal, cost_map)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.planner.no_fly = original_nfz

        if result.success:
            self._current_path = list(result.path)
            self._path_cost = float(len(result.path))

        # Log as initial plan
        record = ReplanRecord(
            replan_id=uuid.uuid4().hex[:12],
            step=0,
            trigger=ReplanTrigger.INITIAL,
            reason="initial_plan",
            position=start,
            old_path_length=0,
            new_path_length=len(result.path),
            old_path_cost=0.0,
            new_path_cost=float(len(result.path)),
            compute_time_ms=elapsed_ms,
            expansions=result.expansions,
            success=result.success,
        )
        self._replan_log.append(record)

        return PlanResult(
            path=result.path,
            success=result.success,
            compute_time_ms=elapsed_ms,
            expansions=result.expansions,
            replans=0,
            reason=result.reason,
        )

    def step_check(
        self,
        current_pos: GridPos,
        path_index: int,
        step: int,
        risk_at_pos: float = 0.0,
    ) -> tuple[bool, ReplanTrigger, str, list[Conflict]]:
        """Check if replan is needed at current step.

        Returns
        -------
        (should_replan, trigger, reason, conflicts)
        """
        # Detect conflicts on remaining path
        conflicts = self.detector.check_path(
            self._current_path,
            obstacle_mask=self._obstacle_mask,
            nfz_mask=self._nfz_mask,
            risk_map=self._risk_map,
            vehicle_positions=self._vehicle_positions,
            start_index=path_index,
        )

        should, trigger, reason = self.policy.evaluate(step, conflicts, risk_at_pos)
        return should, trigger, reason, conflicts

    def try_replan(
        self,
        current_pos: GridPos,
        step: int,
        trigger: ReplanTrigger,
        reason: str,
        conflicts: list[Conflict],
        cost_map: np.ndarray | None = None,
        parent_event_id: str | None = None,
    ) -> PlanResult:
        """Execute a replan from current position.

        Merges all obstacles, re-plans, logs the event, publishes
        a REPLAN event on the bus.
        """
        goal = self._current_goal
        if goal is None:
            return PlanResult(path=[], success=False, compute_time_ms=0, reason="no_goal")

        old_path_len = len(self._current_path)
        old_cost = self._path_cost

        # Merge obstacles
        merged = self.planner.no_fly | self._obstacle_mask | self._nfz_mask
        original_nfz = self.planner.no_fly
        self.planner.no_fly = merged

        t0 = time.perf_counter()
        result = self.planner.plan(current_pos, goal, cost_map)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.planner.no_fly = original_nfz

        new_path_len = len(result.path) if result.success else 0
        new_cost = float(new_path_len)

        if result.success:
            self._current_path = list(result.path)
            self._path_cost = new_cost

        # Log
        record = ReplanRecord(
            replan_id=uuid.uuid4().hex[:12],
            step=step,
            trigger=trigger,
            reason=reason,
            position=current_pos,
            old_path_length=old_path_len,
            new_path_length=new_path_len,
            old_path_cost=old_cost,
            new_path_cost=new_cost,
            compute_time_ms=elapsed_ms,
            expansions=result.expansions,
            success=result.success,
            conflicts=conflicts,
            parent_event_id=parent_event_id,
        )
        self._replan_log.append(record)
        self.policy.record_replan()

        # Publish replan event on bus
        self.bus.publish(UpdateEvent(
            event_type=EventType.REPLAN,
            step=step,
            description=f"replan:{trigger.value}:{reason}",
            severity=0.7 if trigger == ReplanTrigger.EVENT else 0.3,
            position=current_pos,
            payload=record.to_dict(),
            parent_id=parent_event_id,
        ))

        return PlanResult(
            path=result.path,
            success=result.success,
            compute_time_ms=elapsed_ms,
            expansions=result.expansions,
            replans=len(self._replan_log) - 1,  # exclude initial
            reason=result.reason,
        )

    # ── Query ─────────────────────────────────────────────────────

    @property
    def current_path(self) -> list[GridPos]:
        return list(self._current_path)

    @property
    def replan_count(self) -> int:
        """Number of replans (excluding initial plan)."""
        return max(0, len(self._replan_log) - 1)

    @property
    def replan_log(self) -> list[ReplanRecord]:
        return list(self._replan_log)

    def replan_summary(self) -> dict[str, Any]:
        """Summary statistics for the replan log."""
        records = self._replan_log[1:]  # skip initial
        if not records:
            return {"replan_count": 0, "triggers": {}, "total_time_ms": 0.0}
        triggers: dict[str, int] = {}
        for r in records:
            triggers[r.trigger.value] = triggers.get(r.trigger.value, 0) + 1
        return {
            "replan_count": len(records),
            "triggers": triggers,
            "total_time_ms": sum(r.compute_time_ms for r in records),
            "total_expansions": sum(r.expansions for r in records),
            "success_rate": sum(1 for r in records if r.success) / len(records),
        }
