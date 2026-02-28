"""Mission engine: drives a multi-task episode with dynamic injection.

Responsibilities:
  - Maintain task queue (initial + injected)
  - Track partial observability (comms dropouts, latency)
  - Accumulate operational products
  - Compute normalised mission utility and common metrics
  - Provide hooks for human-in-the-loop re-tasking

The engine is MAP-AGNOSTIC: it works on grid coordinates and delegates
path planning to the route layer.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from uavbench.missions.spec import (
    MissionSpec,
    MissionBriefing,
    BRIEFING_TEMPLATES,
    TaskSpec,
    TaskStatus,
    MissionProduct,
    ProductType,
    COMMON_METRICS,
)
from uavbench.planners.base import GridPos, PlanResult


# ── Runtime task state ────────────────────────────────────────────────

@dataclass
class RuntimeTask:
    """Mutable wrapper around a TaskSpec for runtime tracking."""
    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    arrival_step: int | None = None
    service_remaining: int = 0

    def __post_init__(self) -> None:
        self.service_remaining = self.spec.service_time

    @property
    def task_id(self) -> str:
        return self.spec.task_id

    @property
    def xy(self) -> GridPos:
        return self.spec.xy

    def is_available(self, step: int) -> bool:
        if self.status != TaskStatus.PENDING:
            return False
        tw = self.spec.time_window
        if tw is not None:
            return tw[0] <= step <= tw[1]
        return True

    def is_expired(self, step: int) -> bool:
        tw = self.spec.time_window
        if tw is not None and step > tw[1] and self.status == TaskStatus.PENDING:
            return True
        return False


# ── Injection schedule ────────────────────────────────────────────────

@dataclass
class InjectionEvent:
    """A task or constraint injected at a given step."""
    step: int
    task: TaskSpec | None = None
    nfz_polygon: list[GridPos] | None = None   # new restricted polygon
    risk_delta: np.ndarray | None = None       # additive risk-field update
    corridor_blocked: str | None = None        # corridor segment id to block
    description: str = ""


# ── Engine ────────────────────────────────────────────────────────────

class MissionEngine:
    """Stateful engine for a single mission episode.

    Usage:
        engine = MissionEngine(spec, rng)
        engine.set_injection_schedule(schedule)  # from mission builder
        while not engine.done:
            # policy picks next task
            # route-layer plans path
            # engine.step(pos, dyn_state, ...)
    """

    def __init__(
        self,
        spec: MissionSpec,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.spec = spec
        self.knobs = spec.knobs
        self.rng = rng or np.random.default_rng(42)

        # Task tracking
        self.tasks: list[RuntimeTask] = [
            RuntimeTask(spec=t) for t in spec.initial_tasks
        ]
        self._injection_schedule: list[InjectionEvent] = []
        self._pending_injections: list[InjectionEvent] = []

        # Products
        self.products: list[MissionProduct] = []

        # Observability
        self._risk_map_observed: np.ndarray | None = None  # delayed view
        self._pending_updates: list[tuple[int, dict[str, Any]]] = []

        # Metrics accumulators
        self.step_count: int = 0
        self.total_risk: float = 0.0
        self.total_energy: float = 0.0
        self.violation_count: int = 0
        self.replan_count: int = 0
        self.replan_time_ms: float = 0.0
        self.replan_expansions: int = 0
        self.event_detections: list[dict[str, Any]] = []
        self._segment_history: list[dict[str, Any]] = []
        self.done: bool = False

        # Active navigation state
        self.current_target: RuntimeTask | None = None
        self.current_path: list[GridPos] = []
        self.current_path_idx: int = 0

    # ── Setup ─────────────────────────────────────────────────────

    def set_injection_schedule(self, schedule: list[InjectionEvent]) -> None:
        """Set the injection schedule (built by mission-specific builder)."""
        self._injection_schedule = sorted(schedule, key=lambda e: e.step)
        self._pending_injections = list(self._injection_schedule)

    # ── Step ──────────────────────────────────────────────────────

    def step(
        self,
        agent_pos: GridPos,
        dyn_state: dict[str, Any],
        risk_at_pos: float = 0.0,
        energy_cost: float = 1.0,
    ) -> dict[str, Any]:
        """Advance one step.  Called after the route-layer moves the agent.

        Returns dict of events that occurred this step (injections, completions,
        expirations, violations, product updates).
        """
        self.step_count += 1
        step = self.step_count
        events: dict[str, Any] = {"step": step, "injections": [], "completions": [], "expirations": []}

        # 1. Process injections due this step
        while self._pending_injections and self._pending_injections[0].step <= step:
            inj = self._pending_injections.pop(0)
            if inj.task is not None:
                rt = RuntimeTask(spec=inj.task)
                self.tasks.append(rt)
                events["injections"].append(inj.task.task_id)
            events.setdefault("constraint_updates", []).append(inj.description)

        # 2. Comms: apply latency / dropout
        if self.knobs.comms_dropout_prob > 0 and self.rng.random() < self.knobs.comms_dropout_prob:
            events["comms_dropout"] = True
            # delayed risk map stays stale — skip update
        else:
            events["comms_dropout"] = False
            # Could update observed risk map here if engine manages it

        # 3. Check task arrivals
        if self.current_target is not None:
            t = self.current_target
            if agent_pos == t.xy and t.status in (TaskStatus.PENDING, TaskStatus.ACTIVE):
                if t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.ACTIVE
                    t.arrival_step = step
                t.service_remaining -= 1
                if t.service_remaining <= 0:
                    t.status = TaskStatus.COMPLETED
                    events["completions"].append(t.task_id)
                    self.current_target = None

        # 4. Expire overdue tasks
        for t in self.tasks:
            if t.is_expired(step):
                t.status = TaskStatus.EXPIRED
                events["expirations"].append(t.task_id)

        # 5. Accumulate risk + energy
        self.total_risk += risk_at_pos
        self.total_energy += energy_cost

        # 6. Check violations
        if dyn_state.get("in_nfz", False):
            self.violation_count += 1
            events["violation"] = True

        # 7. Check budget exhaustion
        if step >= self.knobs.time_budget:
            self.done = True
            events["budget_exhausted"] = "time"
        if self.total_energy >= self.knobs.energy_budget * self.knobs.time_budget:
            self.done = True
            events["budget_exhausted"] = "energy"

        # 8. Check all-complete
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        if not pending and self.current_target is None:
            self.done = True
            events["all_tasks_complete"] = True

        return events

    # ── Replan accounting ─────────────────────────────────────────

    def record_replan(self, result: PlanResult) -> None:
        """Record a replanning event for metrics."""
        self.replan_count += 1
        self.replan_time_ms += result.compute_time_ms
        self.replan_expansions += result.expansions

    def record_segment(self, segment: dict[str, Any]) -> None:
        """Record a completed path segment for logging."""
        self._segment_history.append(segment)

    # ── Product generation ────────────────────────────────────────

    def add_product(self, product: MissionProduct) -> None:
        self.products.append(product)

    def record_event_detection(
        self,
        event_id: str,
        detected_step: int,
        response_start_step: int | None = None,
        localized_step: int | None = None,
    ) -> None:
        self.event_detections.append({
            "event_id": event_id,
            "detected_step": detected_step,
            "response_start_step": response_start_step,
            "localized_step": localized_step,
        })

    # ── Utility computation ───────────────────────────────────────

    def compute_task_utility(self) -> float:
        """U = Σ w_i · exp(-λ · t_arrival_i) for completed tasks, normalised to [0,1]."""
        max_possible = sum(t.spec.weight for t in self.tasks)
        if max_possible == 0:
            return 0.0
        achieved = 0.0
        for t in self.tasks:
            if t.status == TaskStatus.COMPLETED and t.arrival_step is not None:
                delay = t.arrival_step - t.spec.injected_at
                achieved += t.spec.weight * math.exp(-t.spec.time_decay * delay)
        return min(achieved / max_possible, 1.0)

    def compute_mission_score(self) -> float:
        """Normalised mission score ∈ [0,1] with risk + violation penalties."""
        u = self.compute_task_utility()
        risk_penalty = self.spec.risk_penalty_lambda * self.total_risk / max(self.step_count, 1)
        violation_penalty = self.spec.violation_penalty_lambda * self.violation_count / max(self.step_count, 1)
        return float(max(0.0, min(1.0, u - risk_penalty - violation_penalty)))

    def compute_product_latency(self) -> float:
        """Mean time from event injection → first valid product update (steps)."""
        if not self.event_detections:
            return 0.0
        latencies = []
        for det in self.event_detections:
            resp = det.get("response_start_step") or det.get("localized_step")
            if resp is not None:
                latencies.append(resp - det["detected_step"])
        return float(np.mean(latencies)) if latencies else 0.0

    # ── Common metrics ────────────────────────────────────────────

    def compute_common_metrics(self) -> dict[str, float]:
        """Compute all COMMON_METRICS for this episode."""
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        total_tasks = len(self.tasks)
        return {
            "mission_score": self.compute_mission_score(),
            "task_completion_rate": completed / max(total_tasks, 1),
            "completion_time": float(self.step_count),
            "energy_used": self.total_energy,
            "risk_integral": self.total_risk,
            "violation_count": float(self.violation_count),
            "replanning_cost_ms": self.replan_time_ms,
            "replanning_count": float(self.replan_count),
            "replanning_expansions": float(self.replan_expansions),
            "robustness": completed / max(total_tasks, 1),  # base; sweep in runner
            "product_latency": self.compute_product_latency(),
        }

    # ── Task query helpers ────────────────────────────────────────

    def pending_tasks(self) -> list[RuntimeTask]:
        """Return tasks available now OR whose window hasn't opened yet (reachable in future)."""
        available = [t for t in self.tasks if t.status == TaskStatus.PENDING and t.is_available(self.step_count)]
        if available:
            return available
        # Fallback: include upcoming tasks whose window hasn't started yet
        upcoming = [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and t.spec.time_window is not None
            and t.spec.time_window[0] > self.step_count
            and t.spec.time_window[1] > self.step_count
        ]
        return upcoming

    def completed_tasks(self) -> list[RuntimeTask]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    # ── Product export ────────────────────────────────────────────

    def export_products(self) -> dict[str, list[dict[str, Any]]]:
        """Group products by type for CSV/GeoJSON export."""
        out: dict[str, list[dict[str, Any]]] = {}
        for p in self.products:
            key = p.product_type.value
            out.setdefault(key, []).append({"step": p.timestamp_step, **p.data})
        return out

    def export_episode_log(self) -> dict[str, Any]:
        """Full episode log (JSON-serialisable)."""
        return {
            "mission_id": self.spec.mission_id.value,
            "difficulty": self.spec.difficulty,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "xy": list(t.xy),
                    "status": t.status.value,
                    "arrival_step": t.arrival_step,
                    "category": t.spec.category,
                    "injected_at": t.spec.injected_at,
                }
                for t in self.tasks
            ],
            "metrics": self.compute_common_metrics(),
            "segments": self._segment_history,
            "products": {k: v for k, v in self.export_products().items()},
            "event_detections": self.event_detections,
            "step_count": self.step_count,
        }


# ── Briefing generation ──────────────────────────────────────────────


def generate_briefing(
    scenario_config: Any,
    mission_spec: MissionSpec | None = None,
) -> MissionBriefing:
    """Generate a human-readable mission briefing from scenario config.

    Uses BRIEFING_TEMPLATES for the mission family and enriches with
    scenario-level metadata (difficulty, incident provenance, map tile).

    Args:
        scenario_config: ScenarioConfig instance (or any object with
            ``mission_type``, ``difficulty``, ``description``, etc.)
        mission_spec: Optional MissionSpec for service_time and time_budget.

    Returns:
        Frozen MissionBriefing dataclass.
    """
    # Resolve mission_type string from enum or plain str
    mt_raw = getattr(scenario_config, "mission_type", "point_to_point")
    mt = getattr(mt_raw, "value", str(mt_raw))

    template = BRIEFING_TEMPLATES.get(mt, {})

    # Difficulty → priority mapping
    diff_raw = getattr(scenario_config, "difficulty", "easy")
    diff = getattr(diff_raw, "value", str(diff_raw))
    priority_map = {"hard": "critical", "medium": "high", "easy": "routine"}
    priority = priority_map.get(diff, "routine")

    # Domain
    domain_raw = getattr(scenario_config, "domain", "urban")
    domain = getattr(domain_raw, "value", str(domain_raw))

    # Build constraints from active dynamic layers
    constraints: list[str] = []
    if getattr(scenario_config, "enable_fire", False):
        constraints.append("Avoid active fire zones")
    if getattr(scenario_config, "enable_dynamic_nfz", False):
        constraints.append("Respect dynamic no-fly restrictions")
    if getattr(scenario_config, "enable_traffic", False):
        constraints.append("Maintain safe distance from emergency vehicles")
    if getattr(scenario_config, "fire_blocks_movement", False):
        constraints.append("Burning cells are impassable")
    if getattr(scenario_config, "traffic_blocks_movement", False):
        constraints.append("Vehicle buffer zones reject entry")

    # Enrich origin/destination with tile name if available
    tile = getattr(scenario_config, "osm_tile_id", None) or ""
    origin = template.get("origin_name", "Operations Base")
    destination = template.get("destination_name", "Mission Target")
    if tile and tile.lower() not in origin.lower():
        origin = f"{origin} ({tile.title()})"

    # Max time steps from mission spec or scenario extra
    max_steps = 0
    service_time = 0
    if mission_spec is not None:
        max_steps = mission_spec.knobs.time_budget
        if mission_spec.initial_tasks:
            service_time = mission_spec.initial_tasks[0].service_time
    else:
        extra = getattr(scenario_config, "extra", None) or {}
        max_steps = int(extra.get("time_budget", 4 * getattr(scenario_config, "map_size", 100)))

    return MissionBriefing(
        mission_type=mt,
        domain=domain,
        origin_name=origin,
        destination_name=destination,
        objective=template.get("objective", "Navigate to destination"),
        deliverable=template.get("deliverable", "Navigation completion"),
        constraints=tuple(constraints),
        service_time_steps=service_time,
        priority=priority,
        max_time_steps=max_steps,
    )
