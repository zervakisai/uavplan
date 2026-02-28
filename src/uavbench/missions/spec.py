"""Mission specification: data model for tasks, products, difficulty knobs.

All data classes are frozen for immutability + hashability.  Runtime state
is tracked separately in MissionEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ── Mission identifiers ──────────────────────────────────────────────

class MissionID(str, Enum):
    """The three government-ready missions."""
    CIVIL_PROTECTION = "civil_protection"
    MARITIME_DOMAIN = "maritime_domain"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    SKIPPED = "skipped"


class ProductType(str, Enum):
    """Operational product types (deliverables)."""
    # Mission 1
    FIRE_PERIMETER_GEOJSON = "fire_perimeter.geojson"
    CORRIDOR_STATUS_CSV = "corridor_status.csv"
    ALERT_TIMELINE_CSV = "alert_timeline.csv"
    # Mission 2
    CORRIDOR_COVERAGE_CSV = "corridor_coverage.csv"
    EVENT_RESPONSE_CSV = "event_response.csv"
    EXPOSURE_REPORT_CSV = "exposure_report.csv"
    # Mission 3
    INSPECTION_LOG_CSV = "inspection_log.csv"
    COMPLIANCE_REPORT_CSV = "compliance_report.csv"
    RESILIENCE_CURVE_CSV = "resilience_curve.csv"


# ── Task specification ────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskSpec:
    """Single task (POI / waypoint / inspection site).

    Attributes:
        task_id: unique within the mission episode.
        xy: grid position (x, y).
        weight: importance weight w_i for utility calculation.
        time_decay: exponential decay rate λ for time-decaying utility.
        time_window: optional (earliest, latest) step window.
        service_time: steps required at site (0 = fly-through).
        category: semantic label (perimeter_point, corridor_checkpoint, etc.)
        injected_at: step at which this task was injected (0 = initial).
    """
    task_id: str
    xy: tuple[int, int]
    weight: float = 1.0
    time_decay: float = 0.02
    time_window: tuple[int, int] | None = None   # (earliest, latest) step
    service_time: int = 0
    category: str = "generic"
    injected_at: int = 0


# ── Difficulty knobs ──────────────────────────────────────────────────

@dataclass(frozen=True)
class DifficultyKnobs:
    """Parameterised difficulty (easy / med / hard) — independent of map.

    All three difficulty levels share the SAME map; only these knobs change.
    """
    num_tasks: int                              # (a) 4 / 6 / 8
    injection_rate: Literal["low", "medium", "high"]  # (b) new-task cadence
    dynamics_intensity: Literal["static", "moderate", "severe"]  # (c)
    time_budget: int                            # max episode steps
    energy_budget: float                        # energy proxy (1.0 = full)
    comms_dropout_prob: float = 0.0             # P(update dropped per step)
    comms_latency_steps: int = 0                # delay on risk-map / task updates
    risk_update_cadence: int = 5                # steps between risk-field refreshes

    @classmethod
    def easy(cls, num_tasks: int = 4, time_budget: int = 300) -> DifficultyKnobs:
        return cls(
            num_tasks=num_tasks,
            injection_rate="low",
            dynamics_intensity="static",
            time_budget=time_budget,
            energy_budget=1.0,
            comms_dropout_prob=0.0,
            comms_latency_steps=0,
            risk_update_cadence=8,
        )

    @classmethod
    def medium(cls, num_tasks: int = 6, time_budget: int = 250) -> DifficultyKnobs:
        return cls(
            num_tasks=num_tasks,
            injection_rate="medium",
            dynamics_intensity="moderate",
            time_budget=time_budget,
            energy_budget=0.85,
            comms_dropout_prob=0.05,
            comms_latency_steps=2,
            risk_update_cadence=5,
        )

    @classmethod
    def hard(cls, num_tasks: int = 8, time_budget: int = 200) -> DifficultyKnobs:
        return cls(
            num_tasks=num_tasks,
            injection_rate="high",
            dynamics_intensity="severe",
            time_budget=time_budget,
            energy_budget=0.70,
            comms_dropout_prob=0.15,
            comms_latency_steps=4,
            risk_update_cadence=3,
        )


# ── Operational product ──────────────────────────────────────────────

@dataclass
class MissionProduct:
    """One row / record in an operational product.

    Products are accumulated by the engine and exported at episode end.
    """
    product_type: ProductType
    timestamp_step: int
    data: dict[str, Any] = field(default_factory=dict)


# ── Mission specification ─────────────────────────────────────────────

@dataclass(frozen=True)
class MissionSpec:
    """Full specification of one mission (immutable template).

    Created once per mission × difficulty combination and stored in the
    scenario registry.  The MissionEngine uses this to drive the episode.
    """
    mission_id: MissionID
    label: str                     # human-readable
    difficulty: Literal["easy", "medium", "hard"]
    knobs: DifficultyKnobs
    initial_tasks: tuple[TaskSpec, ...]
    product_types: tuple[ProductType, ...]
    utility_decay: float = 0.02    # default λ for U = Σ w·exp(-λ·t)
    risk_penalty_lambda: float = 0.3   # λ_risk
    violation_penalty_lambda: float = 1.0  # λ_v
    patrol_weight_alpha: float = 0.5   # α for combined patrol + event score (Mission 2)
    strict_compliance: bool = False     # hard-fail on NFZ breach (Mission 3 option)
    description: str = ""


# ── Mission briefing ──────────────────────────────────────────────────

@dataclass(frozen=True)
class MissionBriefing:
    """Human-readable mission briefing generated at episode start.

    Logged as the first event (step_idx=0) of every episode so that
    results are self-documenting.  Enforces MC-1 (mission objective)
    and MC-4 (briefing in results).
    """
    mission_type: str           # "civil_protection", "maritime_domain", ...
    domain: str                 # "urban"
    origin_name: str            # e.g. "Penteli Fire Station"
    destination_name: str       # e.g. "Evacuation Zone Alpha"
    objective: str              # e.g. "Deliver medical supplies to evacuation zone"
    deliverable: str            # e.g. "Thermal-sealed medical kit"
    constraints: tuple[str, ...] = ()  # e.g. ("Avoid active fire zones",)
    service_time_steps: int = 0
    priority: str = "routine"   # "critical", "high", "routine"
    max_time_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialisation for event logging."""
        return {
            "mission_type": self.mission_type,
            "domain": self.domain,
            "origin_name": self.origin_name,
            "destination_name": self.destination_name,
            "objective": self.objective,
            "deliverable": self.deliverable,
            "constraints": list(self.constraints),
            "service_time_steps": self.service_time_steps,
            "priority": self.priority,
            "max_time_steps": self.max_time_steps,
        }


# ── Briefing templates per mission family ─────────────────────────────

BRIEFING_TEMPLATES: dict[str, dict[str, str]] = {
    "civil_protection": {
        "objective": "Emergency medical delivery during wildfire crisis",
        "deliverable": "Thermal-sealed medical kit",
        "origin_name": "Penteli Fire Station",
        "destination_name": "Evacuation Zone Alpha",
    },
    "critical_infrastructure": {
        "objective": "Critical infrastructure inspection under restricted airspace",
        "deliverable": "Structural integrity assessment report",
        "origin_name": "HCAA Operations Centre (Downtown Athens)",
        "destination_name": "Infrastructure Inspection Site",
    },
    "maritime_domain": {
        "objective": "Maritime search and rescue in multi-hazard zone",
        "deliverable": "Survivor location and status report",
        "origin_name": "Piraeus Coast Guard Station",
        "destination_name": "Maritime Distress Zone",
    },
}


# ── Common metric keys ────────────────────────────────────────────────

COMMON_METRICS = (
    "mission_score",
    "task_completion_rate",
    "completion_time",
    "energy_used",
    "risk_integral",
    "violation_count",
    "replanning_cost_ms",
    "replanning_count",
    "replanning_expansions",
    "robustness",
    "product_latency",
)
