"""Mission schemas (MC-1, MC-3).

TaskSpec, MissionBriefing, and mission metadata for the mission engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uavbench2.envs.base import TaskStatus


@dataclass
class TaskSpec:
    """A single mission task (MC-1, MC-2)."""

    task_id: str
    xy: tuple[int, int]
    weight: float = 1.0
    time_decay: float = 0.02
    service_time: int = 0  # consecutive STAY steps at xy to complete
    category: str = "waypoint"
    injected_at: int = 0
    status: TaskStatus = TaskStatus.PENDING
    stay_counter: int = 0  # how many consecutive STAYs at xy


@dataclass
class MissionBriefing:
    """Structured mission briefing logged at episode start (MC-3)."""

    mission_type: str
    domain: str
    origin_name: str
    destination_name: str
    objective: str
    deliverable: str
    constraints: list[str] = field(default_factory=list)
    battery_capacity_wh: float = 150.0
    service_time_steps: int = 0
    priority: str = "normal"
    max_time_steps: int = 2000

    def to_event(self) -> dict:
        """Convert to event dict for logging."""
        return {
            "type": "mission_briefing",
            "step_idx": 0,
            "mission_type": self.mission_type,
            "domain": self.domain,
            "origin_name": self.origin_name,
            "destination_name": self.destination_name,
            "objective": self.objective,
            "deliverable": self.deliverable,
            "constraints": list(self.constraints),
            "battery_capacity_wh": self.battery_capacity_wh,
            "service_time_steps": self.service_time_steps,
            "priority": self.priority,
            "max_time_steps": self.max_time_steps,
        }
