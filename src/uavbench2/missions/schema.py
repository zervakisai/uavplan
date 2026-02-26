"""Mission schemas (MC-1).

TaskSpec and mission metadata for the mission engine.
"""

from __future__ import annotations

from dataclasses import dataclass
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
