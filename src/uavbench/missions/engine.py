"""Mission engine (MC-2).

Tracks task queue, completion via service_time, and generates events.
"""

from __future__ import annotations

from uavbench.envs.base import TaskStatus
from uavbench.missions.schema import MissionBriefing, TaskSpec
from uavbench.scenarios.schema import MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Mission metadata per mission type (MC-1, MC-3)
# ---------------------------------------------------------------------------

_MISSION_META: dict[str, dict] = {
    "fire_delivery": {
        "objective_label": "Emergency Medical Supply Delivery",
        "objective_reason": (
            "Emergency medical supply delivery to fire-isolated settlement"
        ),
        "deliverable_name": "medical_supplies",
        "default_service_time": 0,
        "task_category": "delivery_point",
        "origin_name": "Hospital Depot Alpha",
        "destination_name": "Fire-Isolated Settlement",
        "briefing_objective": (
            "Deliver emergency medical supplies to cut-off settlement"
        ),
        "constraints": ["Avoid active fire zones", "Respect firefighting NFZs"],
        "priority": "critical",
    },
    "flood_rescue": {
        "objective_label": "Flood Search & Rescue Assessment",
        "objective_reason": (
            "Search and rescue assessment of flood-stranded population"
        ),
        "deliverable_name": "rescue_assessment",
        "default_service_time": 2,
        "task_category": "rescue_site",
        "origin_name": "Emergency Operations Center",
        "destination_name": "Flood-Stranded Area",
        "briefing_objective": (
            "Locate and assess flood-stranded population for rescue coordination"
        ),
        "constraints": ["Avoid flooded road corridors", "Maintain safe altitude over water"],
        "priority": "critical",
    },
    "fire_surveillance": {
        "objective_label": "Aerial Fire Perimeter Survey",
        "objective_reason": (
            "Aerial survey of active fire perimeter for command post"
        ),
        "deliverable_name": "perimeter_report",
        "default_service_time": 3,
        "task_category": "survey_point",
        "origin_name": "Fire Command Post",
        "destination_name": "Active Fire Perimeter",
        "briefing_objective": (
            "Survey active fire perimeter and report coverage to command post"
        ),
        "constraints": ["Avoid manned aircraft corridors", "Stay clear of active fire front"],
        "priority": "high",
    },
}


class MissionEngine:
    """Manages task lifecycle and completion events.

    Enforces MC-2: task completion = reaching POI + spending service_time_s
    consecutive STAY steps.
    """

    def __init__(
        self,
        mission_type: MissionType,
        start_xy: tuple[int, int],
        goal_xy: tuple[int, int],
        config: ScenarioConfig | None = None,
        rng: "numpy.random.Generator | None" = None,
    ) -> None:
        self.mission_type = mission_type
        self.start_xy = start_xy
        self.goal_xy = goal_xy
        self._config = config
        self._meta = _MISSION_META[mission_type.value]
        self._events: list[dict] = []
        self._tasks: list[TaskSpec] = []
        self._completed_count = 0

        service_time = int(self._meta["default_service_time"])

        if service_time > 0:
            # Place task at midpoint between start and goal (not at goal)
            # so the agent can complete service_time before episode terminates
            mx = (start_xy[0] + goal_xy[0]) // 2
            my = (start_xy[1] + goal_xy[1]) // 2
            task_xy = (mx, my)
        else:
            # Fly-through: task at the goal
            task_xy = goal_xy

        self._tasks.append(
            TaskSpec(
                task_id="task_0",
                xy=task_xy,
                service_time=service_time,
                category=str(self._meta["task_category"]),
                status=TaskStatus.ACTIVE,
            )
        )

    @property
    def events(self) -> list[dict]:
        return self._events

    @property
    def objective_poi(self) -> tuple[int, int]:
        """Current task location (MC-1)."""
        for t in self._tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.ACTIVE):
                return t.xy
        return self.goal_xy

    @property
    def objective_reason(self) -> str:
        return str(self._meta["objective_reason"])

    @property
    def objective_label(self) -> str:
        return str(self._meta["objective_label"])

    @property
    def deliverable_name(self) -> str:
        return str(self._meta["deliverable_name"])

    @property
    def service_time_s(self) -> int:
        """Service time for the current active task."""
        for t in self._tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.ACTIVE):
                return t.service_time
        return 0

    @property
    def task_progress(self) -> str:
        """e.g. '1/4' (MC-3)."""
        total = len(self._tasks)
        done = sum(1 for t in self._tasks if t.status == TaskStatus.COMPLETED)
        return f"{done}/{total}"

    @property
    def all_tasks_completed(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self._tasks)

    def distance_to_task(self, agent_xy: tuple[int, int]) -> float:
        """Manhattan distance to current objective POI."""
        poi = self.objective_poi
        return float(abs(agent_xy[0] - poi[0]) + abs(agent_xy[1] - poi[1]))

    def generate_briefing(self) -> MissionBriefing:
        """Generate a MissionBriefing for this episode (MC-3)."""
        max_steps = 2000
        if self._config is not None:
            max_steps = self._config.effective_max_steps

        return MissionBriefing(
            mission_type=self.mission_type.value,
            domain=self.mission_type.value,
            origin_name=str(self._meta.get("origin_name", "Base")),
            destination_name=str(self._meta.get("destination_name", "Objective")),
            objective=str(self._meta.get("briefing_objective", self._meta["objective_reason"])),
            deliverable=str(self._meta["deliverable_name"]),
            constraints=list(self._meta.get("constraints", [])),
            service_time_steps=int(self._meta["default_service_time"]),
            priority=str(self._meta.get("priority", "normal")),
            max_time_steps=max_steps,
        )

    def step(
        self,
        agent_xy: tuple[int, int],
        action: int,
        step_idx: int,
    ) -> None:
        """Update mission state after an env step.

        Checks if agent is at a task POI and tracks STAY progress.
        Fires task_completed event when service_time is met (MC-2).
        """
        stay_action = 4  # STAY

        for task in self._tasks:
            if task.status not in (TaskStatus.PENDING, TaskStatus.ACTIVE):
                continue

            if tuple(agent_xy) == tuple(task.xy):
                task.status = TaskStatus.ACTIVE
                if action == stay_action:
                    task.stay_counter += 1
                else:
                    # Non-STAY action at POI: for fly-through (service_time=0)
                    # completion is checked below
                    pass

                # Check completion: service_time=0 means fly-through
                if task.service_time == 0 or task.stay_counter >= task.service_time:
                    task.status = TaskStatus.COMPLETED
                    self._completed_count += 1
                    self._events.append({
                        "type": "task_completed",
                        "task_id": task.task_id,
                        "step_idx": step_idx,
                        "xy": task.xy,
                    })
            else:
                # Agent left POI — reset stay counter
                task.stay_counter = 0
