"""Mission engine (MC-2).

Tracks task queue, completion via service_time, and generates events.
"""

from __future__ import annotations

import numpy as np

from uavbench.envs.base import TaskStatus
from uavbench.missions.schema import MissionBriefing, TaskSpec
from uavbench.scenarios.schema import MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Mission metadata per mission type (MC-1, MC-3)
# ---------------------------------------------------------------------------

_MISSION_META: dict[str, dict] = {
    "pharma_delivery": {
        "objective_label": "Emergency Pharmaceutical Delivery",
        "objective_reason": (
            "Pharmaceutical delivery to fire-isolated settlement"
        ),
        "deliverable_name": "pharmaceuticals",
        "default_service_time": 1,
        "task_category": "pharmacy_pickup",
        "origin_name": "Hospital Depot Alpha",
        "destination_name": "Fire-Isolated Settlement",
        "briefing_objective": (
            "Collect pharmaceuticals from pharmacy, deliver to fire-isolated settlement"
        ),
        "constraints": ["Avoid active fire zones", "Respect firefighting NFZs"],
        "priority": "critical",
    },
    "urban_rescue": {
        "objective_label": "Urban Search & Rescue Assessment",
        "objective_reason": (
            "Search and rescue assessment of stranded casualties"
        ),
        "deliverable_name": "rescue_assessment",
        "default_service_time": 2,
        "task_category": "rescue_site",
        "origin_name": "Emergency Operations Center",
        "destination_name": "Disaster-Stranded Area",
        "briefing_objective": (
            "Locate and assess stranded casualties for rescue coordination"
        ),
        "constraints": ["Avoid active fire zones", "Maintain safe altitude over debris"],
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
        # Default midpoint for initial placement (snap_poi_to_path
        # will distribute along corridor later for multi-POI missions).
        mx = (start_xy[0] + goal_xy[0]) // 2
        my = (start_xy[1] + goal_xy[1]) // 2

        if mission_type.value == "urban_rescue":
            # Multi-POI: 3 casualties with decreasing severity
            weights = [3.0, 2.0, 1.0]  # CRITICAL, SERIOUS, MINOR
            for i, w in enumerate(weights):
                self._tasks.append(
                    TaskSpec(
                        task_id=f"task_{i}",
                        xy=(mx, my),
                        service_time=service_time,
                        weight=w,
                        category=str(self._meta["task_category"]),
                        status=TaskStatus.ACTIVE,
                    )
                )
        elif mission_type.value == "fire_surveillance":
            # Multi-POI: 3 survey points along corridor
            for i in range(3):
                self._tasks.append(
                    TaskSpec(
                        task_id=f"task_{i}",
                        xy=(mx, my),
                        service_time=service_time,
                        weight=1.0,
                        category=str(self._meta["task_category"]),
                        status=TaskStatus.ACTIVE,
                    )
                )
        else:
            # pharma_delivery: 2-POI mission
            #   task_0 = pharmacy pickup (collect meds)
            #   task_1 = delivery point (hand off to settlement)
            self._tasks.append(
                TaskSpec(
                    task_id="task_0",
                    xy=(mx, my),
                    service_time=service_time,
                    weight=1.0,
                    category="pharmacy_pickup",
                    status=TaskStatus.ACTIVE,
                )
            )
            self._tasks.append(
                TaskSpec(
                    task_id="task_1",
                    xy=(mx, my),  # temporary — scatter_pois will reposition
                    service_time=max(1, service_time),
                    weight=1.0,
                    category="delivery_point",
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
    def all_task_positions(self) -> list[tuple[int, int]]:
        """All task positions (for cell clearing after snap)."""
        return [t.xy for t in self._tasks]

    @property
    def task_info_list(self) -> list[dict]:
        """Task positions, categories, status, and service_time for rendering."""
        return [
            {
                "xy": t.xy,
                "category": t.category,
                "status": t.status.value,
                "task_id": t.task_id,
                "weight": t.weight,
                "service_time": t.service_time,
            }
            for t in self._tasks
        ]

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
    def origin_name(self) -> str:
        return str(self._meta.get("origin_name", "Base"))

    @property
    def destination_name(self) -> str:
        return str(self._meta.get("destination_name", "Objective"))

    @property
    def priority(self) -> str:
        return str(self._meta.get("priority", "normal"))

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

    def snap_poi_to_path(self, path: list[tuple[int, int]]) -> None:
        """Legacy: snap task POIs to corridor positions.

        Kept for backward compat; prefer scatter_pois() for realistic placement.
        """
        if not path or not self._tasks:
            return
        active = [t for t in self._tasks if t.service_time > 0]
        n = len(active)
        for i, task in enumerate(active):
            if n == 1:
                frac_idx = len(path) // 2
            else:
                frac = (i + 1) / (n + 2)
                frac_idx = int(len(path) * frac)
            frac_idx = min(frac_idx, len(path) - 1)
            task.xy = path[frac_idx]

    def scatter_pois(
        self,
        corridor: list[tuple[int, int]],
        heightmap: np.ndarray,
        rng: np.random.Generator,
        min_corridor_dist: int = 10,
        search_radius: int = 50,
    ) -> None:
        """Place task POIs at realistic off-corridor positions.

        For each task, finds a walkable cell near the corresponding
        corridor fraction but offset perpendicular to the corridor by
        at least *min_corridor_dist* cells.  Falls back to corridor
        position if no suitable cell exists.

        Deterministic: same (corridor, heightmap, rng state) → same POIs.
        """
        from scipy.ndimage import binary_dilation

        if not corridor or not self._tasks:
            return

        H, W = heightmap.shape
        walkable = heightmap == 0.0

        # Build corridor proximity mask (cells within min_corridor_dist)
        corridor_mask = np.zeros((H, W), dtype=bool)
        for cx, cy in corridor:
            if 0 <= cy < H and 0 <= cx < W:
                corridor_mask[cy, cx] = True
        near_corridor = binary_dilation(
            corridor_mask, iterations=min_corridor_dist,
        )
        # Valid candidates: walkable AND far from corridor
        valid = walkable & ~near_corridor

        n = len(self._tasks)
        for i, task in enumerate(self._tasks):
            # Corridor fraction: front-loaded (same as snap)
            frac = (i + 1) / (n + 2)
            frac_idx = min(int(len(corridor) * frac), len(corridor) - 1)
            base_x, base_y = corridor[frac_idx]

            # Collect valid candidates within search_radius of base
            candidates: list[tuple[int, int]] = []
            for dy in range(-search_radius, search_radius + 1):
                ny = base_y + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in range(-search_radius, search_radius + 1):
                    nx = base_x + dx
                    if nx < 0 or nx >= W:
                        continue
                    if valid[ny, nx]:
                        candidates.append((nx, ny))

            if candidates:
                idx = int(rng.integers(len(candidates)))
                task.xy = candidates[idx]
            else:
                # Fallback: corridor position
                task.xy = (base_x, base_y)

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
                        "weight": task.weight,
                    })
            else:
                # Agent left POI — reset stay counter
                task.stay_counter = 0
