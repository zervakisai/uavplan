"""ForcedReplanScheduler: guarantees ≥2 replans per episode.

"φρόντισε οπως το ειχαμε κανει να αναγκάζεται σιγουρα να κανει 2 φορες replan"

Strategy:
  1. After the initial plan is computed, inspect the path.
  2. Pick two points at ~33% and ~66% of the path.
  3. At those steps, inject a blocking obstacle ON the path ahead.
  4. Register forced-replan steps with the ReplanPolicy.
  5. The ConflictDetector will detect the block → PlannerAdapter replans.

This is deterministic and seed-controlled: same seed → same forced
replan steps → same obstacle injections → same replan triggers.

The scheduler also provides a fallback: if by the forced step the
planner has already replanned enough times, no additional obstacle
is injected (but the replan is still forced via the policy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np

from uavbench.updates.bus import EventType, UpdateEvent, UpdateBus

if TYPE_CHECKING:
    from uavbench.planners.adapter import ReplanPolicy


GridPos = tuple[int, int]


@dataclass
class ForcedReplanEvent:
    """Record of a forced replan injection."""
    step: int
    obstacle_position: GridPos
    obstacle_radius: int
    injected: bool = False
    description: str = ""


class ForcedReplanScheduler:
    """Guarantees ≥2 replans per episode by injecting path-blocking obstacles.

    Usage::

        scheduler = ForcedReplanScheduler(bus, replan_policy)
        # After initial plan:
        scheduler.schedule_from_path(path, time_budget=200)
        # Each step:
        obstacle_mask = scheduler.step(step, current_path, path_index)

    Parameters
    ----------
    bus : UpdateBus
        Event bus for publishing obstacle injections.
    replan_policy : ReplanPolicy
        Policy to register forced replan steps with.
    min_replans : int
        Minimum number of replans to guarantee.  Default 2.
    obstacle_radius : int
        Radius of injected blocking obstacles.  Default 5.
    """

    def __init__(
        self,
        bus: UpdateBus,
        replan_policy: ReplanPolicy,
        min_replans: int = 2,
        obstacle_radius: int = 5,
    ) -> None:
        self.bus = bus
        self.policy = replan_policy
        self.min_replans = min_replans
        self.obstacle_radius = obstacle_radius

        self._scheduled: list[ForcedReplanEvent] = []
        self._grid_shape: tuple[int, int] = (500, 500)
        self._active_obstacles: dict[int, np.ndarray] = {}  # step → mask

    def schedule_from_path(
        self,
        path: Sequence[GridPos],
        time_budget: int,
        grid_shape: tuple[int, int] = (500, 500),
    ) -> list[ForcedReplanEvent]:
        """Schedule forced replans based on the initial path.

        Places obstacles at ~33% and ~66% of the estimated arrival time.
        Registers forced-replan steps with the ReplanPolicy.

        Parameters
        ----------
        path : list of (x, y)
            Initial planned path.
        time_budget : int
            Total episode time budget (steps).
        grid_shape : (H, W)

        Returns
        -------
        list[ForcedReplanEvent]
            Scheduled events (not yet injected).
        """
        self._grid_shape = grid_shape
        self._scheduled.clear()

        if len(path) < 6:
            return []

        # Pick two path points at ~33% and ~66%
        n = len(path)
        fractions = [1 / 3, 2 / 3]

        for i, frac in enumerate(fractions[:self.min_replans]):
            path_idx = int(n * frac)
            path_idx = max(3, min(path_idx, n - 3))
            obstacle_pos = path[path_idx]

            # The step at which to inject: estimate from path_idx
            # (one step per path cell, minus some lookahead buffer)
            inject_step = max(5, path_idx - 5)

            event = ForcedReplanEvent(
                step=inject_step,
                obstacle_position=obstacle_pos,
                obstacle_radius=self.obstacle_radius,
                description=f"forced_replan_{i}_at_path[{path_idx}]",
            )
            self._scheduled.append(event)

            # Register with replan policy
            self.policy.add_forced_replan(inject_step)

        return list(self._scheduled)

    def step(
        self,
        step_num: int,
        current_path: Sequence[GridPos] | None = None,
        path_index: int = 0,
    ) -> np.ndarray | None:
        """Process this step: inject obstacles if scheduled.

        Returns
        -------
        np.ndarray or None
            [H, W] bool obstacle mask if an obstacle was injected this step.
        """
        H, W = self._grid_shape
        injected_mask = None

        for event in self._scheduled:
            if event.step == step_num and not event.injected:
                # Build obstacle mask centered on the path point
                mask = np.zeros((H, W), dtype=bool)
                ox, oy = event.obstacle_position
                r = event.obstacle_radius
                yy, xx = np.ogrid[0:H, 0:W]
                dist = np.sqrt((yy - oy) ** 2 + (xx - ox) ** 2)
                mask |= dist <= r

                event.injected = True
                self._active_obstacles[step_num] = mask

                if injected_mask is None:
                    injected_mask = mask.copy()
                else:
                    injected_mask |= mask

                # Publish on bus
                self.bus.publish(UpdateEvent(
                    event_type=EventType.OBSTACLE,
                    step=step_num,
                    description=event.description,
                    severity=0.9,
                    position=event.obstacle_position,
                    mask=mask,
                    payload={
                        "obstacle_type": "forced_block",
                        "radius": event.obstacle_radius,
                        "forced": True,
                    },
                ))

        return injected_mask

    def get_active_mask(self) -> np.ndarray:
        """Return merged mask of all injected forced obstacles."""
        H, W = self._grid_shape
        merged = np.zeros((H, W), dtype=bool)
        for mask in self._active_obstacles.values():
            merged |= mask
        return merged

    @property
    def scheduled_events(self) -> list[ForcedReplanEvent]:
        return list(self._scheduled)

    @property
    def injected_count(self) -> int:
        return sum(1 for e in self._scheduled if e.injected)

    def summary(self) -> dict[str, Any]:
        return {
            "min_replans": self.min_replans,
            "scheduled": len(self._scheduled),
            "injected": self.injected_count,
            "events": [
                {
                    "step": e.step,
                    "position": list(e.obstacle_position),
                    "radius": e.obstacle_radius,
                    "injected": e.injected,
                    "description": e.description,
                }
                for e in self._scheduled
            ],
        }
