"""Benchmark runner (RU-1, RU-3, RU-4).

Exactly ONE runner. Orchestrates: scenario load -> env reset -> plan ->
step loop -> metrics. Owns the authoritative step_idx (EV-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from uavbench2.envs.urban import UrbanEnvV2
from uavbench2.metrics.compute import compute_episode_metrics
from uavbench2.planners import PLANNERS
from uavbench2.scenarios.loader import load_scenario


@dataclass
class EpisodeResult:
    """Result of a single episode run.

    Attributes used by determinism tests (DC-2):
        events: list of event dicts
        trajectory: list of (x, y) agent positions
        metrics: episode metrics dict
        frame_hashes: per-frame hash strings (empty if not rendering)
    """

    events: list[dict] = field(default_factory=list)
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    frame_hashes: list[str] = field(default_factory=list)


def run_episode(
    scenario_id: str,
    planner_id: str,
    seed: int,
    render: bool = False,
) -> EpisodeResult:
    """Run a single deterministic episode (RU-3).

    Args:
        scenario_id: registered scenario name
        planner_id: registered planner name
        seed: RNG seed (DC-1)
        render: whether to capture frame hashes (Phase 8)

    Returns:
        EpisodeResult with events, trajectory, metrics, frame_hashes.
    """
    # Load scenario
    config = load_scenario(scenario_id)

    # Create environment
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)

    # Export planner inputs
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    # Create planner
    planner_cls = PLANNERS[planner_id]
    planner = planner_cls(heightmap, no_fly, config)

    # Initial plan
    plan_result = planner.plan(start_xy, goal_xy)

    # Track path execution
    path = plan_result.path if plan_result.success else []
    path_idx = 0
    trajectory: list[tuple[int, int]] = [start_xy]

    # Step loop (RU-4: runner owns step_idx)
    step_idx = 0
    terminated = False
    truncated = False
    final_info = info

    while not terminated and not truncated:
        step_idx += 1

        # Determine action from path
        action = _path_to_action(env.agent_xy, path, path_idx)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        final_info = info
        trajectory.append(env.agent_xy)

        # Advance path index if we moved to expected next waypoint
        if path_idx < len(path) - 1:
            next_wp = path[path_idx + 1]
            if env.agent_xy == next_wp:
                path_idx += 1

        # Update planner with dynamic state
        dyn_state = env.get_dynamic_state()
        planner.update(dyn_state)

        # Check if replan needed
        should, reason = planner.should_replan(
            env.agent_xy, path, dyn_state, step_idx
        )
        if should:
            plan_result_new = planner.plan(env.agent_xy, goal_xy)
            if plan_result_new.success:
                path = plan_result_new.path
                path_idx = 0

    # Compute metrics
    metrics = compute_episode_metrics(
        scenario_id=scenario_id,
        planner_id=planner_id,
        seed=seed,
        trajectory=trajectory,
        events=env.events,
        final_info=final_info,
        plan_result=plan_result,
    )

    return EpisodeResult(
        events=env.events,
        trajectory=trajectory,
        metrics=metrics,
        frame_hashes=[],
    )


def _path_to_action(
    agent_xy: tuple[int, int],
    path: list[tuple[int, int]],
    path_idx: int,
) -> int:
    """Convert path waypoint to an action integer.

    If we're at the end of the path or path is empty, use greedy
    navigation toward the goal (last waypoint).
    """
    # Action constants
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    if not path:
        return ACTION_STAY

    # Target: next waypoint on path
    target_idx = path_idx + 1 if path_idx + 1 < len(path) else len(path) - 1
    target = path[target_idx]

    ax, ay = agent_xy
    tx, ty = target

    if (ax, ay) == (tx, ty):
        # At target — try next waypoint or stay
        if target_idx + 1 < len(path):
            tx, ty = path[target_idx + 1]
        else:
            return ACTION_STAY

    dx = tx - ax
    dy = ty - ay

    if dx == 0 and dy == 0:
        return ACTION_STAY
    if abs(dx) >= abs(dy):
        return ACTION_RIGHT if dx > 0 else ACTION_LEFT
    else:
        return ACTION_DOWN if dy > 0 else ACTION_UP
