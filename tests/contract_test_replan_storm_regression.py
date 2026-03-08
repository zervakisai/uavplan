"""Contract tests for Replan Storm Regression (RS-1).

RS-1: Path-progress tracking prevents replan storms (≤20% naive replans).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from uavbench.blocking import compute_blocking_mask
from uavbench.envs.urban import UrbanEnvV2
from uavbench.planners import PLANNERS
from uavbench.planners.base import PlannerBase
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_config(**overrides) -> ScenarioConfig:
    defaults = dict(
        name="test_replan_storm",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.MEDIUM,
        map_size=30,
        building_density=0.12,
        max_episode_steps=200,
        terminate_on_collision=False,
        enable_fire=True,
        fire_blocks_movement=True,
        fire_ignition_points=2,
        enable_traffic=True,
        traffic_blocks_movement=True,
        num_emergency_vehicles=2,
        replan_every_steps=6,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _mask_hash(mask: np.ndarray) -> str:
    """SHA-256 hash of a blocking mask for naive-replan detection."""
    return hashlib.sha256(mask.tobytes()).hexdigest()


def _run_replan_episode(
    config: ScenarioConfig,
    planner_id: str,
    seed: int,
) -> tuple[list[dict], int]:
    """Run an episode tracking all replans.

    Returns (replan_records, total_steps) where each record has:
        position, mask_hash, step, is_naive.
    """
    env = UrbanEnvV2(config)
    obs, info = env.reset(seed=seed)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    planner_cls = PLANNERS[planner_id]
    planner = planner_cls(heightmap, no_fly, config)
    plan_result = planner.plan(start_xy, goal_xy)
    path = plan_result.path if plan_result.success else []
    path_idx = 0

    replan_records: list[dict] = []
    prev_pos = None
    prev_hash = None
    step_idx = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        step_idx += 1

        # Action from path
        from uavbench.benchmark.runner import _path_to_action
        action = _path_to_action(env.agent_xy, path, path_idx)
        obs, reward, terminated, truncated, info = env.step(action)

        if path_idx < len(path) - 1:
            if env.agent_xy == path[path_idx + 1]:
                path_idx += 1

        dyn_state = env.get_dynamic_state()
        planner.update(dyn_state)

        should, reason = planner.should_replan(
            env.agent_xy, path, dyn_state, step_idx
        )
        if should:
            mask = compute_blocking_mask(heightmap, no_fly, config, dyn_state)
            mh = _mask_hash(mask)
            is_naive = (env.agent_xy == prev_pos and mh == prev_hash)

            replan_records.append({
                "position": env.agent_xy,
                "mask_hash": mh,
                "step": step_idx,
                "is_naive": is_naive,
                "reason": reason,
            })

            prev_pos = env.agent_xy
            prev_hash = mh

            new_result = planner.plan(env.agent_xy, goal_xy)
            if new_result.success:
                path = new_result.path
                path_idx = 0

    return replan_records, step_idx


# ===========================================================================
# RS-1: Replan storm prevention
# ===========================================================================


class TestRS1_ReplanStormPrevention:
    """RS-1: Path-progress tracking prevents replan storms."""

    def test_naive_replan_ratio(self):
        """Over a dynamic episode with periodic_replan,
        naive_replan_count / total_replan_count <= 0.20."""
        config = _make_dynamic_config(
            replan_every_steps=6,
            max_episode_steps=200,
        )

        records, total_steps = _run_replan_episode(
            config, "periodic_replan", seed=42
        )

        if len(records) == 0:
            pytest.skip("No replans occurred — cannot test ratio")

        naive_count = sum(1 for r in records if r["is_naive"])
        ratio = naive_count / len(records)

        assert ratio <= 0.20, (
            f"RS-1: naive replan ratio {ratio:.2f} ({naive_count}/{len(records)}) "
            f"exceeds 20% threshold"
        )

    def test_naive_definition(self):
        """Naive replan = same position AND identical blocking mask hash."""
        config = _make_dynamic_config(
            replan_every_steps=4,
            max_episode_steps=100,
        )

        records, _ = _run_replan_episode(config, "periodic_replan", seed=42)

        for i, rec in enumerate(records):
            if rec["is_naive"]:
                # Verify: position and mask hash match previous replan
                assert i > 0, "First replan cannot be naive"
                prev = records[i - 1]
                assert rec["position"] == prev["position"], (
                    f"Naive replan at step {rec['step']} should have same position"
                )
                assert rec["mask_hash"] == prev["mask_hash"], (
                    f"Naive replan at step {rec['step']} should have same mask hash"
                )

    def test_cooldown_enforced(self):
        """No two replans occur within 3 steps of each other (unless forced)."""
        config = _make_dynamic_config(
            replan_every_steps=6,
            max_episode_steps=200,
        )

        records, _ = _run_replan_episode(config, "periodic_replan", seed=42)

        for i in range(1, len(records)):
            gap = records[i]["step"] - records[i - 1]["step"]
            assert gap >= 3, (
                f"RS-1: replans at steps {records[i-1]['step']} and "
                f"{records[i]['step']} are only {gap} steps apart (minimum 3)"
            )
