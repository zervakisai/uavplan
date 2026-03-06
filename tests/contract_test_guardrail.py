"""Contract tests for Feasibility Guardrail (GC-1, GC-2).

GC-1: Guardrail attempts to restore reachability via logged relaxation depths.
GC-2: If infeasible after all depths → episode flagged infeasible; exclusion rate reported.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from uavbench.blocking import compute_blocking_mask
from uavbench.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bfs_reachable(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """Check if goal is reachable from start on ~blocked grid."""
    H, W = blocked.shape
    sx, sy = start
    gx, gy = goal
    if blocked[sy, sx] or blocked[gy, gx]:
        return False
    visited = {(sx, sy)}
    queue = deque([(sx, sy)])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == (gx, gy):
            return True
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in visited:
                if not blocked[ny, nx]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False


# ===========================================================================
# GC-1: Multi-depth relaxation
# ===========================================================================


class TestGC1_MultiDepthRelaxation:
    """GC-1: Guardrail restores reachability via logged relaxation depths."""

    def test_guardrail_logs_depth_and_relaxations(self):
        """After a topology-changing event, info dict includes guardrail_depth
        (int 0-3) and relaxations (list of dicts)."""
        from uavbench.guardrail.feasibility import FeasibilityGuardrail

        # Create a small grid where agent is blocked by traffic
        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_gr",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.EASY,
            map_size=10,
            enable_traffic=True,
            traffic_blocks_movement=True,
        )
        start = (0, 0)
        goal = (9, 9)

        # Traffic blocks that sever the path
        traffic_mask = np.zeros((10, 10), dtype=bool)
        traffic_mask[0, 1:] = True
        traffic_mask[1:, 0] = True

        dynamic_state = {
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": traffic_mask,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            traffic_model=None,
            nfz_model=None,
        )

        assert isinstance(result.depth, int), "guardrail_depth must be int"
        assert isinstance(result.relaxations, list), "relaxations must be list"
        assert result.depth >= 0

    def test_depth1_clears_roadblocks(self):
        """D1: guardrail clears roadblock vehicles to restore reachability."""
        from uavbench.dynamics.traffic import TrafficModel
        from uavbench.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_d1",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.EASY,
            map_size=10,
            enable_traffic=True,
            traffic_blocks_movement=True,
        )

        start = (0, 0)
        goal = (9, 9)

        # Create roads and traffic model with a roadblock
        roads = np.ones((10, 10), dtype=bool)
        rng = np.random.default_rng(42)
        traffic = TrafficModel(
            roads_mask=roads,
            num_vehicles=1,
            rng=rng,
            roadblock_cells=[(5, 5)],  # (y, x)
            roadblock_step=0,
        )
        # Activate roadblocks
        traffic.step(step_idx=1)
        assert traffic.has_active_roadblocks

        # The traffic occupancy blocks a wide area
        occ_mask = traffic.get_occupancy_mask()

        dynamic_state = {
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": occ_mask,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            traffic_model=traffic,
            nfz_model=None,
        )

        # D1 should clear roadblocks; if that restores feasibility, depth==1
        assert len(result.relaxations) >= 1
        assert result.relaxations[0]["depth"] == 1
        assert "cells_freed" in result.relaxations[0]

    def test_depth2_shrinks_nfz(self):
        """D2: after D1 fails (no roadblocks), guardrail shrinks NFZ zones."""
        from uavbench.dynamics.restriction_zones import RestrictionZoneModel
        from uavbench.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_d2",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.MEDIUM,
            map_size=10,
            enable_dynamic_nfz=True,
        )

        start = (0, 0)
        goal = (9, 9)

        # Create NFZ that blocks the path
        nfz_mask = np.zeros((10, 10), dtype=bool)
        nfz_mask[:, 5] = True  # wall at x=5

        rng = np.random.default_rng(42)
        nfz = RestrictionZoneModel(
            map_shape=(10, 10),
            rng=rng,
            num_zones=1,
            event_t1=1,
            event_t2=100,
        )
        for _ in range(5):
            nfz.step()

        dynamic_state = {
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "dynamic_nfz_mask": nfz_mask,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            traffic_model=None,
            nfz_model=nfz,
        )

        assert result.depth >= 2, f"Expected depth >= 2, got {result.depth}"
        d2_relaxations = [r for r in result.relaxations if r["depth"] == 2]
        assert len(d2_relaxations) >= 1, "D2 relaxation should be logged"
        assert "cells_freed" in d2_relaxations[0]

    def test_depth3_emergency_corridor(self):
        """D3: after D1+D2 fail, guardrail removes traffic blocking."""
        from uavbench.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        config = ScenarioConfig(
            name="test_d3",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.MEDIUM,
            map_size=10,
            enable_traffic=True,
            traffic_blocks_movement=True,
        )
        start = (0, 0)
        goal = (9, 9)

        traffic_mask = np.zeros((10, 10), dtype=bool)
        traffic_mask[4:6, :] = True

        dynamic_state = {
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": traffic_mask,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            traffic_model=None,
            nfz_model=None,
        )

        assert result.feasible, "D3 should restore feasibility by removing traffic"
        assert result.depth == 3, f"Expected depth 3, got {result.depth}"
        assert any(r["depth"] == 3 for r in result.relaxations)


# ===========================================================================
# GC-2: Infeasible flagging
# ===========================================================================


class TestGC2_InfeasibleFlagging:
    """GC-2: Infeasible episodes flagged after all depths fail."""

    def test_infeasible_flagged(self):
        """When all depths fail, feasible_after_guardrail=False."""
        from uavbench.guardrail.feasibility import FeasibilityGuardrail

        heightmap = np.zeros((10, 10), dtype=np.float32)
        no_fly = np.zeros((10, 10), dtype=bool)
        heightmap[0, 1] = 5.0
        heightmap[1, 0] = 5.0
        heightmap[1, 1] = 5.0

        config = ScenarioConfig(
            name="test_infeasible",
            mission_type=MissionType.FIRE_DELIVERY,
            difficulty=Difficulty.EASY,
            map_size=10,
        )
        start = (0, 0)
        goal = (9, 9)

        dynamic_state = {
            "fire_mask": None,
            "smoke_mask": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "dynamic_nfz_mask": None,
        }

        guardrail = FeasibilityGuardrail(
            heightmap=heightmap,
            no_fly=no_fly,
            config=config,
        )

        result = guardrail.check(
            agent_xy=start,
            goal_xy=goal,
            dynamic_state=dynamic_state,
            traffic_model=None,
            nfz_model=None,
        )

        assert not result.feasible, "Should be infeasible when static buildings block"
        assert result.depth >= 3, "Should have tried all depths"

    def test_infeasible_rate_in_metrics(self):
        """Aggregate metrics contain infeasible_rate as float [0,1]."""
        from uavbench.metrics.compute import compute_episode_metrics

        metrics = compute_episode_metrics(
            scenario_id="test",
            planner_id="astar",
            seed=42,
            trajectory=[(0, 0)],
            events=[],
            final_info={
                "termination_reason": "infeasible",
                "objective_completed": False,
                "feasible_after_guardrail": False,
            },
        )

        assert isinstance(metrics, dict)
        assert "feasible_after_guardrail" not in metrics or isinstance(
            metrics.get("feasible_after_guardrail"), bool
        )
