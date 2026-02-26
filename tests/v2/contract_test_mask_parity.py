"""Contract tests for Mask Parity (MP-1).

MP-1: ONE compute_blocking_mask() used by BOTH step legality AND guardrail BFS.
No parallel mask computation allowed.
"""

from __future__ import annotations

import importlib
import inspect
import textwrap

import numpy as np
import pytest

from uavbench2.blocking import compute_blocking_mask
from uavbench2.envs.urban import UrbanEnvV2
from uavbench2.scenarios.schema import Difficulty, MissionType, ScenarioConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dynamic_config(**overrides) -> ScenarioConfig:
    """Create a dynamic scenario config with fire + traffic enabled."""
    defaults = dict(
        name="test_mask_parity",
        mission_type=MissionType.CIVIL_PROTECTION,
        difficulty=Difficulty.MEDIUM,
        map_size=20,
        building_density=0.15,
        max_episode_steps=100,
        terminate_on_collision=False,
        enable_fire=True,
        enable_traffic=True,
        fire_blocks_movement=True,
        traffic_blocks_movement=True,
        fire_ignition_points=2,
        num_emergency_vehicles=3,
        event_t1=10,
        event_t2=50,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


# ===========================================================================
# MP-1: Single blocking mask
# ===========================================================================


class TestMP1_SingleDefinition:
    """MP-1: There is exactly ONE compute_blocking_mask definition."""

    def test_single_definition(self):
        """grep finds exactly 1 `def compute_blocking_mask` in src/uavbench2/."""
        import subprocess

        result = subprocess.run(
            ["grep", "-r", "def compute_blocking_mask", "src/uavbench2/"],
            capture_output=True,
            text=True,
        )
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        assert len(lines) == 1, (
            f"MP-1: expected exactly 1 compute_blocking_mask definition, "
            f"found {len(lines)}:\n" + "\n".join(lines)
        )

    def test_step_uses_blocking_mask(self):
        """env.step() code path calls compute_blocking_mask()."""
        source = inspect.getsource(UrbanEnvV2.step)
        assert "compute_blocking_mask" in source, (
            "MP-1: UrbanEnvV2.step() must call compute_blocking_mask()"
        )

    def test_blocking_mask_importable_from_canonical_location(self):
        """compute_blocking_mask is importable from uavbench2.blocking."""
        mod = importlib.import_module("uavbench2.blocking")
        assert hasattr(mod, "compute_blocking_mask"), (
            "MP-1: compute_blocking_mask must be in uavbench2.blocking"
        )


class TestMP1_MaskConsistency:
    """MP-1: The mask used by step legality equals the mask from compute_blocking_mask."""

    def test_static_mask_matches(self):
        """On a static episode, the mask from step matches direct computation."""
        config = ScenarioConfig(
            name="test_static_mask",
            mission_type=MissionType.CIVIL_PROTECTION,
            difficulty=Difficulty.EASY,
            map_size=10,
            building_density=0.3,
            max_episode_steps=50,
            terminate_on_collision=False,
            fixed_start_xy=(0, 0),
            fixed_goal_xy=(9, 9),
        )
        env = UrbanEnvV2(config)
        env.reset(seed=42)

        heightmap, no_fly, _, _ = env.export_planner_inputs()
        expected_mask = compute_blocking_mask(heightmap, no_fly, config)

        # The mask should block exactly the building cells
        assert expected_mask.shape == (10, 10)
        assert expected_mask.dtype == bool
        # Buildings are where heightmap > 0
        np.testing.assert_array_equal(
            expected_mask, (heightmap > 0) | no_fly,
            err_msg="MP-1: static mask must equal (heightmap > 0) | no_fly",
        )

    def test_dynamic_mask_includes_all_layers(self):
        """When dynamic_state has fire/traffic, compute_blocking_mask merges them."""
        config = _make_dynamic_config()
        heightmap = np.zeros((20, 20), dtype=np.float32)
        no_fly = np.zeros((20, 20), dtype=bool)

        # Simulate dynamic state with fire and traffic
        fire_mask = np.zeros((20, 20), dtype=bool)
        fire_mask[5, 5] = True
        fire_mask[5, 6] = True

        traffic_occ = np.zeros((20, 20), dtype=bool)
        traffic_occ[10, 10] = True

        dynamic_state = {
            "fire_mask": fire_mask,
            "smoke_mask": np.zeros((20, 20), dtype=np.float32),
            "traffic_positions": None,
            "forced_block_mask": None,
            "risk_cost_map": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": traffic_occ,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        mask = compute_blocking_mask(heightmap, no_fly, config, dynamic_state)

        # Fire cells should be blocked (fire_blocks_movement=True)
        assert mask[5, 5], "MP-1: fire cell (5,5) should be blocked"
        assert mask[5, 6], "MP-1: fire cell (5,6) should be blocked"
        # Traffic occupancy should be blocked (traffic_blocks_movement=True)
        assert mask[10, 10], "MP-1: traffic cell (10,10) should be blocked"
        # Free cells should not be blocked
        assert not mask[0, 0], "MP-1: free cell (0,0) should not be blocked"

    def test_smoke_threshold(self):
        """Smoke >= 0.3 blocks when fire_blocks_movement is True."""
        config = _make_dynamic_config()
        heightmap = np.zeros((20, 20), dtype=np.float32)
        no_fly = np.zeros((20, 20), dtype=bool)

        smoke = np.zeros((20, 20), dtype=np.float32)
        smoke[3, 3] = 0.5  # above threshold
        smoke[4, 4] = 0.2  # below threshold

        dynamic_state = {
            "fire_mask": None,
            "smoke_mask": smoke,
            "traffic_positions": None,
            "forced_block_mask": None,
            "risk_cost_map": None,
            "traffic_closure_mask": None,
            "traffic_occupancy_mask": None,
            "moving_target_buffer": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
        }

        mask = compute_blocking_mask(heightmap, no_fly, config, dynamic_state)

        assert mask[3, 3], "MP-1: smoke >= 0.3 should block"
        assert not mask[4, 4], "MP-1: smoke < 0.3 should not block"
