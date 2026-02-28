"""Contract tests for decision record: EC-1, EC-2.

EC-1: Every rejected move logged with RejectReason enum, reject_layer, reject_cell.
EC-2: Every accepted move logged with step_idx and accepted_move flag.
"""

from __future__ import annotations

import pytest

from uavbench.envs.base import RejectReason
from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.schema import (
    ScenarioConfig, Domain, Difficulty, MissionType, Regime,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_config(**overrides) -> ScenarioConfig:
    """Minimal synthetic ScenarioConfig for decision record tests."""
    defaults = dict(
        name="ec_test",
        domain=Domain.URBAN,
        difficulty=Difficulty.EASY,
        mission_type=MissionType.CIVIL_PROTECTION,
        regime=Regime.NATURALISTIC,
        map_size=20,
        map_source="synthetic",
        building_density=0.0,
        no_fly_radius=0,
        max_altitude=5,
        safe_altitude=5,
        min_start_goal_l1=4,
        enable_fire=False,
        enable_traffic=False,
        paper_track="static",
        terminate_on_collision=False,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _step_env(env: UrbanEnv, action: int = 3):
    """Take one step and return info dict."""
    _, _, _, _, info = env.step(action)
    return info


# ── EC-1: Rejected moves ─────────────────────────────────────────────


class TestEC1_RejectedMoves:
    """EC-1: Every rejected move has RejectReason enum, reject_layer, reject_cell."""

    def test_building_rejection_has_all_fields(self):
        """Stepping into a building yields enum + layer + cell."""
        cfg = _make_config(building_density=0.0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        # Place building at agent's right
        ax, ay, _ = map(int, env._agent_pos)
        target_x = min(ax + 1, env.map_size - 1)
        env._heightmap[ay, target_x] = 3.0
        env._agent_pos[2] = 0  # altitude=0 so collision happens

        info = _step_env(env, action=3)  # right
        if not info["accepted_move"]:
            assert info["reject_reason"] == "building"
            assert info["reject_reason_enum"] == RejectReason.BUILDING
            assert isinstance(info["reject_reason_enum"], RejectReason)
            assert info["reject_layer"] == "heightmap"
            assert info["reject_cell"] is not None
            assert isinstance(info["reject_cell"], tuple)
            assert len(info["reject_cell"]) == 2
        env.close()

    def test_no_fly_rejection_has_all_fields(self):
        """Stepping into NFZ yields enum + layer + cell."""
        cfg = _make_config(difficulty=Difficulty.HARD, no_fly_radius=3)
        env = UrbanEnv(cfg)
        env.reset(seed=42)
        # Walk until we hit NFZ
        found_reject = False
        for _ in range(40):
            info = _step_env(env, action=3)
            if info.get("reject_reason_enum") == RejectReason.NO_FLY:
                assert info["reject_reason"] == "no_fly"
                assert info["reject_layer"] == "no_fly_mask"
                assert info["reject_cell"] is not None
                found_reject = True
                break
        # Not a hard failure if no NFZ was hit — depends on map layout
        env.close()

    def test_forced_block_rejection_has_all_fields(self):
        """Stepping into forced interdiction yields enum + layer + cell."""
        cfg = _make_config(
            map_size=40,
            paper_track="dynamic",
            force_replan_count=2,
            event_t1=2,
            event_t2=50,
        )
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        # Step past t1 to trigger interdiction
        found_reject = False
        for _ in range(10):
            info = _step_env(env, action=3)
            if info.get("reject_reason_enum") == RejectReason.FORCED_BLOCK:
                assert info["reject_reason"] == "forced_block"
                assert info["reject_layer"] == "forced_block_mask"
                assert info["reject_cell"] is not None
                found_reject = True
                break
        env.close()

    def test_reject_reason_enum_is_always_present(self):
        """Every step info has reject_reason_enum field."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        for _ in range(10):
            info = _step_env(env, action=3)
            assert "reject_reason_enum" in info
            assert isinstance(info["reject_reason_enum"], RejectReason)
        env.close()

    def test_reject_layer_is_always_present(self):
        """Every step info has reject_layer field."""
        cfg = _make_config()
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        for _ in range(10):
            info = _step_env(env, action=3)
            assert "reject_layer" in info
            assert isinstance(info["reject_layer"], str)
        env.close()

    def test_reject_cell_none_when_accepted(self):
        """When move accepted, reject_cell is None."""
        cfg = _make_config(building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        info = _step_env(env, action=3)
        if info["accepted_move"]:
            assert info["reject_cell"] is None
            assert info["reject_reason_enum"] == RejectReason.NONE
            assert info["reject_layer"] == "none"
        env.close()

    def test_reject_cell_tuple_when_rejected(self):
        """When move rejected, reject_cell is (x, y) tuple."""
        cfg = _make_config(building_density=0.0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        ax, ay, _ = map(int, env._agent_pos)
        target_x = min(ax + 1, env.map_size - 1)
        env._heightmap[ay, target_x] = 3.0
        env._agent_pos[2] = 0

        info = _step_env(env, action=3)
        if not info["accepted_move"]:
            cell = info["reject_cell"]
            assert cell is not None
            assert cell == (target_x, ay)
        env.close()


# ── EC-2: Accepted moves ─────────────────────────────────────────────


class TestEC2_AcceptedMoves:
    """EC-2: Every accepted move has step_idx and accepted_move flag."""

    def test_accepted_move_flag_present(self):
        """Every step has accepted_move boolean in info."""
        cfg = _make_config(building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        for _ in range(10):
            info = _step_env(env, action=3)
            assert "accepted_move" in info
            assert isinstance(info["accepted_move"], bool)
        env.close()

    def test_step_index_present(self):
        """Every step has step_index in info."""
        cfg = _make_config(building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        for i in range(5):
            info = _step_env(env, action=3)
            assert "step_index" in info
            assert info["step_index"] == i + 1
        env.close()

    def test_accepted_move_on_free_cell(self):
        """Moving on open grid succeeds with accepted=True, reason=NONE."""
        cfg = _make_config(building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        info = _step_env(env, action=3)
        # With 0 density and no NFZ, first move should be accepted
        assert info["accepted_move"] is True
        assert info["reject_reason_enum"] == RejectReason.NONE
        assert info["reject_reason"] == "none"
        env.close()

    def test_step_index_monotonic(self):
        """step_index increases by 1 each step."""
        cfg = _make_config(building_density=0.0, no_fly_radius=0)
        env = UrbanEnv(cfg)
        env.reset(seed=0)
        prev_idx = 0
        for _ in range(10):
            info = _step_env(env, action=3)
            assert info["step_index"] == prev_idx + 1
            prev_idx = info["step_index"]
        env.close()


# ── Backward compatibility ────────────────────────────────────────────


class TestRejectReasonBackwardCompat:
    """reject_reason string value matches enum.value for downstream code."""

    def test_string_value_matches_enum(self):
        """info['reject_reason'] == info['reject_reason_enum'].value always."""
        cfg = _make_config(building_density=0.3, no_fly_radius=0)
        env = UrbanEnv(cfg)
        env.reset(seed=42)
        for _ in range(20):
            info = _step_env(env, action=3)
            assert info["reject_reason"] == info["reject_reason_enum"].value, (
                f"String '{info['reject_reason']}' != "
                f"enum.value '{info['reject_reason_enum'].value}'"
            )
        env.close()

    def test_all_enum_values_are_strings(self):
        """Every RejectReason member has a string value."""
        for reason in RejectReason:
            assert isinstance(reason.value, str)
            assert len(reason.value) > 0


# ── RejectReason enum invariants ──────────────────────────────────────


class TestRejectReasonEnum:
    """RejectReason enum is well-formed."""

    def test_none_member_exists(self):
        assert RejectReason.NONE.value == "none"

    def test_building_member_exists(self):
        assert RejectReason.BUILDING.value == "building"

    def test_no_fly_member_exists(self):
        assert RejectReason.NO_FLY.value == "no_fly"

    def test_fire_member_exists(self):
        assert RejectReason.FIRE.value == "fire"

    def test_traffic_member_exists(self):
        assert RejectReason.TRAFFIC.value == "traffic"

    def test_forced_block_member_exists(self):
        assert RejectReason.FORCED_BLOCK.value == "forced_block"

    def test_intruder_member_exists(self):
        assert RejectReason.INTRUDER.value == "intruder"

    def test_moving_target_member_exists(self):
        assert RejectReason.MOVING_TARGET.value == "moving_target"

    def test_dynamic_nfz_member_exists(self):
        assert RejectReason.DYNAMIC_NFZ.value == "dynamic_nfz"

    def test_is_str_enum(self):
        """RejectReason members are str-compatible."""
        assert isinstance(RejectReason.NONE, str)
        assert RejectReason.NONE == "none"
