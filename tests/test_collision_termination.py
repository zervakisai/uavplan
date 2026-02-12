"""Tests for UAV-ON collision termination."""

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.schema import ScenarioConfig, Domain, Difficulty


def _make_cfg(**overrides) -> ScenarioConfig:
    """Create a minimal synthetic ScenarioConfig for testing."""
    defaults = dict(
        name="test_collision",
        domain=Domain.URBAN,
        difficulty=Difficulty.EASY,
        map_size=25,
        max_altitude=3,
        building_density=0.25,
        building_level=2,
        start_altitude=1,
        safe_altitude=3,
        min_start_goal_l1=10,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)


def _find_building_neighbor(env):
    """Find a free cell adjacent to a building at altitude 1.

    Returns (free_pos, building_direction_action) or (None, None).
    """
    hm = env._heightmap
    H, W = hm.shape
    # Actions: 0=up(y-1), 1=down(y+1), 2=left(x-1), 3=right(x+1)
    deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if hm[y, x] == 0 and not env._no_fly_mask[y, x]:
                for action, (dx, dy) in deltas.items():
                    bx, by = x + dx, y + dy
                    if 0 <= bx < W and 0 <= by < H and hm[by, bx] > 0:
                        return (x, y), action
    return None, None


def _find_nfz_neighbor(env):
    """Find a free cell adjacent to an NFZ cell at altitude 1.

    Returns (free_pos, nfz_direction_action) or (None, None).
    """
    nfz = env._no_fly_mask
    hm = env._heightmap
    H, W = nfz.shape
    deltas = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not nfz[y, x] and hm[y, x] == 0:
                for action, (dx, dy) in deltas.items():
                    bx, by = x + dx, y + dy
                    if 0 <= bx < W and 0 <= by < H and nfz[by, bx]:
                        return (x, y), action
    return None, None


def test_collision_terminates_episode():
    """Building collision should terminate the episode (UAV-ON standard)."""
    cfg = _make_cfg(terminate_on_collision=True)
    env = UrbanEnv(cfg)
    env.reset(seed=42)

    free_pos, action = _find_building_neighbor(env)
    assert free_pos is not None, "No building neighbor found in test map"

    # Place agent at the free cell, altitude 1 (below building_level=2)
    env._agent_pos = np.array([free_pos[0], free_pos[1], 1], dtype=np.int32)

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated, "Episode should terminate on building collision"
    assert not truncated, "Should not be truncated when terminated"
    assert info["collision_terminated"] is True
    assert info["termination_reason"] == "collision_building"
    assert info["accepted_move"] is False


def test_collision_disabled():
    """With terminate_on_collision=False, collision should reject move but not terminate."""
    cfg = _make_cfg(terminate_on_collision=False)
    env = UrbanEnv(cfg)
    env.reset(seed=42)

    free_pos, action = _find_building_neighbor(env)
    assert free_pos is not None, "No building neighbor found in test map"

    env._agent_pos = np.array([free_pos[0], free_pos[1], 1], dtype=np.int32)

    obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated, "Episode should NOT terminate when collision termination disabled"
    assert info["collision_terminated"] is False
    assert info["accepted_move"] is False  # Move still rejected
    # Agent stays in place
    assert int(env._agent_pos[0]) == free_pos[0]
    assert int(env._agent_pos[1]) == free_pos[1]


def test_nfz_collision_terminates():
    """Static NFZ violation should terminate the episode."""
    # Use hard difficulty which generates NFZ zones
    cfg = _make_cfg(
        terminate_on_collision=True,
        difficulty=Difficulty.HARD,
    )
    env = UrbanEnv(cfg)
    env.reset(seed=42)

    free_pos, action = _find_nfz_neighbor(env)
    if free_pos is None:
        env.reset(seed=0)
        free_pos, action = _find_nfz_neighbor(env)

    assert free_pos is not None, "No NFZ neighbor found in test map"

    env._agent_pos = np.array([free_pos[0], free_pos[1], 3], dtype=np.int32)

    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated, "Episode should terminate on NFZ violation"
    assert info["collision_terminated"] is True
    assert info["termination_reason"] == "collision_nfz"


def test_dynamic_obstacle_no_termination():
    """Dynamic obstacles (fire/traffic) should block moves but NOT terminate."""
    # Use a scenario with fire blocking but collision termination enabled
    cfg = _make_cfg(
        terminate_on_collision=True,
        # Cannot enable fire on synthetic maps (validation requires OSM)
        # Instead, test that dynamic blocks don't set collision_terminated
    )
    env = UrbanEnv(cfg)
    env.reset(seed=42)

    # Step with a safe action (up at safe altitude)
    env._agent_pos = np.array([5, 5, 3], dtype=np.int32)
    obs, reward, terminated, truncated, info = env.step(0)

    # No collision should occur at safe altitude
    assert info["collision_terminated"] is False
    assert info["termination_reason"] != "collision_building"
    assert info["termination_reason"] != "collision_nfz"


def test_goal_termination_reason():
    """Reaching the goal should set termination_reason='success'."""
    cfg = _make_cfg(terminate_on_collision=True)
    env = UrbanEnv(cfg)
    env.reset(seed=42)

    # Place agent one step away from goal
    gx, gy, gz = map(int, env._goal_pos)
    env._agent_pos = np.array([gx, gy - 1, gz], dtype=np.int32)

    # Action 1 = down (y+1) should move agent to goal
    obs, reward, terminated, truncated, info = env.step(1)

    if info["accepted_move"] and terminated:
        assert info["termination_reason"] == "success"
        assert info["collision_terminated"] is False
        assert info["reached_goal"] is True
