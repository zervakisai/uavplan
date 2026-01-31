from pathlib import Path

import gymnasium as gym
import numpy as np

from uavbench.scenarios.loader import load_scenario
from uavbench.envs.urban import UrbanEnv


def test_urban_env_reset_step():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_easy.yaml"))
    env = UrbanEnv(cfg)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    # Κάνε λίγα βήματα
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)

def test_urban_env_logs_trajectory():
    cfg = load_scenario(Path("src/uavbench/scenarios/configs/urban_easy.yaml"))
    env = UrbanEnv(cfg)
    obs, info = env.reset(seed=0)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    traj = env.trajectory
    assert len(traj) == 3
    assert traj[0]["step"] == 1

