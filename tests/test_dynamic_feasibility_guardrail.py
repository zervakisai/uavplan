from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.loader import load_scenario
from uavbench.cli.benchmark import scenario_path


def test_guardrail_relaxes_disconnected_dynamic_state():
    cfg = load_scenario(scenario_path("osm_athens_comms_denied_hard_downtown"))
    env = UrbanEnv(cfg)
    env.reset(seed=0)

    # Force a disconnected state, then verify guardrail relaxes it.
    env._forced_block_mask[:] = True  # type: ignore[attr-defined]
    current = (int(env._agent_pos[0]), int(env._agent_pos[1]))
    goal = (int(env._goal_pos[0]), int(env._goal_pos[1]))

    relaxed = env._enforce_feasibility_guardrail(current, goal)  # type: ignore[attr-defined]
    assert relaxed
    status = env._last_guardrail_status  # type: ignore[attr-defined]
    assert status["reachability_failed_before_relax"] is True
    assert status["feasible_after_guardrail"] is True
