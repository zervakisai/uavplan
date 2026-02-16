from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.loader import load_scenario
from uavbench.cli.benchmark import scenario_path


def test_forced_interdictions_trigger_events():
    cfg = load_scenario(scenario_path("osm_athens_comms_denied_hard_downtown"))
    env = UrbanEnv(cfg)
    env.reset(seed=0)

    env._maybe_trigger_interdictions(12)  # type: ignore[attr-defined]
    env._maybe_trigger_interdictions(28)  # type: ignore[attr-defined]

    event_types = [e["type"] for e in env.events]
    assert "path_interdiction_1" in event_types
    assert "path_interdiction_2" in event_types
    assert "forced_replan_triggered" in event_types
