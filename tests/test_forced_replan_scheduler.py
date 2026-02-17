from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.loader import load_scenario
from uavbench.cli.benchmark import scenario_path
from dataclasses import replace


def test_forced_interdictions_trigger_events():
    cfg = load_scenario(scenario_path("gov_civil_protection_hard"))
    # Inject forced replan fields to test interdiction scheduling
    cfg = replace(
        cfg,
        force_replan_count=2,
        event_t1=12,
        event_t2=28,
    )
    env = UrbanEnv(cfg)
    env.reset(seed=0)

    env._maybe_trigger_interdictions(12)  # type: ignore[attr-defined]
    env._maybe_trigger_interdictions(28)  # type: ignore[attr-defined]

    event_types = [e["type"] for e in env.events]
    assert "path_interdiction_1" in event_types
    assert "path_interdiction_2" in event_types
    assert "forced_replan_triggered" in event_types
