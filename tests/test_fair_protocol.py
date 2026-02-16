from dataclasses import replace

from uavbench.cli.benchmark import scenario_path
from uavbench.envs.urban import UrbanEnv
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.schema import InterdictionReferencePlanner


def test_loader_protocol_defaults_present():
    cfg = load_scenario(scenario_path("osm_athens_border_surveillance_easy"))
    assert cfg.interdiction_reference_planner == InterdictionReferencePlanner.THETA_STAR
    assert cfg.plan_budget_static_ms > 0.0
    assert cfg.plan_budget_dynamic_ms > 0.0
    assert cfg.replan_every_steps >= 1
    assert cfg.max_replans_per_episode >= 1


def test_reference_planner_is_logged_for_interdiction():
    cfg = load_scenario(scenario_path("osm_athens_comms_denied_hard_downtown"))
    cfg = replace(
        cfg,
        interdiction_reference_planner=InterdictionReferencePlanner.ASTAR,
        force_replan_count=1,
        event_t1=8,
        event_t2=None,
    )
    env = UrbanEnv(cfg)
    env.reset(seed=0)
    env._maybe_trigger_interdictions(8)  # type: ignore[attr-defined]
    ev = next(e for e in env.events if e["type"] == "path_interdiction_1")
    assert ev["payload"]["reference_planner"] == "astar"
