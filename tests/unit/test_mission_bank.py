"""Tests for the Greece Government-Ready Mission Bank.

Covers:
  - Mission specs + difficulty knobs
  - Mission engine (task tracking, injections, products, metrics)
  - Policies (Greedy, LookaheadOPTW)
  - Builders (all 3 missions × 3 difficulties)
  - plan_mission() end-to-end
  - Product export
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from uavbench.missions.spec import (
    MissionID,
    MissionSpec,
    TaskSpec,
    TaskStatus,
    DifficultyKnobs,
    ProductType,
    MissionProduct,
    COMMON_METRICS,
)
from uavbench.missions.engine import MissionEngine, InjectionEvent, RuntimeTask
from uavbench.missions.policies import GreedyPolicy, LookaheadOPTWPolicy
from uavbench.missions.builders import (
    build_civil_protection,
    build_maritime_domain,
    build_critical_infrastructure,
    build_mission,
)
from uavbench.missions.runner import plan_mission, MissionResult, export_products_csv, export_episode_json


# ═══════════════════════════════════════════════════════════════════════
#  Spec + knobs
# ═══════════════════════════════════════════════════════════════════════

class TestDifficultyKnobs:
    def test_easy_defaults(self):
        k = DifficultyKnobs.easy()
        assert k.num_tasks == 4
        assert k.injection_rate == "low"
        assert k.dynamics_intensity == "static"
        assert k.comms_dropout_prob == 0.0

    def test_medium_defaults(self):
        k = DifficultyKnobs.medium()
        assert k.num_tasks == 6
        assert k.injection_rate == "medium"
        assert k.dynamics_intensity == "moderate"
        assert k.comms_dropout_prob > 0

    def test_hard_defaults(self):
        k = DifficultyKnobs.hard()
        assert k.num_tasks == 8
        assert k.injection_rate == "high"
        assert k.dynamics_intensity == "severe"
        assert k.comms_dropout_prob >= 0.1

    def test_custom_tasks(self):
        k = DifficultyKnobs.easy(num_tasks=10, time_budget=500)
        assert k.num_tasks == 10
        assert k.time_budget == 500


class TestTaskSpec:
    def test_frozen(self):
        t = TaskSpec(task_id="t1", xy=(5, 5))
        with pytest.raises(AttributeError):
            t.task_id = "t2"  # type: ignore

    def test_time_window(self):
        t = TaskSpec(task_id="t2", xy=(10, 10), time_window=(5, 50))
        assert t.time_window == (5, 50)


# ═══════════════════════════════════════════════════════════════════════
#  Engine
# ═══════════════════════════════════════════════════════════════════════

def _make_engine(num_tasks: int = 3, time_budget: int = 100) -> MissionEngine:
    """Create a simple engine for testing."""
    tasks = tuple(
        TaskSpec(task_id=f"t{i}", xy=(10 * i + 5, 10 * i + 5), weight=1.0, time_decay=0.01)
        for i in range(num_tasks)
    )
    spec = MissionSpec(
        mission_id=MissionID.CIVIL_PROTECTION,
        label="test",
        difficulty="easy",
        knobs=DifficultyKnobs.easy(num_tasks=num_tasks, time_budget=time_budget),
        initial_tasks=tasks,
        product_types=(ProductType.ALERT_TIMELINE_CSV,),
    )
    return MissionEngine(spec)


class TestMissionEngine:
    def test_initial_tasks(self):
        engine = _make_engine(3)
        assert len(engine.tasks) == 3
        assert all(t.status == TaskStatus.PENDING for t in engine.tasks)

    def test_step_increments(self):
        engine = _make_engine()
        engine.step((0, 0), {})
        assert engine.step_count == 1

    def test_task_completion(self):
        engine = _make_engine(1)
        engine.current_target = engine.tasks[0]
        pos = engine.tasks[0].xy
        events = engine.step(pos, {})
        assert "t0" in events["completions"]
        assert engine.tasks[0].status == TaskStatus.COMPLETED

    def test_task_with_service_time(self):
        task = TaskSpec(task_id="s0", xy=(5, 5), service_time=3)
        spec = MissionSpec(
            mission_id=MissionID.CIVIL_PROTECTION,
            label="svc",
            difficulty="easy",
            knobs=DifficultyKnobs.easy(time_budget=50),
            initial_tasks=(task,),
            product_types=(),
        )
        engine = MissionEngine(spec)
        engine.current_target = engine.tasks[0]

        # Need 3 steps at position to complete
        for i in range(2):
            events = engine.step((5, 5), {})
            assert "s0" not in events.get("completions", [])
        events = engine.step((5, 5), {})
        assert "s0" in events["completions"]

    def test_injection(self):
        engine = _make_engine(1, time_budget=50)
        new_task = TaskSpec(task_id="injected", xy=(20, 20), injected_at=5)
        engine.set_injection_schedule([
            InjectionEvent(step=5, task=new_task, description="test injection"),
        ])
        # Step to injection point
        for _ in range(5):
            engine.step((0, 0), {})
        assert len(engine.tasks) == 2
        assert engine.tasks[1].task_id == "injected"

    def test_time_budget_exhaustion(self):
        engine = _make_engine(1, time_budget=3)
        for _ in range(3):
            engine.step((0, 0), {})
        assert engine.done

    def test_violation_tracking(self):
        engine = _make_engine()
        engine.step((0, 0), {"in_nfz": True})
        assert engine.violation_count == 1

    def test_utility_calculation(self):
        engine = _make_engine(2, time_budget=100)
        # Complete first task immediately
        engine.current_target = engine.tasks[0]
        engine.step(engine.tasks[0].xy, {})
        u = engine.compute_task_utility()
        assert 0.0 < u <= 1.0

    def test_mission_score_penalises_risk(self):
        engine = _make_engine(1, time_budget=50)
        engine.current_target = engine.tasks[0]
        engine.step(engine.tasks[0].xy, {}, risk_at_pos=10.0)
        score = engine.compute_mission_score()
        assert score < 1.0

    def test_common_metrics_keys(self):
        engine = _make_engine()
        engine.step((0, 0), {})
        m = engine.compute_common_metrics()
        for key in COMMON_METRICS:
            assert key in m, f"Missing metric: {key}"

    def test_product_export(self):
        engine = _make_engine()
        engine.add_product(MissionProduct(
            product_type=ProductType.ALERT_TIMELINE_CSV,
            timestamp_step=1,
            data={"event_id": "e1", "detected_time": 0, "first_response_time": 1},
        ))
        products = engine.export_products()
        assert "alert_timeline.csv" in products
        assert len(products["alert_timeline.csv"]) == 1

    def test_episode_log(self):
        engine = _make_engine()
        engine.step((0, 0), {})
        log = engine.export_episode_log()
        assert "mission_id" in log
        assert "metrics" in log
        assert "tasks" in log

    def test_time_window_expiration(self):
        task = TaskSpec(task_id="tw0", xy=(5, 5), time_window=(1, 3))
        spec = MissionSpec(
            mission_id=MissionID.CIVIL_PROTECTION,
            label="tw",
            difficulty="easy",
            knobs=DifficultyKnobs.easy(time_budget=50),
            initial_tasks=(task,),
            product_types=(),
        )
        engine = MissionEngine(spec)
        for _ in range(5):
            engine.step((0, 0), {})
        assert engine.tasks[0].status == TaskStatus.EXPIRED


# ═══════════════════════════════════════════════════════════════════════
#  Policies
# ═══════════════════════════════════════════════════════════════════════

class TestPolicies:
    def test_greedy_picks_nearest(self):
        engine = _make_engine(3)
        policy = GreedyPolicy()
        # Agent at (0,0) — nearest task is t0 at (5,5)
        task = policy.select_next_task((0, 0), engine)
        assert task is not None
        assert task.task_id == "t0"

    def test_greedy_returns_none_when_all_done(self):
        engine = _make_engine(1, time_budget=50)
        engine.tasks[0].status = TaskStatus.COMPLETED
        policy = GreedyPolicy()
        assert policy.select_next_task((0, 0), engine) is None

    def test_lookahead_picks_task(self):
        engine = _make_engine(3)
        policy = LookaheadOPTWPolicy(depth=2)
        task = policy.select_next_task((0, 0), engine)
        assert task is not None

    def test_lookahead_respects_budget(self):
        """With very tight budget, lookahead should still pick something."""
        engine = _make_engine(3, time_budget=15)
        policy = LookaheadOPTWPolicy(depth=2)
        task = policy.select_next_task((0, 0), engine)
        # Should pick nearest reachable task
        assert task is not None


# ═══════════════════════════════════════════════════════════════════════
#  Builders
# ═══════════════════════════════════════════════════════════════════════

_DIFFICULTIES = ["easy", "medium", "hard"]
_MISSION_CONFIGS = [
    ("civil_protection", build_civil_protection, MissionID.CIVIL_PROTECTION),
    ("maritime_domain", build_maritime_domain, MissionID.MARITIME_DOMAIN),
    ("critical_infrastructure", build_critical_infrastructure, MissionID.CRITICAL_INFRASTRUCTURE),
]


class TestBuilders:
    @pytest.mark.parametrize("label,builder,mid", _MISSION_CONFIGS)
    @pytest.mark.parametrize("diff", _DIFFICULTIES)
    def test_builder_produces_valid_spec(self, label, builder, mid, diff):
        spec, schedule = builder(diff, map_size=64)
        assert spec.mission_id == mid
        assert spec.difficulty == diff
        assert len(spec.initial_tasks) > 0
        assert len(spec.product_types) == 3

    @pytest.mark.parametrize("label,builder,mid", _MISSION_CONFIGS)
    def test_task_count_scales(self, label, builder, mid):
        easy_spec, _ = builder("easy", map_size=64)
        hard_spec, _ = builder("hard", map_size=64)
        assert len(easy_spec.initial_tasks) <= len(hard_spec.initial_tasks)

    @pytest.mark.parametrize("label,builder,mid", _MISSION_CONFIGS)
    def test_injection_rate_scales(self, label, builder, mid):
        _, easy_sched = builder("easy", map_size=64)
        _, hard_sched = builder("hard", map_size=64)
        easy_tasks = sum(1 for e in easy_sched if e.task is not None)
        hard_tasks = sum(1 for e in hard_sched if e.task is not None)
        assert easy_tasks <= hard_tasks

    def test_build_mission_dispatch(self):
        spec, schedule = build_mission(MissionID.CIVIL_PROTECTION, "medium", 64, seed=99)
        assert spec.mission_id == MissionID.CIVIL_PROTECTION

    def test_civil_protection_categories(self):
        spec, _ = build_civil_protection("medium", map_size=64)
        cats = {t.category for t in spec.initial_tasks}
        assert "perimeter_point" in cats
        assert "corridor_checkpoint" in cats

    def test_maritime_patrol_loop(self):
        spec, _ = build_maritime_domain("easy", map_size=64)
        assert all(t.category == "patrol_waypoint" for t in spec.initial_tasks)

    def test_maritime_distress_injection(self):
        _, schedule = build_maritime_domain("medium", map_size=64)
        distress = [e for e in schedule if e.task and e.task.category == "distress_event"]
        assert len(distress) >= 1

    def test_infrastructure_time_windows(self):
        spec, _ = build_critical_infrastructure("medium", map_size=64)
        assert all(t.time_window is not None for t in spec.initial_tasks)

    def test_infrastructure_service_time(self):
        spec, _ = build_critical_infrastructure("easy", map_size=64)
        assert all(t.service_time > 0 for t in spec.initial_tasks)


# ═══════════════════════════════════════════════════════════════════════
#  plan_mission() end-to-end
# ═══════════════════════════════════════════════════════════════════════

def _make_open_map(size: int = 32):
    """Create a simple open map for testing."""
    heightmap = np.zeros((size, size), dtype=np.float32)
    no_fly = np.zeros((size, size), dtype=bool)
    return heightmap, no_fly


class TestPlanMission:
    @pytest.mark.parametrize("mission_id", list(MissionID))
    def test_runs_all_missions(self, mission_id):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=mission_id,
            difficulty="easy",
            planner_id="astar",
            policy_id="greedy",
            seed=42,
        )
        assert isinstance(result, MissionResult)
        assert result.mission_id == mission_id.value
        assert result.step_count > 0

    @pytest.mark.parametrize("policy_id", ["greedy", "lookahead"])
    def test_both_policies(self, policy_id):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=MissionID.CIVIL_PROTECTION,
            difficulty="easy",
            planner_id="astar",
            policy_id=policy_id,
            seed=42,
        )
        assert result.step_count > 0

    def test_metrics_complete(self):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=MissionID.CIVIL_PROTECTION,
            difficulty="easy",
            planner_id="astar",
            seed=0,
        )
        for key in COMMON_METRICS:
            assert key in result.metrics, f"Missing metric: {key}"

    def test_mission_score_range(self):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=MissionID.CIVIL_PROTECTION,
            difficulty="easy",
            planner_id="astar",
            seed=0,
        )
        assert 0.0 <= result.metrics["mission_score"] <= 1.0

    def test_products_generated(self):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=MissionID.CIVIL_PROTECTION,
            difficulty="easy",
            planner_id="astar",
            seed=0,
        )
        # Should have at least some products if tasks were completed
        if result.metrics["task_completion_rate"] > 0:
            assert len(result.products) > 0

    def test_deterministic(self):
        h, nf = _make_open_map(64)
        r1 = plan_mission(start=(8, 8), heightmap=h, no_fly=nf,
                          mission_id=MissionID.CIVIL_PROTECTION,
                          difficulty="easy", planner_id="astar", seed=42)
        r2 = plan_mission(start=(8, 8), heightmap=h, no_fly=nf,
                          mission_id=MissionID.CIVIL_PROTECTION,
                          difficulty="easy", planner_id="astar", seed=42)
        assert r1.step_count == r2.step_count
        assert r1.metrics["mission_score"] == r2.metrics["mission_score"]


# ═══════════════════════════════════════════════════════════════════════
#  Export helpers
# ═══════════════════════════════════════════════════════════════════════

class TestExport:
    def test_csv_export(self):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=MissionID.CIVIL_PROTECTION,
            difficulty="easy",
            planner_id="astar",
            seed=0,
        )
        with tempfile.TemporaryDirectory() as td:
            paths = export_products_csv(result, Path(td))
            for p in paths:
                assert p.exists()
                content = p.read_text()
                assert len(content) > 0

    def test_json_export(self):
        h, nf = _make_open_map(64)
        result = plan_mission(
            start=(8, 8),
            heightmap=h,
            no_fly=nf,
            mission_id=MissionID.CIVIL_PROTECTION,
            difficulty="easy",
            planner_id="astar",
            seed=0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = export_episode_json(result, Path(td) / "episode.json")
            assert path.exists()
            import json
            data = json.loads(path.read_text())
            assert "metrics" in data
            assert "tasks" in data
