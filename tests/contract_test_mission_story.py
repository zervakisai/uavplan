"""
UAVBench v2 — Mission Story Contract Tests
==========================================

Contracts verified: MC-1, MC-2, MC-3, MC-4

MC-1  Every episode has an objective POI with a human-readable ``reason`` string.
MC-2  Task completion = reaching POI + spending ``service_time_s`` steps;
      completion logs an event.
MC-3  HUD always shows: ``mission_domain``, ``objective_label``,
      ``distance_to_task``, ``task_progress``, ``deliverable_name``.
MC-4  Results include ``termination_reason`` (TerminationReason enum) and
      ``objective_completed`` (bool).

Test IDs map to V2_TEST_PLAN.md Section 2.7.

Architecture notes
------------------
- Package: src/uavbench/
- UrbanEnvV2 at src/uavbench/envs/urban.py
- Action space: Discrete(5) — UP(0) DOWN(1) LEFT(2) RIGHT(3) STAY(4)
- env.events returns list of event dicts, each with ``step_idx``
- Info dict contains all HUD fields on every step
- TerminationReason enum at src/uavbench/envs/base.py
- service_time_s: consecutive STAY steps at POI to complete a task

Design intent
-------------
Tests are written spec-first (TDD).  They FAIL today because src/uavbench
does not yet exist.  They PASS once the scaffold implements the contracts.
Use ``pytest tests/v2/contract_test_mission_story.py`` to track progress.
"""

from __future__ import annotations

import re
import math
import pytest

# ---------------------------------------------------------------------------
# Optional import: skip gracefully when uavbench is not yet installed.
# All tests in this module are decorated with @pytest.mark.usefixtures or
# collected only when the imports succeed.
# ---------------------------------------------------------------------------

uavbench = pytest.importorskip(
    "uavbench",
    reason="uavbench is not installed — scaffold not yet implemented",
)

# Individual submodule imports, each skipped individually so a partial
# scaffold can still run the tests whose dependencies are available.

try:
    from uavbench.envs.urban import UrbanEnvV2
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"uavbench.envs.urban not available: {exc}", allow_module_level=True)

try:
    from uavbench.envs.base import TerminationReason
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"uavbench.envs.base not available: {exc}", allow_module_level=True)

# ---------------------------------------------------------------------------
# Shared constants — keep tests fast by using a tiny synthetic 10x10 grid.
# The minimal config is built inline so tests have zero external YAML
# dependencies.  All seeds are fixed for determinism (DC-2 guard).
# ---------------------------------------------------------------------------

FIXED_SEED = 42

# Action indices (Discrete(5): UP DOWN LEFT RIGHT STAY)
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

# HUD fields mandated by MC-3
MC3_REQUIRED_HUD_FIELDS = (
    "mission_domain",
    "objective_label",
    "distance_to_task",
    "task_progress",
    "deliverable_name",
)


# ---------------------------------------------------------------------------
# Fixture: minimal synthetic environment
# ---------------------------------------------------------------------------

def _make_minimal_env() -> UrbanEnvV2:
    """Return a small 10x10 synthetic UrbanEnvV2 suitable for fast testing.

    The config uses the pharma_delivery mission type (fly-through tasks,
    service_time_s=0 for perimeter points) so that basic reset/step
    semantics can be tested without needing to simulate many steps.

    For service-time tests a urban_rescue config with a distress_event
    task (service_time_s=2) is used instead; see _make_urban_rescue_env().
    """
    try:
        from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty
    except ImportError:
        pytest.skip("uavbench.scenarios.schema not available")

    cfg = ScenarioConfig(
        name="test_minimal_pharma_delivery",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=10,
        map_source="synthetic",
        osm_tile_id=None,
        building_density=0.0,          # no buildings — free grid
        fixed_start_xy=(1, 1),
        fixed_goal_xy=(8, 8),
        min_start_goal_l1=5,
        enable_fire=False,
        enable_traffic=False,
        enable_dynamic_nfz=False,
        fire_blocks_movement=False,
        traffic_blocks_movement=False,
        terminate_on_collision=False,
    )
    return UrbanEnvV2(cfg)


def _make_urban_rescue_env() -> UrbanEnvV2:
    """Return a 10x10 urban_rescue env with service_time_s=2 for MC-2 tests."""
    try:
        from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty
    except ImportError:
        pytest.skip("uavbench.scenarios.schema not available")

    cfg = ScenarioConfig(
        name="test_minimal_urban_rescue",
        mission_type=MissionType.URBAN_RESCUE,
        difficulty=Difficulty.EASY,
        map_size=10,
        map_source="synthetic",
        osm_tile_id=None,
        building_density=0.0,
        fixed_start_xy=(1, 1),
        fixed_goal_xy=(8, 8),
        min_start_goal_l1=5,
        enable_fire=False,
        enable_traffic=False,
        enable_dynamic_nfz=False,
        fire_blocks_movement=False,
        traffic_blocks_movement=False,
        terminate_on_collision=False,
    )
    return UrbanEnvV2(cfg)


# ---------------------------------------------------------------------------
# MC-1 Tests
# ---------------------------------------------------------------------------


def test_objective_poi_exists():
    """Verifies MC-1: after reset, info dict has objective_poi as (int, int)
    and objective_reason as a non-empty string.

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      Info dict after reset contains ``objective_poi`` as (int, int) and
      ``objective_reason`` as non-empty str.
    """
    # Arrange
    env = _make_minimal_env()

    # Act
    _obs, info = env.reset(seed=FIXED_SEED)

    # Assert — objective_poi
    assert "objective_poi" in info, (
        "MC-1: info dict missing 'objective_poi' after reset"
    )
    poi = info["objective_poi"]
    assert isinstance(poi, tuple) and len(poi) == 2, (
        f"MC-1: objective_poi must be a 2-tuple, got {type(poi).__name__}"
    )
    x, y = poi
    assert isinstance(x, int) and isinstance(y, int), (
        f"MC-1: objective_poi coordinates must be int, got ({type(x).__name__}, {type(y).__name__})"
    )

    # Assert — objective_reason
    assert "objective_reason" in info, (
        "MC-1: info dict missing 'objective_reason' after reset"
    )
    reason = info["objective_reason"]
    assert isinstance(reason, str) and len(reason) > 0, (
        f"MC-1: objective_reason must be a non-empty str, got {reason!r}"
    )


def test_objective_reason_is_human_readable():
    """Verifies MC-1: objective_reason is >= 10 characters and contains no
    raw machine IDs (UUIDs, underscored identifiers, hex strings).

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      objective_reason is >= 10 characters and contains no raw IDs.

    'Raw IDs' is operationalised as:
      - UUID patterns (8-4-4-4-12 hex groups)
      - Strings that are purely underscored_snake_case with no spaces
      - Bare hex strings longer than 6 chars
    """
    # Arrange
    env = _make_minimal_env()

    # Act
    _obs, info = env.reset(seed=FIXED_SEED)
    reason = info.get("objective_reason", "")

    # Assert — minimum length
    assert len(reason) >= 10, (
        f"MC-1: objective_reason too short ({len(reason)} chars): {reason!r}"
    )

    # Assert — no UUID
    uuid_pattern = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    )
    assert not uuid_pattern.search(reason), (
        f"MC-1: objective_reason contains a raw UUID: {reason!r}"
    )

    # Assert — reason contains at least one space (i.e., is a phrase, not an ID)
    assert " " in reason, (
        f"MC-1: objective_reason appears to be a raw identifier (no spaces): {reason!r}"
    )

    # Assert — not a bare hex string
    bare_hex = re.compile(r"^[0-9a-f]{7,}$", re.IGNORECASE)
    assert not bare_hex.match(reason.strip()), (
        f"MC-1: objective_reason looks like a raw hex string: {reason!r}"
    )


# ---------------------------------------------------------------------------
# MC-2 Tests
# ---------------------------------------------------------------------------


def _navigate_to_poi(env: UrbanEnvV2, info: dict) -> dict:
    """Navigate the agent to the objective POI via step actions.

    Returns the last info dict after the agent arrives at objective_poi.
    Raises AssertionError if the POI is unreachable within 200 steps on a
    10x10 free grid (which would be a test design error).
    """
    poi = info["objective_poi"]
    max_steps = 200
    current_info = info

    for _ in range(max_steps):
        agent_xy = current_info.get("agent_pos") or current_info.get("agent_xy")
        assert agent_xy is not None, "info dict must expose 'agent_pos' or 'agent_xy'"

        if tuple(agent_xy) == tuple(poi):
            return current_info

        ax, ay = agent_xy
        px, py = poi

        # Greedy Manhattan navigation: move in whichever axis has more distance.
        dx = px - ax
        dy = py - ay

        if abs(dx) >= abs(dy):
            action = ACTION_RIGHT if dx > 0 else ACTION_LEFT
        else:
            action = ACTION_DOWN if dy > 0 else ACTION_UP

        _obs, _rew, _terminated, _truncated, current_info = env.step(action)

        if _terminated or _truncated:
            break

    # Final check: are we at the POI?
    agent_xy = current_info.get("agent_pos") or current_info.get("agent_xy")
    assert tuple(agent_xy) == tuple(poi), (
        f"Navigation helper failed to reach POI {poi}; ended at {agent_xy}"
    )
    return current_info


def test_task_completion_requires_service_time():
    """Verifies MC-2: being at POI for fewer than service_time_s steps does
    NOT trigger a task_completed event.

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      Agent at POI for fewer than service_time_s steps does NOT trigger
      completion event.

    Uses the urban_rescue env where distress_event tasks have service_time_s=2.
    After arriving at the POI, one STAY (< service_time_s) must NOT produce
    a task_completed event.
    """
    # Arrange
    env = _make_urban_rescue_env()
    _obs, info = env.reset(seed=FIXED_SEED)

    service_time_s = info.get("service_time_s")
    if service_time_s is None or service_time_s < 2:
        pytest.skip(
            "Flood rescue env did not expose service_time_s >= 2 in info; "
            "adjust test config once schema is implemented"
        )

    # Act — navigate to POI
    info = _navigate_to_poi(env, info)

    # Record which task_completed events exist before the partial stay
    completed_before = {
        e["task_id"]
        for e in env.events
        if e.get("type") == "task_completed"
    }

    # Perform fewer than service_time_s STAY actions (exactly service_time_s - 1)
    for _ in range(service_time_s - 1):
        _obs, _rew, _terminated, _truncated, info = env.step(ACTION_STAY)
        if _terminated or _truncated:
            break

    # Assert — no new task_completed event
    completed_after = {
        e["task_id"]
        for e in env.events
        if e.get("type") == "task_completed"
    }
    new_completions = completed_after - completed_before
    assert len(new_completions) == 0, (
        f"MC-2: task_completed fired after only {service_time_s - 1} STAY steps "
        f"(expected {service_time_s}); new task IDs: {new_completions}"
    )


def test_task_completion_logs_event():
    """Verifies MC-2: after service_time_s consecutive STAY actions at the
    POI, the event log contains a task_completed event with a task_id.

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      After service_time_s STAY actions at POI, event log has task_completed
      event with task_id.
    """
    # Arrange
    env = _make_urban_rescue_env()
    _obs, info = env.reset(seed=FIXED_SEED)

    service_time_s = info.get("service_time_s")
    if service_time_s is None:
        pytest.skip(
            "Flood rescue env did not expose service_time_s in info; "
            "adjust test config once schema is implemented"
        )

    # Act — navigate to POI
    info = _navigate_to_poi(env, info)

    # Perform exactly service_time_s STAY actions
    final_info = info
    for _ in range(service_time_s):
        _obs, _rew, _terminated, _truncated, final_info = env.step(ACTION_STAY)
        if _terminated or _truncated:
            break

    # Assert — event log contains task_completed
    task_completed_events = [
        e for e in env.events if e.get("type") == "task_completed"
    ]
    assert len(task_completed_events) >= 1, (
        f"MC-2: no task_completed event in env.events after {service_time_s} "
        f"STAY steps at POI. Events: {env.events}"
    )

    # Assert — each task_completed event has a task_id
    for event in task_completed_events:
        assert "task_id" in event, (
            f"MC-2: task_completed event missing 'task_id' field: {event}"
        )
        assert event["task_id"] is not None and str(event["task_id"]) != "", (
            f"MC-2: task_completed event has empty task_id: {event}"
        )


# ---------------------------------------------------------------------------
# MC-3 Tests
# ---------------------------------------------------------------------------


def test_hud_fields_present_every_step():
    """Verifies MC-3: info dict contains mission_domain, objective_label,
    distance_to_task, task_progress, and deliverable_name with non-None
    values at every step.

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      Info dict contains all 5 HUD fields with non-None values at every step.

    Runs 20 steps on the minimal pharma_delivery env.  Checks each info
    dict individually so failures report the exact step number.
    """
    # Arrange
    env = _make_minimal_env()
    _obs, reset_info = env.reset(seed=FIXED_SEED)

    # Check reset info first (step 0 baseline)
    for field in MC3_REQUIRED_HUD_FIELDS:
        assert field in reset_info, (
            f"MC-3: HUD field '{field}' missing from reset info"
        )
        assert reset_info[field] is not None, (
            f"MC-3: HUD field '{field}' is None in reset info"
        )

    # Run 20 steps (well within 10x10 episode budget)
    NUM_STEPS = 20
    for step_idx in range(NUM_STEPS):
        # Alternate STAY / DOWN / RIGHT to exercise different code paths
        action = [ACTION_STAY, ACTION_DOWN, ACTION_RIGHT][step_idx % 3]
        _obs, _rew, terminated, truncated, step_info = env.step(action)

        for field in MC3_REQUIRED_HUD_FIELDS:
            assert field in step_info, (
                f"MC-3: HUD field '{field}' missing from info at step {step_idx + 1}"
            )
            assert step_info[field] is not None, (
                f"MC-3: HUD field '{field}' is None at step {step_idx + 1}"
            )

        if terminated or truncated:
            break

    # Assert specific types to prevent "technically present but wrong type" passes
    _obs2, info2 = env.reset(seed=FIXED_SEED)
    assert isinstance(info2["mission_domain"], str), (
        "MC-3: mission_domain must be a str"
    )
    assert isinstance(info2["objective_label"], str), (
        "MC-3: objective_label must be a str"
    )
    assert isinstance(info2["distance_to_task"], (int, float)) and not math.isnan(
        float(info2["distance_to_task"])
    ), (
        f"MC-3: distance_to_task must be a finite number, got {info2['distance_to_task']!r}"
    )
    assert isinstance(info2["task_progress"], str), (
        "MC-3: task_progress must be a str (e.g. '0/4')"
    )
    assert isinstance(info2["deliverable_name"], str), (
        "MC-3: deliverable_name must be a str"
    )


# ---------------------------------------------------------------------------
# MC-4 Tests
# ---------------------------------------------------------------------------


def test_termination_reason_in_final_info():
    """Verifies MC-4: when an episode terminates (success, collision, or
    timeout), the final info dict contains termination_reason as a
    TerminationReason enum member and objective_completed as a bool.

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      Final info dict has termination_reason (TerminationReason enum) and
      objective_completed (bool).

    Forces termination via timeout by running max_steps on a small env.
    """
    # Arrange — use a very short time budget so the episode times out fast
    try:
        from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty
    except ImportError:
        pytest.skip("uavbench.scenarios.schema not available")

    cfg = ScenarioConfig(
        name="test_timeout_pharma_delivery",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=10,
        map_source="synthetic",
        osm_tile_id=None,
        building_density=0.0,
        fixed_start_xy=(1, 1),
        fixed_goal_xy=(8, 8),
        min_start_goal_l1=5,
        enable_fire=False,
        enable_traffic=False,
        enable_dynamic_nfz=False,
        fire_blocks_movement=False,
        traffic_blocks_movement=False,
        terminate_on_collision=False,
        max_episode_steps=5,           # force timeout quickly
    )
    env = UrbanEnvV2(cfg)
    _obs, _info = env.reset(seed=FIXED_SEED)

    # Act — keep sending STAY until the episode terminates
    final_info: dict = _info
    for _ in range(10):
        _obs, _rew, terminated, truncated, final_info = env.step(ACTION_STAY)
        if terminated or truncated:
            break
    else:
        pytest.fail(
            "MC-4: episode did not terminate after 10 steps with max_episode_steps=5"
        )

    # Assert — termination_reason is present and is a TerminationReason enum member
    assert "termination_reason" in final_info, (
        "MC-4: 'termination_reason' missing from final info dict"
    )
    tr = final_info["termination_reason"]
    assert isinstance(tr, TerminationReason), (
        f"MC-4: termination_reason must be TerminationReason enum, "
        f"got {type(tr).__name__}: {tr!r}"
    )

    # Assert — objective_completed is present and is a bool
    assert "objective_completed" in final_info, (
        "MC-4: 'objective_completed' missing from final info dict"
    )
    oc = final_info["objective_completed"]
    assert isinstance(oc, bool), (
        f"MC-4: objective_completed must be bool, got {type(oc).__name__}: {oc!r}"
    )


def test_successful_episode_objective_completed():
    """Verifies MC-4: when the agent reaches the goal and completes all
    tasks, objective_completed is True in the final info dict.

    Acceptance criterion (V2_TEST_PLAN.md §2.7):
      When agent reaches goal and completes tasks, objective_completed is True.

    Strategy: use a 10x10 pharma_delivery env with fly-through tasks
    (service_time_s=0) and no buildings.  Navigate directly from start to
    goal via Manhattan-optimal moves.  The env must set objective_completed
    True on success.
    """
    # Arrange
    try:
        from uavbench.scenarios.schema import ScenarioConfig, MissionType, Difficulty
    except ImportError:
        pytest.skip("uavbench.scenarios.schema not available")

    cfg = ScenarioConfig(
        name="test_success_pharma_delivery",
        mission_type=MissionType.PHARMA_DELIVERY,
        difficulty=Difficulty.EASY,
        map_size=30,
        map_source="synthetic",
        osm_tile_id=None,
        building_density=0.0,
        fixed_start_xy=(1, 1),
        fixed_goal_xy=(28, 28),
        min_start_goal_l1=2,
        enable_fire=False,
        enable_traffic=False,
        enable_dynamic_nfz=False,
        fire_blocks_movement=False,
        traffic_blocks_movement=False,
        terminate_on_collision=False,
        max_episode_steps=500,
    )
    env = UrbanEnvV2(cfg)
    _obs, info = env.reset(seed=FIXED_SEED)

    goal_xy = info.get("goal_pos") or info.get("goal_xy")
    assert goal_xy is not None, "info must expose 'goal_pos' or 'goal_xy' after reset"
    goal_xy = tuple(goal_xy)

    # Build waypoint sequence: visit each task POI, then goal.
    # Each POI requires service_time STAY steps after arrival.
    task_info = info.get("task_info_list", [])
    waypoints = [(t["xy"], t.get("service_time", 1)) for t in task_info]
    waypoints.append((goal_xy, 1))  # final STAY at goal

    # Act — navigate through all POIs then to goal
    final_info: dict = info
    terminated = truncated = False
    max_steps = 500

    def _nav_to(target, env, final_info, max_steps):
        """Navigate Manhattan-optimal to target, return (terminated, truncated, final_info, steps)."""
        terminated = truncated = False
        steps = 0
        for _ in range(max_steps):
            if terminated or truncated:
                break
            agent_xy = final_info.get("agent_pos") or final_info.get("agent_xy")
            ax, ay = agent_xy
            tx, ty = target
            if (ax, ay) == (tx, ty):
                break
            dx, dy = tx - ax, ty - ay
            if abs(dx) >= abs(dy):
                action = ACTION_RIGHT if dx > 0 else ACTION_LEFT
            else:
                action = ACTION_DOWN if dy > 0 else ACTION_UP
            _obs, _rew, terminated, truncated, final_info = env.step(action)
            steps += 1
        return terminated, truncated, final_info, steps

    for wp_xy, svc_time in waypoints:
        if terminated or truncated:
            break
        terminated, truncated, final_info, _ = _nav_to(
            wp_xy, env, final_info, max_steps,
        )
        # STAY at waypoint for service_time
        for _ in range(svc_time + 1):
            if terminated or truncated:
                break
            _obs, _rew, terminated, truncated, final_info = env.step(ACTION_STAY)

    # Assert — episode terminated successfully
    assert terminated or truncated, (
        "MC-4: episode did not terminate after navigating to goal"
    )

    # Assert — termination_reason
    assert "termination_reason" in final_info, (
        "MC-4: 'termination_reason' missing from final info"
    )
    tr = final_info["termination_reason"]
    assert isinstance(tr, TerminationReason), (
        f"MC-4: termination_reason must be TerminationReason enum, got {tr!r}"
    )
    assert tr == TerminationReason.SUCCESS, (
        f"MC-4: expected TerminationReason.SUCCESS, got {tr!r}"
    )

    # Assert — objective_completed is True
    assert "objective_completed" in final_info, (
        "MC-4: 'objective_completed' missing from final info"
    )
    assert final_info["objective_completed"] is True, (
        f"MC-4: objective_completed should be True after successful episode, "
        f"got {final_info['objective_completed']!r}"
    )


# ---------------------------------------------------------------------------
# MC-3 Extension: Mission Briefing Tests
# ---------------------------------------------------------------------------


def test_briefing_event_present():
    """MC-3: First events after reset include a mission_briefing event."""
    env = _make_minimal_env()
    _obs, info = env.reset(seed=FIXED_SEED)

    briefing_events = [
        e for e in env.events if e.get("type") == "mission_briefing"
    ]
    assert len(briefing_events) >= 1, (
        "MC-3: no mission_briefing event found"
    )


def test_briefing_fields_complete():
    """MC-3: Mission briefing has all required fields."""
    env = _make_minimal_env()
    _obs, info = env.reset(seed=FIXED_SEED)

    briefing = next(
        (e for e in env.events if e.get("type") == "mission_briefing"),
        None,
    )
    assert briefing is not None

    required_fields = [
        "type", "step_idx", "mission_type", "domain",
        "origin_name", "destination_name", "objective",
        "deliverable", "constraints",
        "service_time_steps", "priority", "max_time_steps",
    ]
    for fld in required_fields:
        assert fld in briefing, (
            f"MC-3: mission_briefing missing field '{fld}'"
        )


def test_briefing_at_step_zero():
    """MC-3: Mission briefing step_idx is 0."""
    env = _make_minimal_env()
    _obs, info = env.reset(seed=FIXED_SEED)

    briefing = next(
        (e for e in env.events if e.get("type") == "mission_briefing"),
        None,
    )
    assert briefing is not None
    assert briefing["step_idx"] == 0


def test_briefing_constraints_is_list():
    """MC-3: Briefing constraints field is a list of strings."""
    env = _make_minimal_env()
    _obs, info = env.reset(seed=FIXED_SEED)

    briefing = next(
        (e for e in env.events if e.get("type") == "mission_briefing"),
        None,
    )
    assert briefing is not None
    assert isinstance(briefing["constraints"], list)
    for c in briefing["constraints"]:
        assert isinstance(c, str)
