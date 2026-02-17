"""Mission builders: create MissionSpec + injection schedule for each mission.

Each builder is MAP-AGNOSTIC: it receives map dimensions and an RNG, then
places tasks at valid grid positions.  The injection schedule controls
when new tasks / constraints appear.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from uavbench.missions.spec import (
    MissionID,
    MissionSpec,
    TaskSpec,
    DifficultyKnobs,
    ProductType,
)
from uavbench.missions.engine import InjectionEvent
from uavbench.planners.base import GridPos


def _scatter_pois(
    n: int,
    map_size: int,
    rng: np.random.Generator,
    margin: int = 8,
    existing: list[GridPos] | None = None,
    min_sep: int = 5,
) -> list[GridPos]:
    """Place n POIs on a grid with minimum separation."""
    placed: list[GridPos] = list(existing or [])
    attempts = 0
    while len(placed) < (len(existing or []) + n) and attempts < n * 50:
        x = int(rng.integers(margin, map_size - margin))
        y = int(rng.integers(margin, map_size - margin))
        ok = all(abs(x - px) + abs(y - py) >= min_sep for px, py in placed)
        if ok:
            placed.append((x, y))
        attempts += 1
    return placed[len(existing or []):]


# ═══════════════════════════════════════════════════════════════════════
# MISSION 1 — Civil Protection: Wildfire Crisis SA + Evacuation Corridor
# ═══════════════════════════════════════════════════════════════════════

def build_civil_protection(
    difficulty: Literal["easy", "medium", "hard"],
    map_size: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[MissionSpec, list[InjectionEvent]]:
    """Build Mission 1 spec + injection schedule."""
    rng = rng or np.random.default_rng(0)

    knobs = {
        "easy": DifficultyKnobs.easy(num_tasks=4, time_budget=350),
        "medium": DifficultyKnobs.medium(num_tasks=6, time_budget=300),
        "hard": DifficultyKnobs.hard(num_tasks=8, time_budget=250),
    }[difficulty]

    # Place initial POIs: half perimeter-points, half corridor-checkpoints
    n_perim = knobs.num_tasks // 2
    n_corr = knobs.num_tasks - n_perim

    pois = _scatter_pois(n_perim + n_corr, map_size, rng)
    tasks: list[TaskSpec] = []
    for i, xy in enumerate(pois):
        cat = "perimeter_point" if i < n_perim else "corridor_checkpoint"
        weight = 1.2 if cat == "perimeter_point" else 1.0
        tasks.append(TaskSpec(
            task_id=f"cp_{cat}_{i}",
            xy=xy,
            weight=weight,
            time_decay=0.025 if cat == "perimeter_point" else 0.015,
            category=cat,
        ))

    # Injection schedule
    schedule: list[InjectionEvent] = []
    inject_cadence = {"low": 80, "medium": 50, "high": 30}[knobs.injection_rate]
    n_inject = {"low": 1, "medium": 2, "high": 3}[knobs.injection_rate]

    for j in range(n_inject):
        inject_step = inject_cadence * (j + 1)
        if inject_step >= knobs.time_budget:
            break
        new_pois = _scatter_pois(1, map_size, rng, existing=[t.xy for t in tasks])
        if new_pois:
            new_task = TaskSpec(
                task_id=f"cp_injected_{j}",
                xy=new_pois[0],
                weight=1.5,  # injected tasks are urgent
                time_decay=0.04,
                category="perimeter_point",
                injected_at=inject_step,
            )
            schedule.append(InjectionEvent(
                step=inject_step,
                task=new_task,
                description=f"Fire perimeter shift detected — new POI injected",
            ))

    # Add dynamic constraint injections (NFZ, corridor blocks)
    if knobs.dynamics_intensity in ("moderate", "severe"):
        nfz_step = knobs.time_budget // 3
        schedule.append(InjectionEvent(
            step=nfz_step,
            description="Temporary aviation restriction zone (smoke/wind)",
        ))
    if knobs.dynamics_intensity == "severe":
        block_step = knobs.time_budget // 2
        schedule.append(InjectionEvent(
            step=block_step,
            corridor_blocked="corridor_seg_2",
            description="Corridor blocked — traffic congestion spike",
        ))

    spec = MissionSpec(
        mission_id=MissionID.CIVIL_PROTECTION,
        label="Wildfire Crisis SA + Evacuation Corridor Monitoring",
        difficulty=difficulty,
        knobs=knobs,
        initial_tasks=tuple(tasks),
        product_types=(
            ProductType.FIRE_PERIMETER_GEOJSON,
            ProductType.CORRIDOR_STATUS_CSV,
            ProductType.ALERT_TIMELINE_CSV,
        ),
        utility_decay=0.025,
        risk_penalty_lambda=0.35,
        violation_penalty_lambda=1.0,
        description="ΓΓ Πολιτικής Προστασίας — fire-edge perimeter + evacuation corridors",
    )

    return spec, schedule


# ═══════════════════════════════════════════════════════════════════════
# MISSION 2 — Maritime Domain Awareness: Coastal Patrol + Distress
# ═══════════════════════════════════════════════════════════════════════

def build_maritime_domain(
    difficulty: Literal["easy", "medium", "hard"],
    map_size: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[MissionSpec, list[InjectionEvent]]:
    """Build Mission 2 spec + injection schedule."""
    rng = rng or np.random.default_rng(1)

    knobs = {
        "easy": DifficultyKnobs.easy(num_tasks=4, time_budget=400),
        "medium": DifficultyKnobs.medium(num_tasks=6, time_budget=350),
        "hard": DifficultyKnobs.hard(num_tasks=8, time_budget=280),
    }[difficulty]

    # Patrol waypoints along a corridor (roughly circular/loop)
    cx, cy = map_size // 2, map_size // 2
    radius = map_size // 3
    tasks: list[TaskSpec] = []
    for i in range(knobs.num_tasks):
        angle = 2 * math.pi * i / knobs.num_tasks
        x = int(cx + radius * math.cos(angle))
        y = int(cy + radius * math.sin(angle))
        x = max(8, min(map_size - 8, x))
        y = max(8, min(map_size - 8, y))
        tasks.append(TaskSpec(
            task_id=f"md_patrol_{i}",
            xy=(x, y),
            weight=1.0,
            time_decay=0.01,  # patrol is less time-critical
            category="patrol_waypoint",
        ))

    # Distress event injections
    schedule: list[InjectionEvent] = []
    n_events = {"low": 0, "medium": 1, "high": 2}[knobs.injection_rate]
    # Always inject at least 1 event for easy too
    if difficulty == "easy":
        n_events = 1

    for j in range(n_events):
        event_step = 60 + j * 80
        if event_step >= knobs.time_budget:
            break
        event_pois = _scatter_pois(1, map_size, rng, margin=20)
        if event_pois:
            event_task = TaskSpec(
                task_id=f"md_distress_{j}",
                xy=event_pois[0],
                weight=3.0,  # distress = high priority
                time_decay=0.06,  # rapid decay
                category="distress_event",
                injected_at=event_step,
                service_time=2,  # must loiter to localise
            )
            schedule.append(InjectionEvent(
                step=event_step,
                task=event_task,
                description=f"Distress/safety event detected — immediate response required",
            ))

    # Hazard zone injections
    if knobs.dynamics_intensity in ("moderate", "severe"):
        schedule.append(InjectionEvent(
            step=knobs.time_budget // 4,
            description="Weather/hazard alert — temporary high-risk region",
        ))
    if knobs.dynamics_intensity == "severe":
        schedule.append(InjectionEvent(
            step=knobs.time_budget // 2,
            description="Restricted area expansion — safety zone policy update",
        ))

    spec = MissionSpec(
        mission_id=MissionID.MARITIME_DOMAIN,
        label="Coastal Search Corridor Patrol + Distress Event Response",
        difficulty=difficulty,
        knobs=knobs,
        initial_tasks=tuple(tasks),
        product_types=(
            ProductType.CORRIDOR_COVERAGE_CSV,
            ProductType.EVENT_RESPONSE_CSV,
            ProductType.EXPOSURE_REPORT_CSV,
        ),
        utility_decay=0.01,
        risk_penalty_lambda=0.25,
        violation_penalty_lambda=1.2,
        patrol_weight_alpha=0.6 if difficulty == "easy" else 0.5,
        description="ΛΣ-ΕΛΑΚΤ — coastal corridor patrol + distress event injection",
    )

    return spec, schedule


# ═══════════════════════════════════════════════════════════════════════
# MISSION 3 — Critical Infrastructure: Inspection Tour + Dynamic NFZ
# ═══════════════════════════════════════════════════════════════════════

def build_critical_infrastructure(
    difficulty: Literal["easy", "medium", "hard"],
    map_size: int = 256,
    rng: np.random.Generator | None = None,
) -> tuple[MissionSpec, list[InjectionEvent]]:
    """Build Mission 3 spec + injection schedule."""
    rng = rng or np.random.default_rng(2)

    knobs = {
        "easy": DifficultyKnobs.easy(num_tasks=4, time_budget=300),
        "medium": DifficultyKnobs.medium(num_tasks=6, time_budget=260),
        "hard": DifficultyKnobs.hard(num_tasks=8, time_budget=220),
    }[difficulty]

    # Inspection sites — scattered across map
    pois = _scatter_pois(knobs.num_tasks, map_size, rng, margin=15, min_sep=8)
    tasks: list[TaskSpec] = []

    # Time windows tighten with difficulty
    window_slack = {"easy": 120, "medium": 80, "hard": 50}[difficulty]

    for i, xy in enumerate(pois):
        earliest = 10 + i * 15
        latest = earliest + window_slack
        tasks.append(TaskSpec(
            task_id=f"ci_site_{i}",
            xy=xy,
            weight=1.0 + 0.1 * i,  # later sites slightly more valuable
            time_decay=0.03,
            time_window=(earliest, min(latest, knobs.time_budget - 5)),
            service_time=3,  # inspection requires dwelling
            category="inspection_site",
        ))

    # Dynamic restriction injections
    schedule: list[InjectionEvent] = []
    n_restrictions = {"low": 1, "medium": 3, "high": 5}[knobs.injection_rate]

    for j in range(n_restrictions):
        restrict_step = 30 + j * (knobs.time_budget // (n_restrictions + 1))
        if restrict_step >= knobs.time_budget:
            break
        schedule.append(InjectionEvent(
            step=restrict_step,
            description=f"Dynamic restriction #{j+1} — temporary closure / topology change",
        ))

    # Comms degradation pockets (hard only)
    if difficulty == "hard":
        schedule.append(InjectionEvent(
            step=knobs.time_budget // 3,
            description="Degraded comms pocket — delayed map updates",
        ))

    spec = MissionSpec(
        mission_id=MissionID.CRITICAL_INFRASTRUCTURE,
        label="Time-Critical Inspection Tour under Dynamic Restrictions",
        difficulty=difficulty,
        knobs=knobs,
        initial_tasks=tuple(tasks),
        product_types=(
            ProductType.INSPECTION_LOG_CSV,
            ProductType.COMPLIANCE_REPORT_CSV,
            ProductType.RESILIENCE_CURVE_CSV,
        ),
        utility_decay=0.03,
        risk_penalty_lambda=0.30,
        violation_penalty_lambda=1.5,  # stricter for infrastructure
        strict_compliance=difficulty == "hard",  # hard = fail on NFZ breach
        description="ΥΠΕΘΑ/ISR-support — inspection tour + dynamic restrictions",
    )

    return spec, schedule


# ── Builder dispatch ──────────────────────────────────────────────────

MISSION_BUILDERS = {
    MissionID.CIVIL_PROTECTION: build_civil_protection,
    MissionID.MARITIME_DOMAIN: build_maritime_domain,
    MissionID.CRITICAL_INFRASTRUCTURE: build_critical_infrastructure,
}


def build_mission(
    mission_id: MissionID,
    difficulty: Literal["easy", "medium", "hard"],
    map_size: int = 256,
    seed: int = 0,
) -> tuple[MissionSpec, list[InjectionEvent]]:
    """Build a mission spec + injection schedule."""
    rng = np.random.default_rng(seed)
    return MISSION_BUILDERS[mission_id](difficulty, map_size, rng)
