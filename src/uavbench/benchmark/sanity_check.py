"""Sanity check (SC-1 through SC-4).

Post-run analysis that detects benchmark bugs through result patterns.
Applied after all episodes complete to flag implausible rankings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Violation types
# ---------------------------------------------------------------------------


class ViolationType(str, Enum):
    """Types of sanity violations."""

    ADAPTIVE_BEHIND_STATIC = "SC-1"  # static beats all adaptive in fire
    DIFFICULTY_ORDERING = "SC-2"     # medium worse than hard
    DSTAR_BEHIND_ASTAR = "SC-4"     # D*Lite < A* (implementation bug)


class Severity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SanityViolation:
    """A single sanity violation."""

    violation_type: ViolationType
    severity: Severity
    scenario: str
    details: str


@dataclass
class SanityReport:
    """Result of sanity check over all results."""

    violations: list[SanityViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no ERROR-level violations."""
        return not any(v.severity == Severity.ERROR for v in self.violations)


# ---------------------------------------------------------------------------
# Planner classification
# ---------------------------------------------------------------------------

_STATIC_PLANNERS = {"astar"}
_ADAPTIVE_PLANNERS = {"periodic_replan", "aggressive_replan", "dstar_lite", "apf"}

# Scenarios with expanding fire (vs. moving obstacles)
# "pharma" included because pharma_delivery operates in fire zones
_FIRE_SCENARIO_KEYWORDS = {"fire", "pharma"}


def _is_fire_scenario(scenario_id: str) -> bool:
    """Check if scenario involves fire (expanding obstacles)."""
    return any(kw in scenario_id.lower() for kw in _FIRE_SCENARIO_KEYWORDS)


def _get_difficulty(scenario_id: str) -> str | None:
    """Extract difficulty level from scenario_id."""
    for diff in ("easy", "medium", "hard"):
        if diff in scenario_id.lower():
            return diff
    return None


def _get_scenario_base(scenario_id: str) -> str:
    """Strip difficulty suffix to get scenario family."""
    for diff in ("_easy", "_medium", "_hard"):
        if scenario_id.endswith(diff):
            return scenario_id[: -len(diff)]
    return scenario_id


# ---------------------------------------------------------------------------
# Main sanity check
# ---------------------------------------------------------------------------


def run_sanity_check(
    results: list[dict[str, Any]],
) -> SanityReport:
    """Run sanity checks over episode results (SC-1, SC-2, SC-4).

    Each result dict must have:
      - scenario_id: str
      - planner_id: str
      - success: bool (or int 0/1)

    Returns SanityReport with list of violations.
    """
    report = SanityReport()

    # Group by (scenario_id, planner_id) → success_rate
    from collections import defaultdict

    counts: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in results:
        key = (r["scenario_id"], r["planner_id"])
        counts[key].append(bool(r.get("success", False)))

    def success_rate(scenario_id: str, planner_id: str) -> float | None:
        key = (scenario_id, planner_id)
        if key not in counts:
            return None
        successes = counts[key]
        return sum(successes) / len(successes) if successes else 0.0

    # Collect all scenarios
    scenarios = sorted({r["scenario_id"] for r in results})
    planners = sorted({r["planner_id"] for r in results})

    # --- SC-1: Adaptive >= Static in fire scenarios ---
    for scenario in scenarios:
        if not _is_fire_scenario(scenario):
            continue

        best_static = 0.0
        best_static_planner = ""
        for p in planners:
            if p in _STATIC_PLANNERS:
                rate = success_rate(scenario, p)
                if rate is not None and rate > best_static:
                    best_static = rate
                    best_static_planner = p

        best_adaptive = 0.0
        best_adaptive_planner = ""
        for p in planners:
            if p in _ADAPTIVE_PLANNERS:
                rate = success_rate(scenario, p)
                if rate is not None and rate > best_adaptive:
                    best_adaptive = rate
                    best_adaptive_planner = p

        if best_static > best_adaptive and best_static > 0:
            report.violations.append(SanityViolation(
                violation_type=ViolationType.ADAPTIVE_BEHIND_STATIC,
                severity=Severity.ERROR,
                scenario=scenario,
                details=(
                    f"Static planner '{best_static_planner}' ({best_static:.0%}) "
                    f"beats all adaptive planners (best: '{best_adaptive_planner}' "
                    f"{best_adaptive:.0%}) in fire scenario"
                ),
            ))

    # --- SC-2: Difficulty ordering ---
    scenario_families = sorted({_get_scenario_base(s) for s in scenarios})
    for family in scenario_families:
        for p in planners:
            medium_rate = success_rate(f"{family}_medium", p)
            hard_rate = success_rate(f"{family}_hard", p)
            if medium_rate is not None and hard_rate is not None:
                if hard_rate > medium_rate + 0.05:  # 5% tolerance
                    report.violations.append(SanityViolation(
                        violation_type=ViolationType.DIFFICULTY_ORDERING,
                        severity=Severity.WARNING,
                        scenario=family,
                        details=(
                            f"Planner '{p}': hard ({hard_rate:.0%}) > "
                            f"medium ({medium_rate:.0%}) — difficulty ordering violated"
                        ),
                    ))

    # --- SC-4: D*Lite >= A* ---
    for scenario in scenarios:
        astar_rate = success_rate(scenario, "astar")
        dstar_rate = success_rate(scenario, "dstar_lite")
        if astar_rate is not None and dstar_rate is not None:
            if astar_rate > dstar_rate + 0.05:  # 5% tolerance
                report.violations.append(SanityViolation(
                    violation_type=ViolationType.DSTAR_BEHIND_ASTAR,
                    severity=Severity.WARNING,
                    scenario=scenario,
                    details=(
                        f"D*Lite ({dstar_rate:.0%}) < A* ({astar_rate:.0%}) — "
                        f"possible implementation bug"
                    ),
                ))

    return report
