"""Theoretical validation report generator.

Auto-generates paper_theoretical_validation.json containing:
- Hypothesis → metric mapping
- Effect sizes (Cohen's d)
- Confidence intervals
- Falsifiability statement per claim
- No claim without metric linkage
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _ci95(values: list[float]) -> tuple[float, float, float]:
    """Return (mean, std, ci_half_width)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    ci = float(1.96 * std / math.sqrt(max(arr.size, 1)))
    return mean, std, ci


def _cohen_d(x: list[float], y: list[float]) -> float:
    """Cohen's d effect size between two groups."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size == 0 or ya.size == 0:
        return float("nan")
    mx, my = float(np.mean(xa)), float(np.mean(ya))
    vx = float(np.var(xa, ddof=1)) if xa.size > 1 else 0.0
    vy = float(np.var(ya, ddof=1)) if ya.size > 1 else 0.0
    pooled = math.sqrt(max(((xa.size - 1) * vx + (ya.size - 1) * vy)
                            / max(xa.size + ya.size - 2, 1), 1e-12))
    return float((mx - my) / pooled)


# ─── Hypotheses ───

HYPOTHESES = [
    {
        "id": "H1",
        "claim": "Dynamic replanning planners achieve higher success rates than static planners in dynamic scenarios.",
        "metric": "success_rate",
        "comparison": ("adaptive_group", "static_group"),
        "direction": "greater",
        "falsifiable": "Falsified if static planners achieve >= 80% success rate in dynamic scenarios.",
    },
    {
        "id": "H2",
        "claim": "Risk-aware planners produce lower cumulative risk exposure than non-risk-aware planners.",
        "metric": "risk_exposure_integral",
        "comparison": ("risk_aware_group", "non_risk_aware_group"),
        "direction": "less",
        "falsifiable": "Falsified if risk-aware planners show >= risk exposure of non-risk-aware planners.",
    },
    {
        "id": "H3",
        "claim": "Behaviorally distinct adaptive planners produce statistically different replan trigger distributions.",
        "metric": "total_replans",
        "comparison": ("incremental_dstar_lite", "grid_mppi"),
        "direction": "different",
        "falsifiable": "Falsified if Cohen's d < 0.5 between any two adaptive planner pairs.",
    },
    {
        "id": "H4",
        "claim": "Path length spread meaningfully differentiates planner quality.",
        "metric": "path_length",
        "comparison": ("best_planner", "worst_planner"),
        "direction": "greater",
        "falsifiable": "Falsified if normalized path length range < 0.1 across planners.",
    },
    {
        "id": "H5",
        "claim": "Guardrail is necessary: disabling it increases infeasibility collapse rate.",
        "metric": "feasible_after_guardrail",
        "comparison": ("default", "no_guardrail"),
        "direction": "greater",
        "falsifiable": "Falsified if no_guardrail variant shows <= 5% increase in infeasibility.",
    },
    {
        "id": "H6",
        "claim": "Fire-traffic causal feedback creates measurable interaction effects.",
        "metric": "fire_traffic_feedback_rate",
        "comparison": ("with_interactions", "no_interactions"),
        "direction": "greater",
        "falsifiable": "Falsified if fire_traffic_feedback_rate = 0 in all wildfire scenarios.",
    },
    {
        "id": "H7",
        "claim": "Incremental D* Lite requires fewer total replans than periodic-replan A* on dynamic scenarios.",
        "metric": "total_replans",
        "comparison": ("incremental_dstar_lite", "periodic_replan"),
        "direction": "less",
        "falsifiable": "Falsified if incremental_dstar_lite shows >= replans as periodic_replan.",
    },
]


def generate_validation_report(
    episode_results: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Generate theoretical validation report from episode results.

    Args:
        episode_results: List of run_dynamic_episode / run_planner_once dicts
        output_path: Where to save the JSON report

    Returns:
        The validation report dict
    """
    # Group by planner
    by_planner: dict[str, list[dict]] = {}
    for r in episode_results:
        pid = r.get("planner", r.get("planner_id", "unknown"))
        by_planner.setdefault(pid, []).append(r)

    # Define planner groups — MUST use honest (paper) planner IDs
    static_planners = {"astar", "theta_star"}
    adaptive_planners = {"periodic_replan", "aggressive_replan", "incremental_dstar_lite", "grid_mppi"}
    risk_aware_planners = {"grid_mppi"}

    static_group = [r for pid in static_planners for r in by_planner.get(pid, [])]
    adaptive_group = [r for pid in adaptive_planners for r in by_planner.get(pid, [])]
    risk_aware_group = [r for pid in risk_aware_planners for r in by_planner.get(pid, [])]
    non_risk_group = [r for pid in (adaptive_planners - risk_aware_planners)
                      for r in by_planner.get(pid, [])]

    report: dict[str, Any] = {
        "report_version": "2.0",
        "num_episodes": len(episode_results),
        "num_planners": len(by_planner),
        "planners": sorted(by_planner.keys()),
        "hypotheses": [],
    }

    for hyp in HYPOTHESES:
        result: dict[str, Any] = {
            "id": hyp["id"],
            "claim": hyp["claim"],
            "metric": hyp["metric"],
            "falsifiable": hyp["falsifiable"],
        }

        # Extract metric values for comparison groups
        metric_key = hyp["metric"]
        group_a_name, group_b_name = hyp["comparison"]

        values_a = _extract_metric(
            group_a_name, metric_key, by_planner,
            static_group, adaptive_group, risk_aware_group, non_risk_group
        )
        values_b = _extract_metric(
            group_b_name, metric_key, by_planner,
            static_group, adaptive_group, risk_aware_group, non_risk_group
        )

        if values_a and values_b:
            mean_a, std_a, ci_a = _ci95(values_a)
            mean_b, std_b, ci_b = _ci95(values_b)
            d = _cohen_d(values_a, values_b)

            result["group_a"] = {
                "name": group_a_name,
                "n": len(values_a),
                "mean": round(mean_a, 4),
                "std": round(std_a, 4),
                "ci_95": [round(mean_a - ci_a, 4), round(mean_a + ci_a, 4)],
            }
            result["group_b"] = {
                "name": group_b_name,
                "n": len(values_b),
                "mean": round(mean_b, 4),
                "std": round(std_b, 4),
                "ci_95": [round(mean_b - ci_b, 4), round(mean_b + ci_b, 4)],
            }
            result["cohen_d"] = round(d, 4)
            result["effect_size_category"] = (
                "large" if abs(d) >= 0.8 else
                "medium" if abs(d) >= 0.5 else
                "small" if abs(d) >= 0.2 else
                "negligible"
            )
            result["supported"] = _check_direction(hyp["direction"], mean_a, mean_b, d)
        else:
            result["group_a"] = {"name": group_a_name, "n": len(values_a)}
            result["group_b"] = {"name": group_b_name, "n": len(values_b)}
            result["cohen_d"] = float("nan")
            result["effect_size_category"] = "insufficient_data"
            result["supported"] = False

        report["hypotheses"].append(result)

    # Summary
    supported = sum(1 for h in report["hypotheses"] if h.get("supported", False))
    total = len(report["hypotheses"])
    report["summary"] = {
        "hypotheses_supported": supported,
        "hypotheses_total": total,
        "support_rate": round(supported / max(total, 1), 3),
        "best_paper_ready": supported == total,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)

    return report


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _extract_metric(
    group_name: str,
    metric_key: str,
    by_planner: dict[str, list[dict]],
    static_group: list[dict],
    adaptive_group: list[dict],
    risk_aware_group: list[dict],
    non_risk_group: list[dict],
) -> list[float]:
    """Extract metric values for a named group."""
    if group_name == "static_group":
        group = static_group
    elif group_name == "adaptive_group":
        group = adaptive_group
    elif group_name == "risk_aware_group":
        group = risk_aware_group
    elif group_name == "non_risk_aware_group":
        group = non_risk_group
    elif group_name in by_planner:
        group = by_planner[group_name]
    elif group_name == "best_planner":
        # Find planner with highest success rate
        rates = {p: np.mean([1.0 if r.get("success") else 0.0 for r in rs])
                 for p, rs in by_planner.items()}
        best = max(rates, key=lambda k: rates[k]) if rates else None
        group = by_planner.get(best, []) if best else []
    elif group_name == "worst_planner":
        rates = {p: np.mean([1.0 if r.get("success") else 0.0 for r in rs])
                 for p, rs in by_planner.items()}
        worst = min(rates, key=lambda k: rates[k]) if rates else None
        group = by_planner.get(worst, []) if worst else []
    elif group_name == "with_interactions":
        group = [r for r in adaptive_group if r.get("protocol_variant", "default") == "default"]
    elif group_name == "no_interactions":
        group = [r for r in adaptive_group if r.get("protocol_variant") == "no_interactions"]
    elif group_name == "default":
        group = [r for r in adaptive_group if r.get("protocol_variant", "default") == "default"]
    elif group_name == "no_guardrail":
        group = [r for r in adaptive_group if r.get("protocol_variant") == "no_guardrail"]
    else:
        group = []

    values = []
    for r in group:
        v = r.get(metric_key)
        if v is not None:
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                pass
    # Also check nested: success -> 1.0/0.0
    if metric_key == "success_rate" and not values:
        values = [1.0 if r.get("success") else 0.0 for r in group]

    return values


def _check_direction(direction: str, mean_a: float, mean_b: float, d: float) -> bool:
    """Check if the effect is in the expected direction with meaningful size."""
    if math.isnan(d):
        return False
    if direction == "greater":
        return mean_a > mean_b and abs(d) >= 0.2
    if direction == "less":
        return mean_a < mean_b and abs(d) >= 0.2
    if direction == "different":
        return abs(d) >= 0.5
    return False
