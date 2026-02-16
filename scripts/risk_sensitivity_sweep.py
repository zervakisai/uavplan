#!/usr/bin/env python3
"""Risk weight sensitivity sweep: validates ranking stability under perturbation.

Sweeps ±20% on each risk weight (fire, traffic, smoke) and measures:
- Planner ranking stability via Kendall tau correlation
- Per-weight sensitivity magnitude
- Ranking invariance across perturbation levels

Usage:
    python scripts/risk_sensitivity_sweep.py --seeds 5 --output-root results/sensitivity
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uavbench.cli.benchmark import run_dynamic_episode
from uavbench.scenarios.registry import list_scenarios_by_track


def _kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Compute Kendall tau rank correlation between two rankings."""
    n = len(ranking_a)
    if n < 2:
        return 1.0
    # Build index maps
    idx_a = {name: i for i, name in enumerate(ranking_a)}
    idx_b = {name: i for i, name in enumerate(ranking_b)}
    common = [name for name in ranking_a if name in idx_b]
    if len(common) < 2:
        return 1.0

    concordant = 0
    discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a_order = idx_a[common[i]] - idx_a[common[j]]
            b_order = idx_b[common[i]] - idx_b[common[j]]
            if a_order * b_order > 0:
                concordant += 1
            elif a_order * b_order < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return float(concordant - discordant) / float(total)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_sweep(
    scenarios: list[str],
    planners: list[str],
    seeds: list[int],
    output_root: Path,
    perturbation: float = 0.20,
) -> None:
    """Run the risk weight sensitivity sweep."""
    output_root.mkdir(parents=True, exist_ok=True)

    # Default weights
    base_weights = {
        "risk_weight_population": 0.55,
        "risk_weight_adversarial": 0.30,
        "risk_weight_smoke": 0.15,
    }

    # Generate perturbation configs: base, and ±perturbation for each weight
    configs: list[dict[str, Any]] = [
        {"label": "baseline", **base_weights},
    ]
    for key in base_weights:
        for direction in [-perturbation, +perturbation]:
            perturbed = dict(base_weights)
            perturbed[key] = float(np.clip(base_weights[key] + direction, 0.01, 1.0))
            # Renormalize
            total = sum(perturbed.values())
            perturbed = {k: v / total for k, v in perturbed.items()}
            sign = "minus" if direction < 0 else "plus"
            configs.append({
                "label": f"{key}_{sign}{int(abs(direction)*100)}pct",
                **perturbed,
            })

    print(f"Sweep: {len(configs)} weight configs × {len(scenarios)} scenarios × "
          f"{len(planners)} planners × {len(seeds)} seeds")

    all_rows: list[dict[str, Any]] = []
    rankings_per_config: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for cfg_idx, weight_cfg in enumerate(configs):
        label = weight_cfg["label"]
        overrides = {
            "risk_weight_population": weight_cfg["risk_weight_population"],
            "risk_weight_adversarial": weight_cfg["risk_weight_adversarial"],
            "risk_weight_smoke": weight_cfg["risk_weight_smoke"],
        }
        print(f"\n[{cfg_idx+1}/{len(configs)}] Config: {label}")
        print(f"  Weights: pop={overrides['risk_weight_population']:.3f} "
              f"adv={overrides['risk_weight_adversarial']:.3f} "
              f"smoke={overrides['risk_weight_smoke']:.3f}")

        for scenario_id in scenarios:
            planner_scores: dict[str, list[float]] = defaultdict(list)

            for planner_id in planners:
                for seed in seeds:
                    try:
                        res = run_dynamic_episode(
                            scenario_id,
                            planner_id,
                            seed=seed,
                            protocol_variant="default",
                            config_overrides=overrides,
                        )
                        # Score: success bonus - risk exposure - path length penalty
                        success = float(res.get("success", False))
                        risk = float(res.get("risk_exposure_integral", 0.0))
                        path_len = float(res.get("path_length", 0))
                        score = 100.0 * success - risk - 0.1 * path_len

                        planner_scores[planner_id].append(score)

                        all_rows.append({
                            "weight_config": label,
                            "scenario": scenario_id,
                            "planner": planner_id,
                            "seed": seed,
                            "success": success,
                            "risk_exposure": risk,
                            "path_length": path_len,
                            "score": round(score, 4),
                            "w_pop": round(overrides["risk_weight_population"], 4),
                            "w_adv": round(overrides["risk_weight_adversarial"], 4),
                            "w_smoke": round(overrides["risk_weight_smoke"], 4),
                        })
                    except Exception as e:
                        print(f"    [ERROR] {scenario_id}/{planner_id}/seed={seed}: {e}")

            # Compute ranking for this scenario under this weight config
            avg_scores = {p: float(np.mean(s)) for p, s in planner_scores.items() if s}
            ranking = sorted(avg_scores.keys(), key=lambda p: avg_scores[p], reverse=True)
            for p in ranking:
                rankings_per_config[label][p] = rankings_per_config[label].get(p, [])
                rankings_per_config[label][p].append(avg_scores.get(p, 0.0))

    # Save raw results
    _write_csv(all_rows, output_root / "risk_sensitivity_raw.csv")
    print(f"\n✓ Raw results: {output_root / 'risk_sensitivity_raw.csv'}")

    # Compute ranking stability (Kendall tau vs baseline)
    baseline_ranking = sorted(
        rankings_per_config.get("baseline", {}).keys(),
        key=lambda p: float(np.mean(rankings_per_config["baseline"].get(p, [0.0]))),
        reverse=True,
    )

    stability_rows: list[dict[str, Any]] = []
    for label, planner_scores in rankings_per_config.items():
        ranking = sorted(
            planner_scores.keys(),
            key=lambda p: float(np.mean(planner_scores.get(p, [0.0]))),
            reverse=True,
        )
        tau = _kendall_tau(baseline_ranking, ranking)
        stability_rows.append({
            "weight_config": label,
            "kendall_tau": round(tau, 4),
            "ranking": ",".join(ranking),
            "baseline_ranking": ",".join(baseline_ranking),
            "rank_stable": tau >= 0.7,
        })

    _write_csv(stability_rows, output_root / "planner_ranking_stability.csv")
    print(f"✓ Ranking stability: {output_root / 'planner_ranking_stability.csv'}")

    # Summary
    taus = [r["kendall_tau"] for r in stability_rows]
    mean_tau = float(np.mean(taus))
    min_tau = float(np.min(taus))
    print(f"\n{'='*60}")
    print(f"SENSITIVITY SWEEP SUMMARY")
    print(f"  Mean Kendall τ: {mean_tau:.3f}")
    print(f"  Min  Kendall τ: {min_tau:.3f}")
    print(f"  Ranking stable: {'YES' if min_tau >= 0.7 else 'NO'}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk weight sensitivity sweep")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    parser.add_argument("--output-root", type=Path, default=Path("results/sensitivity"))
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario IDs (default: first 3 dynamic scenarios)",
    )
    parser.add_argument(
        "--planners",
        type=str,
        default="adaptive_astar,dstar_lite,risk_mpc,event_triggered,risk_gradient,stability_aware",
        help="Comma-separated planner IDs",
    )
    args = parser.parse_args()

    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",")]
    else:
        dynamic = list_scenarios_by_track("dynamic")
        scenarios = dynamic[:3] if len(dynamic) >= 3 else dynamic

    planners = [p.strip() for p in args.planners.split(",")]
    seeds = list(range(args.seeds))

    run_sweep(scenarios, planners, seeds, args.output_root)


if __name__ == "__main__":
    main()
