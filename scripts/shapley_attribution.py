#!/usr/bin/env python3
"""Shapley attribution analysis — which hazard layers matter most per planner.

Computes exact Shapley values for N=3 hazard features:
  1. fire      (enable_fire: wildfire spread + fire buffer)
  2. traffic   (enable_traffic: corridor patrol vehicles)
  3. collapse  (enable_collapse: structural collapse → debris)

For each planner, runs episodes with all 2^3=8 feature coalitions.
Baseline = all hazards disabled (static grid only).
Shapley value = marginal contribution of each feature averaged over
all coalition orderings.

Outputs:
  - CSV with per-planner Shapley values
  - Bar chart figure (matplotlib, IEEE 300 DPI)

Usage:
    python scripts/shapley_attribution.py [--seeds 10] [--scenario osm_penteli_pharma_delivery_medium]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = "outputs/paper_figures"

FEATURES = ["fire", "traffic", "collapse"]
N_FEATURES = len(FEATURES)

# Feature → scenario config field name
_FEATURE_CONFIG_KEYS = {
    "fire": "enable_fire",
    "traffic": "enable_traffic",
    "collapse": "enable_collapse",
}

PLANNERS = ["astar", "periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
PLANNER_LABELS = {
    "astar": "A*",
    "periodic_replan": "Periodic",
    "aggressive_replan": "Aggressive",
    "incremental_astar": "Incr. A*",
    "apf": "APF",
}

DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"


# ---------------------------------------------------------------------------
# Coalition evaluation
# ---------------------------------------------------------------------------


def _coalition_key(coalition: tuple[bool, ...]) -> str:
    """Human-readable key for a feature coalition."""
    active = [f for f, on in zip(FEATURES, coalition) if on]
    return "+".join(active) if active else "baseline"


def _evaluate_coalition(
    scenario_id: str,
    planner_id: str,
    coalition: tuple[bool, ...],
    seeds: list[int],
) -> float:
    """Run episodes for a planner with a given feature coalition.

    Uses the real run_episode runner with config_override to ensure
    full mission handling (POI visits, goal stall, energy, etc.).
    """
    from uavbench.benchmark.runner import run_episode
    from uavbench.scenarios.loader import load_scenario

    # Build config with coalition overrides (frozen dataclass)
    config = load_scenario(scenario_id)
    for i, feat in enumerate(FEATURES):
        key = _FEATURE_CONFIG_KEYS[feat]
        object.__setattr__(config, key, coalition[i])

    successes = 0
    total = 0

    for seed in seeds:
        try:
            result = run_episode(
                scenario_id, planner_id, seed, config_override=config
            )
            term = result.metrics.get("termination_reason", "timeout")
            if term == "success":
                successes += 1
            total += 1
        except Exception as e:
            print(f"        [ERROR seed={seed}] {e}")
            total += 1

    return successes / max(total, 1)


def compute_shapley_values(
    scenario_id: str,
    planner_id: str,
    seeds: list[int],
) -> dict[str, float]:
    """Compute exact Shapley values for one planner.

    For N=3 features, iterates over all 2^3=8 coalitions.
    Shapley value for feature i = average marginal contribution across
    all coalitions S ⊆ N\\{i}:
        φ_i = Σ_{S} [|S|!(N-|S|-1)!/N!] * [v(S∪{i}) - v(S)]
    """
    # Pre-compute coalition values
    coalition_cache: dict[tuple[bool, ...], float] = {}

    for bits in range(2 ** N_FEATURES):
        coalition = tuple(bool(bits & (1 << i)) for i in range(N_FEATURES))
        key = _coalition_key(coalition)
        print(f"      Coalition {key:40s}", end="", flush=True)
        val = _evaluate_coalition(scenario_id, planner_id, coalition, seeds)
        coalition_cache[coalition] = val
        print(f"  SR={val:.2f}")

    # Compute Shapley values
    shapley = {}
    n = N_FEATURES

    for i, feat in enumerate(FEATURES):
        phi = 0.0
        others = [j for j in range(n) if j != i]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s_without = [False] * n
                for j in subset:
                    s_without[j] = True
                s_without_tuple = tuple(s_without)

                s_with = list(s_without)
                s_with[i] = True
                s_with_tuple = tuple(s_with)

                marginal = coalition_cache[s_with_tuple] - coalition_cache[s_without_tuple]

                s_size = sum(s_without)
                weight = (
                    math.factorial(s_size) * math.factorial(n - s_size - 1)
                    / math.factorial(n)
                )
                phi += weight * marginal

        shapley[feat] = phi

    return shapley


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_shapley_bar_chart(
    results: dict[str, dict[str, float]],
    output_dir: str,
) -> None:
    """Stacked bar chart: planners on x-axis, Shapley contributions stacked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    planners = list(results.keys())
    n_planners = len(planners)
    x = np.arange(n_planners)

    colors = ["#e74c3c", "#f39c12", "#8e44ad"]
    feature_labels = ["Fire", "Traffic", "Collapse"]

    bottoms_pos = np.zeros(n_planners)
    bottoms_neg = np.zeros(n_planners)

    for i, feat in enumerate(FEATURES):
        values = [results[p][feat] for p in planners]
        pos_vals = [max(v, 0) for v in values]
        neg_vals = [min(v, 0) for v in values]

        ax.bar(x, pos_vals, bottom=bottoms_pos, color=colors[i],
               label=feature_labels[i], edgecolor="white", linewidth=0.5)
        ax.bar(x, neg_vals, bottom=bottoms_neg, color=colors[i],
               edgecolor="white", linewidth=0.5, alpha=0.5)

        bottoms_pos += np.array(pos_vals)
        bottoms_neg += np.array(neg_vals)

    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_LABELS.get(p, p) for p in planners], fontsize=10)
    ax.set_ylabel("Shapley Value ($\\Delta$ Success Rate)", fontsize=11)
    ax.set_title("Hazard Layer Attribution per Planner", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(output_dir, "shapley_attribution.png")
    pdf_path = os.path.join(output_dir, "shapley_attribution.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shapley attribution analysis.")
    p.add_argument("--seeds", type=int, default=10, help="Seeds per coalition (default: 10)")
    p.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    p.add_argument("--output", type=str, default=OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output, exist_ok=True)
    seeds = list(range(args.seeds))

    print("=== Shapley Attribution Analysis ===")
    print(f"  Scenario:  {args.scenario}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Features:  {FEATURES}")
    print(f"  Coalitions: {2**N_FEATURES}")
    print(f"  Planners:  {len(PLANNERS)}")
    print(f"  Total episodes: {2**N_FEATURES * len(PLANNERS) * args.seeds}")
    print()

    results: dict[str, dict[str, float]] = {}

    for planner_id in PLANNERS:
        print(f"  Planner: {planner_id}")
        t0 = time.perf_counter()
        shapley = compute_shapley_values(args.scenario, planner_id, seeds)
        elapsed = time.perf_counter() - t0
        results[planner_id] = shapley
        print(f"    Shapley values: {shapley}")
        print(f"    Sum: {sum(shapley.values()):.4f} (elapsed: {elapsed:.1f}s)")
        print()

    # Save CSV
    csv_path = os.path.join(args.output, "shapley_values.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["planner"] + FEATURES)
        writer.writeheader()
        for planner_id, shapley in results.items():
            row = {"planner": planner_id, **shapley}
            writer.writerow(row)
    print(f"  CSV: {csv_path}")

    plot_shapley_bar_chart(results, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
