#!/usr/bin/env python3
"""Improved Shapley attribution figure (Task 11).

IEEE single-column formatted grouped bar chart. Reuses existing
Shapley computation from shapley_attribution.py but with better
figure styling and annotations.

Output: outputs/paper_figures/shapley_attribution.{png,pdf}

Usage:
    python scripts/gen_shapley_figure.py [--seeds 5] [--scenario ...]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare.visualization.labels import (
    PLANNER_ORDER, PLANNER_SHORT,
)

ROOT = Path(__file__).resolve().parent.parent

# IEEE formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

FEATURES = ["fire", "traffic", "collapse"]
N_FEATURES = len(FEATURES)

_FEATURE_CONFIG_KEYS = {
    "fire": "enable_fire",
    "traffic": "enable_traffic",
    "collapse": "enable_collapse",
}

FEATURE_COLORS = {
    "fire": "#D55E00",
    "traffic": "#E69F00",
    "collapse": "#8B5A2B",
}
FEATURE_LABELS = {
    "fire": "Fire",
    "traffic": "Traffic",
    "collapse": "Collapse",
}

DEFAULT_SCENARIO = "osm_penteli_pharma_delivery_medium"


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def _evaluate_coalition(scenario_id, planner_id, coalition, seeds):
    """Run episodes for a planner with a given feature coalition."""
    from flare.benchmark.runner import run_episode
    from flare.scenarios.loader import load_scenario

    config = load_scenario(scenario_id)
    for i, feat in enumerate(FEATURES):
        key = _FEATURE_CONFIG_KEYS[feat]
        object.__setattr__(config, key, coalition[i])

    successes = 0
    total = 0
    for seed in seeds:
        try:
            result = run_episode(scenario_id, planner_id, seed, config_override=config)
            if result.metrics.get("termination_reason") == "success":
                successes += 1
            total += 1
        except Exception as e:
            print(f"        [ERROR seed={seed}] {e}")
            total += 1
    return successes / max(total, 1)


def compute_shapley_values(scenario_id, planner_id, seeds):
    """Compute exact Shapley values for one planner."""
    coalition_cache = {}
    for bits in range(2 ** N_FEATURES):
        coalition = tuple(bool(bits & (1 << i)) for i in range(N_FEATURES))
        active = [f for f, on in zip(FEATURES, coalition) if on]
        key = "+".join(active) if active else "baseline"
        print(f"      {key:30s}", end="", flush=True)
        val = _evaluate_coalition(scenario_id, planner_id, coalition, seeds)
        coalition_cache[coalition] = val
        print(f"  SR={val:.2f}")

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
                s_with = list(s_without)
                s_with[i] = True
                marginal = coalition_cache[tuple(s_with)] - coalition_cache[tuple(s_without)]
                s_size = sum(s_without)
                weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / math.factorial(n)
                phi += weight * marginal
        shapley[feat] = phi
    return shapley


def main() -> None:
    parser = argparse.ArgumentParser(description="Shapley attribution figure (improved).")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    print("=== Shapley Attribution Analysis ===")
    print(f"  Scenario:  {args.scenario}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Total episodes: {2**N_FEATURES * len(PLANNER_ORDER) * args.seeds}")
    print()

    results = {}
    for pid in PLANNER_ORDER:
        print(f"  Planner: {PLANNER_SHORT[pid]}")
        t0 = time.perf_counter()
        shapley = compute_shapley_values(args.scenario, pid, seeds)
        elapsed = time.perf_counter() - t0
        results[pid] = shapley
        print(f"    Shapley: {shapley}")
        print(f"    ({elapsed:.1f}s)\n")

    # Save CSV
    out_dir = ROOT / "outputs" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "shapley_values.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["planner"] + FEATURES)
        writer.writeheader()
        for pid, sv in results.items():
            writer.writerow({"planner": pid, **sv})
    print(f"  CSV: {csv_path}")

    # --- IEEE figure: grouped bars ---
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    n_planners = len(PLANNER_ORDER)
    n_features = len(FEATURES)
    bar_width = 0.22
    x = np.arange(n_planners)

    for i, feat in enumerate(FEATURES):
        values = [results[p][feat] for p in PLANNER_ORDER]
        offset = (i - (n_features - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            color=FEATURE_COLORS[feat],
            edgecolor="white", linewidth=0.3,
            label=FEATURE_LABELS[feat],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_SHORT[p] for p in PLANNER_ORDER], fontsize=7)
    ax.set_ylabel(r"Shapley value ($\Delta$ SR)", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=6.5, loc="lower left")

    # Key finding annotations
    # Find most negative fire Shapley (should be A*)
    fire_vals = {p: results[p]["fire"] for p in PLANNER_ORDER}
    worst_fire_p = min(fire_vals, key=fire_vals.get)
    if fire_vals[worst_fire_p] < -0.1:
        idx = PLANNER_ORDER.index(worst_fire_p)
        ax.annotate(
            "Cannot reroute",
            xy=(idx - bar_width, fire_vals[worst_fire_p]),
            xytext=(idx + 0.5, fire_vals[worst_fire_p] - 0.08),
            fontsize=5.5, color=FEATURE_COLORS["fire"],
            arrowprops=dict(arrowstyle="->", color=FEATURE_COLORS["fire"], lw=0.6),
        )

    # Find most negative collapse Shapley
    collapse_vals = {p: results[p]["collapse"] for p in PLANNER_ORDER}
    worst_collapse_p = min(collapse_vals, key=collapse_vals.get)
    if collapse_vals[worst_collapse_p] < -0.05:
        idx = PLANNER_ORDER.index(worst_collapse_p)
        ax.annotate(
            "Incremental repair\nfails on debris",
            xy=(idx + bar_width, collapse_vals[worst_collapse_p]),
            xytext=(idx - 1.0, collapse_vals[worst_collapse_p] - 0.06),
            fontsize=5.5, color=FEATURE_COLORS["collapse"],
            arrowprops=dict(arrowstyle="->", color=FEATURE_COLORS["collapse"], lw=0.6),
        )

    ax.set_title("Hazard attribution per planner (Shapley values)",
                  fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, "shapley_attribution")


if __name__ == "__main__":
    main()
