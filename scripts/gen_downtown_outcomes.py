#!/usr/bin/env python3
"""Per-planner termination outcomes for the Downtown scenario.

Generates a single-panel stacked horizontal bar chart that tells the
Downtown-specific story:
  - Static density 0.50 closes 4/30 seeds for every planner (infeasibility floor)
  - A* catastrophic 0/30 success (26 collisions)
  - Incremental A* wins 18/30 (60%)
  - Distinct failure-mode signatures per planner

Reads `outputs/paper_results/all_episodes.csv` and writes
`paper/figures/downtown_outcomes.{pdf,png}`.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flare.visualization.labels import PLANNER_ORDER, PLANNER_SHORT  # noqa: E402

CSV_PATH = REPO / "outputs/paper_results/all_episodes.csv"
OUT_DIR = REPO / "paper/figures"
SCENARIO_ID = "osm_downtown_fire_surveillance_medium"
N_SEEDS = 30

OUTCOME_ORDER = [
    "success",
    "fire_caught",
    "debris_caught",
    "vehicle_collision",
    "goal_stall",
    "infeasible",
]

OUTCOME_LABELS = {
    "success": "Success",
    "fire_caught": "Fire caught",
    "debris_caught": "Debris caught",
    "vehicle_collision": "Vehicle collision",
    "goal_stall": "Goal stall",
    "infeasible": "Infeasible (static)",
}

OUTCOME_COLORS = {
    "success": "#2E7D32",
    "fire_caught": "#D55E00",
    "debris_caught": "#5D4037",
    "vehicle_collision": "#9C27B0",
    "goal_stall": "#0072B2",
    "infeasible": "#9E9E9E",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 300,
    }
)


def load_downtown_outcomes() -> dict[str, dict[str, int]]:
    """Return {planner_id: {outcome: count}} for Downtown rows."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["scenario_id"] != SCENARIO_ID:
                continue
            planner = row["planner_id"]
            if row["infeasible"] in ("True", "1", "true"):
                outcome = "infeasible"
            else:
                outcome = row["termination_reason"]
            counts[planner][outcome] += 1
    return {p: dict(counts[p]) for p in counts}


def main() -> None:
    counts = load_downtown_outcomes()
    if not counts:
        raise SystemExit(f"No rows for scenario_id={SCENARIO_ID} in {CSV_PATH}")

    # Sort planners by success count (descending) for visual clarity.
    planners = sorted(
        [p for p in PLANNER_ORDER if p in counts],
        key=lambda p: counts[p].get("success", 0),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    y_positions = list(range(len(planners)))
    for y, planner in zip(y_positions, planners):
        x = 0
        for outcome in OUTCOME_ORDER:
            n = counts[planner].get(outcome, 0)
            if n == 0:
                continue
            ax.barh(
                y,
                n,
                left=x,
                color=OUTCOME_COLORS[outcome],
                edgecolor="white",
                linewidth=0.6,
                height=0.7,
            )
            if n >= 2:
                ax.text(
                    x + n / 2,
                    y,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
            x += n

    feasibility_ceiling = N_SEEDS - max(
        counts[p].get("infeasible", 0) for p in planners
    )
    ax.axvline(
        feasibility_ceiling,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        zorder=5,
    )
    ax.annotate(
        f"feasibility ceiling = {feasibility_ceiling}/30\n"
        f"(static density closes {N_SEEDS - feasibility_ceiling} seeds for every planner)",
        xy=(feasibility_ceiling, -0.45),
        xytext=(feasibility_ceiling - 0.5, -0.9),
        ha="right",
        va="center",
        fontsize=6.5,
        style="italic",
        color="black",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.6, alpha=0.7),
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([PLANNER_SHORT[p] for p in planners])
    ax.invert_yaxis()
    ax.set_ylim(len(planners) - 0.5, -1.4)
    ax.set_xlim(0, N_SEEDS)
    ax.set_xticks(range(0, N_SEEDS + 1, 5))
    ax.set_xlabel(f"Number of seeds (out of {N_SEEDS})")
    ax.set_title(
        "Downtown — per-planner termination outcomes (fire-surveillance, density 0.50)",
        loc="left",
        pad=6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o])
        for o in OUTCOME_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        handlelength=1.3,
        columnspacing=1.5,
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.28)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "downtown_outcomes.pdf"
    png_path = OUT_DIR / "downtown_outcomes.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print()
    print("Summary (Downtown, n=30 seeds per planner):")
    for p in planners:
        s = counts[p].get("success", 0)
        i = counts[p].get("infeasible", 0)
        print(f"  {PLANNER_SHORT[p]:>10s}: success={s:>2d}/30, infeasible={i}/30")


if __name__ == "__main__":
    main()
