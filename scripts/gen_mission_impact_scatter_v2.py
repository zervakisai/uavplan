#!/usr/bin/env python3
"""Ranking inversion scatter — minimal best-paper style.

Left:  SR_feas vs Normalized Mission Score
Right: Mission score bar chart

Output: outputs/paper_figures/mission_impact_scatter_v2.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flare.visualization.labels import PLANNER_ORDER, PLANNER_SHORT

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import PLANNER_COLORS, apply_style, save

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"

COEFF_LABELS = {
    "astar": "(no risk)",
    "periodic_replan": r"$\alpha\!=\!5.0$",
    "aggressive_replan": r"$\beta\!=\!0.5$",
    "incremental_astar": r"$\gamma\!=\!2.0$",
    "apf": r"$\delta\!=\!3.0$",
}


def main() -> None:
    apply_style()

    if not CSV.exists():
        print("ERROR: Run experiments first.")
        sys.exit(1)

    df = pd.read_csv(CSV)
    df = df[df["planner_id"] != "dstar_lite"].copy()
    if "infeasible" in df.columns:
        feasible = df[df["infeasible"] != True].copy()  # noqa: E712
    else:
        feasible = df.copy()

    sr = feasible.groupby("planner_id")["success"].mean().reindex(PLANNER_ORDER) * 100
    mean_ms = (
        feasible.groupby(["scenario_id", "planner_id"])["mission_score"]
        .mean().unstack("planner_id").reindex(columns=PLANNER_ORDER)
    )
    scenario_max = mean_ms.max(axis=1)
    norm_ms = mean_ms.div(scenario_max, axis=0)
    avg_norm_ms = norm_ms.mean(axis=0)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                   gridspec_kw={"width_ratios": [1.3, 1]})

    # === LEFT: Scatter ===
    # Subtle quadrant hints — just thin dashed lines, no color fill
    ax.axhline(0.5, ls=":", lw=0.4, color="#999999", zorder=0)
    ax.axvline(50, ls=":", lw=0.4, color="#999999", zorder=0)
    ax.text(88, 0.97, "effective\nrescuer", fontsize=5.5, ha="center",
            color="#aaaaaa", style="italic", va="top")
    ax.text(88, 0.28, "navigates well,\nrescues poorly", fontsize=5.5,
            ha="center", color="#aaaaaa", style="italic", va="top")

    # Diagonal
    ax.plot([0, 100], [0, 1], ls="--", lw=0.4, color="#cccccc", zorder=0)

    # Points — small, clean
    for p in PLANNER_ORDER:
        ax.scatter(sr[p], avg_norm_ms[p], color=PLANNER_COLORS[p],
                   s=40, edgecolors="white", linewidths=0.6, zorder=5)

    # Labels — grey, not bold, with thin connector
    offsets = {
        "astar": (10, 6),
        "periodic_replan": (-6, -12),
        "aggressive_replan": (-12, 6),
        "incremental_astar": (8, 6),
        "apf": (8, -6),
    }
    for p in PLANNER_ORDER:
        ox, oy = offsets[p]
        label = f"{PLANNER_SHORT[p]} {COEFF_LABELS[p]}"
        ax.annotate(label, (sr[p], avg_norm_ms[p]),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=5.5, color="#444444",
                    arrowprops=dict(arrowstyle="-", color="#cccccc",
                                    lw=0.4) if abs(ox) > 8 else None)

    # Ranking inversion line
    best_sr_p = sr.idxmax()
    best_ms_p = avg_norm_ms.idxmax()
    if best_sr_p != best_ms_p:
        ax.plot([sr[best_sr_p], sr[best_ms_p]],
                [avg_norm_ms[best_sr_p], avg_norm_ms[best_ms_p]],
                ls="--", lw=0.8, color="#c75b3a", alpha=0.6, zorder=4)
        mid_x = (sr[best_sr_p] + sr[best_ms_p]) / 2
        mid_y = (avg_norm_ms[best_sr_p] + avg_norm_ms[best_ms_p]) / 2
        ax.text(mid_x - 2, mid_y + 0.06, "ranking\ninversion",
                fontsize=5.5, ha="center", color="#c75b3a", style="italic")

    ax.set_xlabel(r"Navigation success rate SR$_{\mathrm{feas}}$ (%)")
    ax.set_ylabel("Normalized mission score")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("(a) Navigation vs. mission performance", fontsize=8)

    # === RIGHT: Bar chart ===
    x = np.arange(len(PLANNER_ORDER))
    bars = ax2.bar(x, [avg_norm_ms[p] for p in PLANNER_ORDER],
                   color=[PLANNER_COLORS[p] for p in PLANNER_ORDER],
                   edgecolor="white", linewidth=0.4, width=0.7)

    for bar, p in zip(bars, PLANNER_ORDER):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{avg_norm_ms[p]:.2f}",
                 ha="center", va="bottom", fontsize=6, color="#444444")

    ax2.set_xticks(x)
    ax2.set_xticklabels([PLANNER_SHORT[p] for p in PLANNER_ORDER],
                         fontsize=6.5, rotation=0)
    ax2.set_ylabel("Normalized mission score")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("(b) Mission score comparison", fontsize=8)

    fig.tight_layout(w_pad=2.0)
    save(fig, "mission_impact_scatter_v2")

    print(f"\n{'Planner':<15} {'SR%':>6} {'NormMS':>7}")
    print("-" * 30)
    for p in PLANNER_ORDER:
        print(f"{PLANNER_SHORT[p]:<15} {sr[p]:6.1f} {avg_norm_ms[p]:7.3f}")
    if best_sr_p != best_ms_p:
        print(f"\nRanking inversion: best navigator={PLANNER_SHORT[best_sr_p]}, "
              f"best rescuer={PLANNER_SHORT[best_ms_p]}")


if __name__ == "__main__":
    main()
