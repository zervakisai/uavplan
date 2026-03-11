#!/usr/bin/env python3
"""Mission Impact scatter: Navigation SR vs Normalized Mission Score.

Generates outputs/paper_figures/mission_impact_scatter.{pdf,png}.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"
OUT = ROOT / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── planner config ───────────────────────────────────────────────────────
PLANNERS = ["astar", "periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
COLORS = {
    "astar": "#E69F00",
    "periodic_replan": "#009E73",
    "aggressive_replan": "#D55E00",
    "incremental_astar": "#CC79A7",
    "apf": "#0072B2",
}
LABELS = {
    "astar": "A*",
    "periodic_replan": "Periodic",
    "aggressive_replan": "Aggressive",
    "incremental_astar": "Incr. A*",
    "apf": "APF",
}

# ── load & filter ────────────────────────────────────────────────────────
df = pd.read_csv(CSV)
# Exclude dstar_lite rows (use incremental_astar only)
df = df[df["planner_id"] != "dstar_lite"].copy()

# ── SR_feas: success rate over feasible episodes ─────────────────────────
feasible = df[df["infeasible"] != True].copy()  # noqa: E712
sr = (
    feasible.groupby("planner_id")["success"]
    .mean()
    .reindex(PLANNERS)
    * 100
)

# ── Normalized mission score ─────────────────────────────────────────────
# Per (scenario, planner): mean mission_score over all episodes (feasible only)
mean_ms = (
    feasible.groupby(["scenario_id", "planner_id"])["mission_score"]
    .mean()
    .unstack("planner_id")
    .reindex(columns=PLANNERS)
)
# Normalize each scenario row to [0, 1] by dividing by that scenario's max
scenario_max = mean_ms.max(axis=1)
norm_ms = mean_ms.div(scenario_max, axis=0)
# Average across scenarios
avg_norm_ms = norm_ms.mean(axis=0)

# ── figure ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Dashed diagonal reference line (from bottom-left to top-right)
ax.plot([0, 100], [0, 1], ls="--", lw=0.8, color="grey", alpha=0.45, zorder=0)

for p in PLANNERS:
    ax.scatter(
        sr[p],
        avg_norm_ms[p],
        color=COLORS[p],
        s=60,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label=LABELS[p],
    )

# Labels near points (hand-tuned offsets based on actual positions)
offsets = {
    "astar": (12, 8),              # corner point, push above-right for visibility
    "periodic_replan": (-8, -12),   # top-right, push label below-left to avoid edge
    "aggressive_replan": (-12, 6),  # near periodic, push label above-left
    "incremental_astar": (7, 6),   # above-right of dot
    "apf": (7, -2),                # mid-right, label right
}
for p in PLANNERS:
    ax.annotate(
        LABELS[p],
        (sr[p], avg_norm_ms[p]),
        textcoords="offset points",
        xytext=offsets[p],
        fontsize=6.5,
        color=COLORS[p],
        fontweight="bold",
    )

# Annotation arrow for ranking inversion insight
ax.annotate(
    "Best navigator $\\neq$\nbest rescuer",
    xy=(sr["incremental_astar"], avg_norm_ms["incremental_astar"]),
    xytext=(sr["incremental_astar"] - 32, avg_norm_ms["incremental_astar"] - 0.22),
    fontsize=6.5,
    fontweight="bold",
    color=COLORS["incremental_astar"],
    arrowprops=dict(
        arrowstyle="->",
        color=COLORS["incremental_astar"],
        lw=1.0,
        connectionstyle="arc3,rad=0.2",
    ),
    ha="center",
)

ax.set_xlabel("Navigation Success Rate  SR$_{\\mathrm{feas}}$ (%)")
ax.set_ylabel("Normalized Mission Score")
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.15)

fig.tight_layout()
fig.savefig(OUT / "mission_impact_scatter.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "mission_impact_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── summary table ────────────────────────────────────────────────────────
print(f"{'Planner':<22} {'SR_feas%':>8}  {'NormMS':>7}")
print("-" * 40)
for p in PLANNERS:
    print(f"{LABELS[p]:<22} {sr[p]:8.1f}  {avg_norm_ms[p]:7.3f}")
print(f"\nSaved → {OUT / 'mission_impact_scatter.pdf'}")
print(f"Saved → {OUT / 'mission_impact_scatter.png'}")
