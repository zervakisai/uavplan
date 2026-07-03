#!/usr/bin/env python3
"""Regenerate Figure 8 (ranking inversion) to match its caption, with the
fire-coupled data: (a) SR vs normalised mission score scatter; (b) per-planner
mission-score decomposition — grouped bars (Penteli/Piraeus/Downtown) + avg."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from paper_style import apply_style, PLANNER_COLORS, PLANNER_LABELS, save
import matplotlib.pyplot as plt

CSV = "outputs/paper_results/all_episodes.csv"
SCEN = {"osm_penteli_pharma_delivery_medium": ("Penteli", "#4a7c59"),
        "osm_piraeus_urban_rescue_medium": ("Piraeus", "#c75b3a"),
        "osm_downtown_fire_surveillance_medium": ("Downtown", "#5b8fa8")}
ORDER = ["astar", "periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
# Ranking figure: planners are identified by name only. The risk coefficient ρ
# is deliberately NOT shown here — ρ's effect is reported per-planner-family in
# the ρ-sweep figures (cost surface, trajectory, response curves) and Table 5.


def main():
    apply_style()
    df = pd.read_csv(CSV)
    feas = df[df["infeasible"] != True] if "infeasible" in df.columns else df  # noqa: E712
    sr = {p: feas[feas.planner_id == p]["success"].mean() * 100 for p in ORDER}
    # per-scenario normalised MS
    per = {}
    for sid in SCEN:
        s = feas[feas.scenario_id == sid]
        means = {p: s[s.planner_id == p]["mission_score"].mean() for p in ORDER}
        mx = max(means.values()) or 1.0
        per[sid] = {p: (means[p] / mx if mx else 0.0) for p in ORDER}
    normms = {p: float(np.mean([per[sid][p] for sid in SCEN])) for p in ORDER}

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) scatter
    # ranking-inversion judgment zone: below the identity line a planner ranks
    # higher on navigation than on mission (normalised M < SR) -> inversion region
    _xs = np.linspace(0, 100, 200)
    axa.fill_between(_xs, 0, _xs / 100.0, color="#c75b3a", alpha=0.07, zorder=0)
    axa.plot([0, 100], [0, 1], ls="--", color="#bbbbbb", lw=0.8, zorder=0)
    for p in ORDER:
        axa.scatter(sr[p], normms[p], s=48, color=PLANNER_COLORS[p], zorder=3,
                    edgecolor="white", linewidth=0.6)
        axa.annotate(PLANNER_LABELS[p], (sr[p], normms[p]), fontsize=6.3,
                     xytext=(5, 3), textcoords="offset points")
    # ranking-inversion arrow: nav-best (Incr) up to mission-best (Aggressive)
    axa.annotate("", xy=(sr["aggressive_replan"], normms["aggressive_replan"]),
                 xytext=(sr["incremental_astar"], normms["incremental_astar"]),
                 arrowprops=dict(arrowstyle="->", color="#c75b3a", lw=1.0, ls="--"))
    axa.text(52, 0.72, "ranking\ninversion", color="#c75b3a", fontsize=6.3, ha="center")
    axa.text(85, 0.995, "aligned region", color="#999999", fontsize=5.8, ha="center", style="italic")
    axa.text(72, 0.22, "ranking-inversion zone\n(high navigation, low mission)",
             color="#c75b3a", fontsize=5.8, ha="center", style="italic")
    axa.set_xlabel(r"Navigation success rate SR$_{\mathrm{feas}}$ (%)")
    axa.set_ylabel("Normalised mission score $M$")
    axa.set_title("(a) Navigation vs. mission performance", loc="left", fontweight="bold", fontsize=8)
    axa.set_xlim(-3, 103); axa.set_ylim(-0.03, 1.12)

    # (b) grouped per-scenario bars + avg
    x = np.arange(len(ORDER)); w = 0.24
    for j, sid in enumerate(SCEN):
        vals = [per[sid][p] for p in ORDER]
        axb.bar(x + (j - 1) * w, vals, w, color=SCEN[sid][1], label=SCEN[sid][0],
                edgecolor="white", linewidth=0.3)
    for i, p in enumerate(ORDER):
        axb.text(i, 1.06, f"avg {normms[p]:.2f}", ha="center", fontsize=6, color="#333333")
    axb.set_xticks(x); axb.set_xticklabels([PLANNER_LABELS[p] for p in ORDER])
    axb.set_ylabel("Per-scenario normalised $M$")
    axb.set_title("(b) Mission-score decomposition per scenario", loc="left", fontweight="bold", fontsize=8)
    axb.set_ylim(0, 1.16)
    axb.legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout()
    save(fig, "fig8_ranking_inversion")
    print("normMS:", {p: round(normms[p], 3) for p in ORDER})
    print("Piraeus per-scenario:", {p: round(per['osm_piraeus_urban_rescue_medium'][p], 2) for p in ORDER})


if __name__ == "__main__":
    main()
