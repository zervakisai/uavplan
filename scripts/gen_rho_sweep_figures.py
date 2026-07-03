#!/usr/bin/env python3
"""Figures for the unified-ρ sensitivity study (reviewer revision, Comment 1).

Reads outputs/rho_sweep/rho_sweep.csv (original 4 swept planners) and
outputs/rho_sweep/rho_sweep_new.csv (4 additional risk paradigms) and produces:
  1. rho_response_curves      — SR(ρ) and M(ρ) per planner (2 panels, 8 planners)
  2. rho_sensitivity_spectrum — Spearman(ρ, M) per planner, grouped by paradigm
  3. rho_ms_by_scenario       — M(ρ) per planner, one panel per scenario (robustness)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from paper_style import apply_style, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CSV = "outputs/rho_sweep/rho_sweep.csv"
CSV_NEW = "outputs/rho_sweep/rho_sweep_new.csv"
RHOS = [0.0, 1.0, 2.0, 5.0, 10.0]

# 8 risk-sensitive planners (A* is the risk-ignoring ρ=0 baseline, excluded).
PARADIGM = {
    "periodic_replan":   "Soft cost-inflation",
    "aggressive_replan": "Soft cost-inflation",
    "incremental_astar": "Soft cost-inflation",
    "chance_constrained": "Hard threshold",
    "cvar_conservative": "Worst-case / CVaR",
    "risk_sensitive":    "Exponential / entropic",
    "risk_aware_rrt":    "Sampling (RRT*)",
    "apf":               "Potential field",
}
PARADIGM_COLOR = {
    "Soft cost-inflation":     "#4a7c59",  # forest
    "Hard threshold":          "#b08968",  # tan
    "Worst-case / CVaR":       "#a03d5f",  # wine
    "Exponential / entropic":  "#d99b30",  # amber
    "Sampling (RRT*)":         "#3a7d7b",  # teal
    "Potential field":         "#5b8fa8",  # steel
}
LABELS = {
    "periodic_replan": "Periodic", "aggressive_replan": "Aggressive",
    "incremental_astar": "Incr. A*", "apf": "APF",
    "chance_constrained": "Chance-Constr.", "cvar_conservative": "CVaR",
    "risk_sensitive": "Risk-Sensitive", "risk_aware_rrt": "RRT*",
}
# distinct per-planner colors (paradigm hue, shaded within group)
PCOLOR = {
    "periodic_replan": "#4a7c59", "aggressive_replan": "#c75b3a",
    "incremental_astar": "#7b6d8e", "apf": "#5b8fa8",
    "chance_constrained": "#b08968", "cvar_conservative": "#a03d5f",
    "risk_sensitive": "#d99b30", "risk_aware_rrt": "#3a7d7b",
}
ORDER = ["periodic_replan", "aggressive_replan", "incremental_astar",
         "chance_constrained", "cvar_conservative", "risk_sensitive",
         "risk_aware_rrt", "apf"]
SCEN = {"osm_penteli_pharma_delivery_medium": "Penteli (pharma)",
        "osm_piraeus_urban_rescue_medium": "Piraeus (SAR)",
        "osm_downtown_fire_surveillance_medium": "Downtown (surveillance)"}


def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    if os.path.exists(CSV_NEW):
        df = pd.concat([df, pd.read_csv(CSV_NEW)], ignore_index=True)
    df = df[df["risk_rho"].astype(str) != "none"].copy()
    df["rho"] = df["risk_rho"].astype(float)
    return df


def _fsr(s):
    f = s[~s["infeasible"].astype(bool)]
    return f["success"].astype(float).mean() if len(f) else np.nan


def _xpos():
    return list(range(len(RHOS)))  # equal spacing; labels are the ρ values


def fig_curves(df):
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    xs = _xpos()
    for p in ORDER:
        d = df[df.planner_id == p]
        sr = [100 * _fsr(d[d.rho == r]) for r in RHOS]
        ms = [d[d.rho == r]["mission_score"].mean() for r in RHOS]
        kw = dict(color=PCOLOR[p], marker="o", markersize=3.2, linewidth=1.3)
        axl.plot(xs, sr, **kw)
        axr.plot(xs, ms, label=LABELS[p], **kw)
    for ax, ylab, ttl in ((axl, "Feasible success rate (%)", "(a) Navigation"),
                          (axr, "Mission score $M$", "(b) Mission value")):
        ax.set_xticks(xs); ax.set_xticklabels([f"{r:g}" for r in RHOS])
        ax.set_xlabel(r"Risk coefficient $\rho$")
        ax.set_ylabel(ylab); ax.set_title(ttl, loc="left", fontweight="bold")
        ax.margins(x=0.03)
    handles = [Line2D([0], [0], color=PCOLOR[p], marker="o", markersize=3,
                      linewidth=1.3, label=LABELS[p]) for p in ORDER]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.03),
               handlelength=1.6, columnspacing=1.3)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "rho_response_curves")


def fig_spectrum(df):
    rows = []
    for p in ORDER:
        d = df[df.planner_id == p]
        rm, pm = spearmanr(d["rho"], d["mission_score"].astype(float))
        rows.append((p, rm, pm))
    rows.sort(key=lambda t: t[1])  # most negative (most affected) first
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ys = range(len(rows))
    for y, (p, rm, pm) in zip(ys, rows):
        c = PARADIGM_COLOR[PARADIGM[p]]
        ax.barh(y, rm, color=c, edgecolor="white", linewidth=0.4, height=0.68)
        star = "*" if pm < 0.05 else ""
        # place value labels in the whitespace on the opposite side of zero
        if rm < 0:
            ax.text(0.006, y, f"{rm:+.2f}{star}", va="center", ha="left", fontsize=6)
        else:
            ax.text(-0.006, y, f"{rm:+.2f}{star}", va="center", ha="right", fontsize=6)
    ax.set_yticks(list(ys)); ax.set_yticklabels([LABELS[p] for p, _, _ in rows])
    ax.set_xlim(-0.30, 0.06)
    ax.axvline(0, color="#2c2c2c", linewidth=0.6)
    ax.set_xlabel(r"Spearman $\rho_s$ between $\rho$ and mission score $M$")
    ax.set_title("How strongly $\\rho$ erodes mission value, by paradigm",
                 loc="left", fontweight="bold", fontsize=7.5)
    ax.invert_yaxis()
    seen = []
    handles = []
    for p in ORDER:
        par = PARADIGM[p]
        if par not in seen:
            seen.append(par)
            handles.append(Patch(facecolor=PARADIGM_COLOR[par], label=par))
    ax.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower left",
              handlelength=1.0)
    ax.margins(y=0.02)
    fig.tight_layout()
    save(fig, "rho_sensitivity_spectrum")


def fig_by_scenario(df):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
    xs = _xpos()
    for ax, (sid, lbl) in zip(axes, SCEN.items()):
        s = df[df.scenario_id == sid]
        for p in ORDER:
            d = s[s.planner_id == p]
            ms = [d[d.rho == r]["mission_score"].mean() for r in RHOS]
            ax.plot(xs, ms, color=PCOLOR[p], marker="o", markersize=2.6,
                    linewidth=1.1, label=LABELS[p])
        ax.set_xticks(xs); ax.set_xticklabels([f"{r:g}" for r in RHOS])
        ax.set_xlabel(r"$\rho$"); ax.set_title(lbl, loc="left", fontsize=7.5)
    axes[0].set_ylabel("Mission score $M$")
    handles = [Line2D([0], [0], color=PCOLOR[p], marker="o", markersize=2.6,
                      linewidth=1.1, label=LABELS[p]) for p in ORDER]
    fig.legend(handles=handles, ncol=4, frameon=False, fontsize=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.06),
               handlelength=1.5, columnspacing=1.2)
    fig.suptitle("Mission value vs. ρ is consistent across scenarios",
                 x=0.01, ha="left", fontweight="bold", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    save(fig, "rho_ms_by_scenario")


def main():
    apply_style()
    df = _load()
    print(f"Loaded {len(df)} factorial rows; planners: {sorted(df.planner_id.unique())}")
    fig_curves(df)
    fig_spectrum(df)
    fig_by_scenario(df)
    print("Done.")


if __name__ == "__main__":
    main()
