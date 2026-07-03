#!/usr/bin/env python3
"""Scenario-separated Shapley hazard-attribution bar charts (Comment D).

3 panels (Penteli / Piraeus / Downtown). Within each panel, planners on the
x-axis with the three hazard layers (Fire / Traffic / Collapse) stacked as their
exact Shapley contribution to feasible success rate (Δ SR, in percentage points).

Output: outputs/paper_figures/shapley_scenarios.{png,pdf}
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from shapley_attribution import compute_shapley_values, FEATURES  # noqa: E402
from flare.visualization.labels import PLANNER_LABELS  # noqa: E402

SCEN = [
    ("osm_penteli_pharma_delivery_medium", "Penteli"),
    ("osm_piraeus_urban_rescue_medium", "Piraeus"),
    ("osm_downtown_fire_surveillance_medium", "Downtown"),
]
PLANNERS = ["periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
SEEDS = list(range(10))
COLORS = {"fire": "#e74c3c", "traffic": "#f39c12", "collapse": "#8e44ad"}
FLABEL = {"fire": "Fire", "traffic": "Traffic", "collapse": "Collapse"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "figure.dpi": 300})


def _task(args):
    sid, pid = args
    return sid, pid, compute_shapley_values(sid, pid, SEEDS)


def main() -> None:
    # results[scenario][planner][feature] = shapley (fraction of SR)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as _mp
    tasks = [(sid, pid) for sid, _ in SCEN for pid in PLANNERS]
    results: dict = {sid: {} for sid, _ in SCEN}
    workers = min(6, max(1, (_mp.cpu_count() or 4) - 2))
    print(f"  computing {len(tasks)} Shapley tasks on {workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for sid, pid, shap in ex.map(_task, tasks):
            results[sid][pid] = shap
            print(f"  done: {sid.split('_')[1]}/{pid}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    x = np.arange(len(PLANNERS))
    for ax, (sid, name) in zip(axes, SCEN):
        bpos = np.zeros(len(PLANNERS))
        bneg = np.zeros(len(PLANNERS))
        for feat in FEATURES:
            vals = np.array([results[sid][p][feat] * 100.0 for p in PLANNERS])  # -> pp
            pos = np.clip(vals, 0, None)
            neg = np.clip(vals, None, 0)
            ax.bar(x, pos, bottom=bpos, color=COLORS[feat], label=FLABEL[feat],
                   edgecolor="white", linewidth=0.4)
            ax.bar(x, neg, bottom=bneg, color=COLORS[feat], alpha=0.55,
                   edgecolor="white", linewidth=0.4)
            bpos += pos
            bneg += neg
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([PLANNER_LABELS.get(p, p) for p in PLANNERS],
                           rotation=30, ha="right", fontsize=6.5)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Shapley contribution to SR (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    # dedupe legend
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), frameon=False, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs("outputs/paper_figures", exist_ok=True)
    fig.savefig("outputs/paper_figures/shapley_scenarios.pdf", bbox_inches="tight")
    fig.savefig("outputs/paper_figures/shapley_scenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # dump numbers for the caption / verification
    for sid, name in SCEN:
        print(f"{name}:", {p: {f: round(results[sid][p][f] * 100, 1) for f in FEATURES} for p in PLANNERS})
    print("Saved -> outputs/paper_figures/shapley_scenarios.{pdf,png}")


if __name__ == "__main__":
    main()
