#!/usr/bin/env python3
"""Generate ablation study figures and LaTeX tables from CSV results.

Reads outputs/ablation_results/*.csv, produces:
  - outputs/paper_figures/ablation_dynamics.{png,pdf}
  - outputs/paper_figures/ablation_replan_freq.{png,pdf}
  - outputs/paper_figures/ablation_fire_intensity.{png,pdf}
  - outputs/paper_tables/ablation_dynamics.tex
  - outputs/paper_tables/ablation_replan_freq.tex

Usage:
    python scripts/gen_ablation_figures.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uavbench.visualization.labels import (
    PLANNER_ORDER, PLANNER_SHORT, PLANNER_COLORS,
)

ABLATION_DIR = "outputs/ablation_results"
FIG_DIR = "outputs/paper_figures"
TABLE_DIR = "outputs/paper_tables"

VARIANT_ORDER_DYN = ["fire_only", "traffic_only", "all_dynamics"]
VARIANT_LABELS_DYN = {
    "fire_only": "Fire Only",
    "traffic_only": "Traffic Only",
    "all_dynamics": "All Dynamics",
}


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR}/{name}.{{png,pdf}}")


# ── Ablation 1: Dynamics Isolation ──────────────────────────────────────────


def gen_dynamics_figure() -> None:
    csv = os.path.join(ABLATION_DIR, "dynamics_isolation.csv")
    if not os.path.exists(csv):
        print("  SKIP: dynamics_isolation.csv not found")
        return
    df = pd.read_csv(csv)

    # Grouped bar chart: variant on x-axis, bars per planner
    fig, ax = plt.subplots(figsize=(8, 5))
    n_variants = len(VARIANT_ORDER_DYN)
    n_planners = len(PLANNER_ORDER)
    bar_width = 0.15
    x = np.arange(n_variants)

    for i, pid in enumerate(PLANNER_ORDER):
        srs = []
        for var in VARIANT_ORDER_DYN:
            sub = df[(df["variant"] == var) & (df["planner_id"] == pid)]
            srs.append(sub["success"].mean() * 100 if len(sub) > 0 else 0)
        offset = (i - n_planners / 2 + 0.5) * bar_width
        ax.bar(x + offset, srs, bar_width, label=PLANNER_SHORT[pid],
               color=PLANNER_COLORS[pid], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS_DYN[v] for v in VARIANT_ORDER_DYN], fontsize=10)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Dynamics Isolation Ablation", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "ablation_dynamics")


def gen_dynamics_table() -> None:
    csv = os.path.join(ABLATION_DIR, "dynamics_isolation.csv")
    if not os.path.exists(csv):
        return
    df = pd.read_csv(csv)

    rows = []
    for var in VARIANT_ORDER_DYN:
        row = {"Condition": VARIANT_LABELS_DYN[var]}
        for pid in PLANNER_ORDER:
            sub = df[(df["variant"] == var) & (df["planner_id"] == pid)]
            sr = sub["success"].mean() * 100 if len(sub) > 0 else 0
            row[PLANNER_SHORT[pid]] = f"{sr:.1f}"
        rows.append(row)

    cols = ["Condition"] + [PLANNER_SHORT[p] for p in PLANNER_ORDER]
    header = " & ".join(cols) + r" \\"
    lines = [
        r"\begin{tabular}{l" + "r" * len(PLANNER_ORDER) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in rows:
        vals = " & ".join(str(row[c]) for c in cols) + r" \\"
        lines.append(vals)
    lines += [r"\bottomrule", r"\end{tabular}"]

    tex_path = os.path.join(TABLE_DIR, "ablation_dynamics.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {tex_path}")


# ── Ablation 2: Replan Frequency ───────────────────────────────────────────


def gen_replan_freq_figure() -> None:
    csv = os.path.join(ABLATION_DIR, "replan_frequency.csv")
    if not os.path.exists(csv):
        print("  SKIP: replan_frequency.csv not found")
        return
    df = pd.read_csv(csv)

    # Extract cadence from variant name
    df["cadence"] = df["variant"].str.extract(r"replan_(\d+)").astype(int)
    cadences = sorted(df["cadence"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: SR vs cadence
    grouped = df.groupby("cadence")["success"].mean() * 100
    ax1.plot(cadences, [grouped[c] for c in cadences], "o-", color="#009E73",
             linewidth=2, markersize=8)
    ax1.set_xlabel("Replan Every N Steps", fontsize=11)
    ax1.set_ylabel("Success Rate (%)", fontsize=11)
    ax1.set_title("SR vs Replan Frequency", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 55)
    ax1.grid(alpha=0.3)

    # Right: Replans vs cadence
    grouped_r = df.groupby("cadence")["replans"].mean()
    ax2.plot(cadences, [grouped_r[c] for c in cadences], "s-", color="#D55E00",
             linewidth=2, markersize=8)
    ax2.set_xlabel("Replan Every N Steps", fontsize=11)
    ax2.set_ylabel("Avg. Replans", fontsize=11)
    ax2.set_title("Computation Cost vs Frequency", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "ablation_replan_freq")


def gen_replan_freq_table() -> None:
    csv = os.path.join(ABLATION_DIR, "replan_frequency.csv")
    if not os.path.exists(csv):
        return
    df = pd.read_csv(csv)
    df["cadence"] = df["variant"].str.extract(r"replan_(\d+)").astype(int)

    lines = [
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"Cadence & SR (\%) & Avg. Replans \\",
        r"\midrule",
    ]
    for c in sorted(df["cadence"].unique()):
        sub = df[df["cadence"] == c]
        sr = sub["success"].mean() * 100
        rp = sub["replans"].mean()
        lines.append(f"{c} & {sr:.1f} & {rp:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    tex_path = os.path.join(TABLE_DIR, "ablation_replan_freq.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {tex_path}")


# ── Ablation 3: Fire Intensity ─────────────────────────────────────────────


def gen_fire_intensity_figure() -> None:
    csv = os.path.join(ABLATION_DIR, "fire_intensity.csv")
    if not os.path.exists(csv):
        print("  SKIP: fire_intensity.csv not found")
        return
    df = pd.read_csv(csv)

    df["ignitions"] = df["variant"].str.extract(r"ignitions_(\d+)").astype(int)
    ignitions = sorted(df["ignitions"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))

    for pid in PLANNER_ORDER:
        srs = []
        for ign in ignitions:
            sub = df[(df["ignitions"] == ign) & (df["planner_id"] == pid)]
            srs.append(sub["success"].mean() * 100 if len(sub) > 0 else 0)
        ax.plot(ignitions, srs, "o-", label=PLANNER_SHORT[pid],
                color=PLANNER_COLORS[pid], linewidth=2, markersize=7)

    ax.set_xlabel("Fire Ignition Points", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Fire Intensity Ablation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(-2, 55)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "ablation_fire_intensity")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    print("Generating ablation figures and tables...")
    gen_dynamics_figure()
    gen_dynamics_table()
    gen_replan_freq_figure()
    gen_replan_freq_table()
    gen_fire_intensity_figure()
    print("Done.")


if __name__ == "__main__":
    main()
