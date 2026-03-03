#!/usr/bin/env python3
"""Analyze paper experiment results and generate paper-ready outputs.

Reads outputs/paper_results/all_episodes.csv.
Generates:
  - LaTeX tables in outputs/paper_tables/
  - Figures (PNG 300dpi + PDF) in outputs/paper_figures/

Requirements: pandas, matplotlib, scipy, numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths (defaults, overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULT_INPUT_CSV = "outputs/paper_results/all_episodes.csv"
DEFAULT_TABLE_DIR = "outputs/paper_tables"
DEFAULT_FIG_DIR = "outputs/paper_figures"

# Planner display order
PLANNER_ORDER = [
    "astar", "theta_star", "periodic_replan",
    "aggressive_replan", "dstar_lite", "mppi_grid",
]
PLANNER_LABELS = {
    "astar": "A*",
    "theta_star": r"$\theta$*",
    "periodic_replan": "Periodic",
    "aggressive_replan": "Aggressive",
    "dstar_lite": "D* Lite",
    "mppi_grid": "MPPI",
}
PLANNER_COLORS = {
    "astar": "#1f77b4",
    "theta_star": "#ff7f0e",
    "periodic_replan": "#2ca02c",
    "aggressive_replan": "#d62728",
    "dstar_lite": "#9467bd",
    "mppi_grid": "#8c564b",
}
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label(pid: str) -> str:
    return PLANNER_LABELS.get(pid, pid)


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both 300dpi PNG and vector PDF."""
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {png_path}")


def _save_tex(tex: str, name: str) -> None:
    """Save LaTeX table to .tex file."""
    path = os.path.join(TABLE_DIR, f"{name}.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Table:  {path}")


def _mean_std_str(series: pd.Series, fmt: str = ".1f") -> str:
    """Format a series as 'mean +/- std'."""
    return f"{series.mean():{fmt}} $\\pm$ {series.std():{fmt}}"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def gen_table_per_scenario(df: pd.DataFrame) -> str:
    """LaTeX table: planner comparison per scenario (Table a)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-scenario planner comparison (mean $\pm$ std over 30 seeds).}",
        r"\label{tab:per_scenario}",
        r"\footnotesize",
        r"\begin{tabular}{ll" + "r" * 5 + "}",
        r"\toprule",
        r"Scenario & Planner & Success\% & Path Len & Steps & Replans & Time (ms) \\",
        r"\midrule",
    ]

    scenarios = sorted(df["scenario_id"].unique())
    for i, scn in enumerate(scenarios):
        sub = df[df["scenario_id"] == scn]
        for j, pid in enumerate(PLANNER_ORDER):
            ps = sub[sub["planner_id"] == pid]
            if ps.empty:
                continue
            scn_label = scn.replace("gov_", "").replace("_", r"\_") if j == 0 else ""
            sr = f"{100 * ps['success'].mean():.0f}"
            pl = _mean_std_str(ps["path_length"])
            es = _mean_std_str(ps["executed_steps"])
            rp = _mean_std_str(ps["replans"], ".0f")
            ct = _mean_std_str(ps["computation_time_ms"], ".0f")
            lines.append(
                f"  {scn_label} & {_label(pid)} & {sr} & {pl} & {es} & {rp} & {ct} \\\\"
            )
        if i < len(scenarios) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table_by_difficulty(df: pd.DataFrame) -> str:
    """LaTeX table: aggregated by difficulty (Table b)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Planner performance aggregated by difficulty.}",
        r"\label{tab:by_difficulty}",
        r"\begin{tabular}{ll" + "r" * 3 + "}",
        r"\toprule",
        r"Difficulty & Planner & Success\% & Path Len & Time (ms) \\",
        r"\midrule",
    ]

    for i, diff in enumerate(DIFFICULTY_ORDER):
        sub = df[df["difficulty"] == diff]
        for j, pid in enumerate(PLANNER_ORDER):
            ps = sub[sub["planner_id"] == pid]
            if ps.empty:
                continue
            diff_label = diff.capitalize() if j == 0 else ""
            sr = f"{100 * ps['success'].mean():.0f}"
            pl = _mean_std_str(ps["path_length"])
            ct = _mean_std_str(ps["computation_time_ms"], ".0f")
            lines.append(f"  {diff_label} & {_label(pid)} & {sr} & {pl} & {ct} \\\\")
        if i < len(DIFFICULTY_ORDER) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table_exclusion(df: pd.DataFrame) -> str:
    """LaTeX table: infeasible/exclusion rate per scenario (Table g)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Episode exclusion rate (infeasible episodes per scenario).}",
        r"\label{tab:exclusion}",
        r"\begin{tabular}{l" + "r" * len(PLANNER_ORDER) + "}",
        r"\toprule",
        "Scenario & " + " & ".join(_label(p) for p in PLANNER_ORDER) + r" \\",
        r"\midrule",
    ]

    scenarios = sorted(df["scenario_id"].unique())
    for scn in scenarios:
        sub = df[df["scenario_id"] == scn]
        scn_label = scn.replace("gov_", "").replace("_", r"\_")
        vals = []
        for pid in PLANNER_ORDER:
            ps = sub[sub["planner_id"] == pid]
            if ps.empty:
                vals.append("--")
            else:
                rate = ps["infeasible"].mean()
                vals.append(f"{100 * rate:.0f}\\%")
        lines.append(f"  {scn_label} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def gen_table_significance(df: pd.DataFrame) -> str:
    """LaTeX table: Wilcoxon signed-rank p-values between planner pairs (Table h)."""
    # Aggregate per scenario×seed to get one value per (planner, scenario, seed)
    # We test on path_length across all scenarios
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Wilcoxon signed-rank p-values on path length (all scenarios).}",
        r"\label{tab:significance}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "c" * len(PLANNER_ORDER) + "}",
        r"\toprule",
        " & " + " & ".join(_label(p) for p in PLANNER_ORDER) + r" \\",
        r"\midrule",
    ]

    # Build pivot: index = (scenario_id, seed), columns = planner_id, values = path_length
    pivot = df.pivot_table(
        index=["scenario_id", "seed"],
        columns="planner_id",
        values="path_length",
        aggfunc="first",
    )

    for pa in PLANNER_ORDER:
        vals = []
        for pb in PLANNER_ORDER:
            if pa == pb:
                vals.append("--")
            elif pa in pivot.columns and pb in pivot.columns:
                a = pivot[pa].dropna()
                b = pivot[pb].dropna()
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    vals.append("n/a")
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            stat, p = stats.wilcoxon(
                                a.loc[common].values,
                                b.loc[common].values,
                                alternative="two-sided",
                                zero_method="wilcox",
                            )
                        except ValueError:
                            p = 1.0
                    if p < 0.001:
                        vals.append(r"\textbf{<.001}")
                    elif p < 0.01:
                        vals.append(f"\\textbf{{{p:.3f}}}")
                    elif p < 0.05:
                        vals.append(f"{p:.3f}*")
                    else:
                        vals.append(f"{p:.3f}")
            else:
                vals.append("n/a")
        lines.append(f"  {_label(pa)} & " + " & ".join(vals) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------


def fig_boxplot_path_length(df: pd.DataFrame) -> None:
    """Box plots: path_length by planner, one subplot per difficulty (Figure c)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, diff in zip(axes, DIFFICULTY_ORDER):
        sub = df[df["difficulty"] == diff]
        data = [sub[sub["planner_id"] == p]["path_length"].dropna() for p in PLANNER_ORDER]
        bp = ax.boxplot(
            data,
            tick_labels=[_label(p) for p in PLANNER_ORDER],
            patch_artist=True,
            widths=0.6,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, pid in zip(bp["boxes"], PLANNER_ORDER):
            patch.set_facecolor(PLANNER_COLORS[pid])
            patch.set_alpha(0.7)
        ax.set_title(f"{diff.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Path Length" if diff == "easy" else "")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Path Length by Planner and Difficulty", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "boxplot_path_length")


def fig_bar_success_rate(df: pd.DataFrame) -> None:
    """Bar chart: success rate per planner x scenario (Figure d)."""
    scenarios = sorted(df["scenario_id"].unique())
    n_scn = len(scenarios)
    n_pln = len(PLANNER_ORDER)
    bar_w = 0.8 / n_pln
    x = np.arange(n_scn)

    fig, ax = plt.subplots(figsize=(max(12, n_scn * 1.2), 5))

    for i, pid in enumerate(PLANNER_ORDER):
        rates = []
        for scn in scenarios:
            ps = df[(df["scenario_id"] == scn) & (df["planner_id"] == pid)]
            rates.append(100 * ps["success"].mean() if not ps.empty else 0)
        ax.bar(
            x + i * bar_w - 0.4 + bar_w / 2,
            rates,
            bar_w * 0.9,
            label=_label(pid),
            color=PLANNER_COLORS[pid],
            alpha=0.85,
        )

    ax.set_xticks(x)
    short_labels = [s.replace("gov_", "").replace("_", "\n") for s in scenarios]
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Success Rate per Planner and Scenario", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "bar_success_rate")


def fig_heatmap_success(df: pd.DataFrame) -> None:
    """Heatmap: planner x scenario success rate matrix (Figure e)."""
    scenarios = sorted(df["scenario_id"].unique())
    matrix = np.zeros((len(PLANNER_ORDER), len(scenarios)))

    for i, pid in enumerate(PLANNER_ORDER):
        for j, scn in enumerate(scenarios):
            ps = df[(df["scenario_id"] == scn) & (df["planner_id"] == pid)]
            matrix[i, j] = 100 * ps["success"].mean() if not ps.empty else 0

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 0.9), 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(scenarios)))
    short_labels = [s.replace("gov_", "").replace("_", "\n") for s in scenarios]
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_yticks(range(len(PLANNER_ORDER)))
    ax.set_yticklabels([_label(p) for p in PLANNER_ORDER], fontsize=9)

    for i in range(len(PLANNER_ORDER)):
        for j in range(len(scenarios)):
            val = matrix[i, j]
            color = "white" if val < 50 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, label="Success Rate (%)", shrink=0.8)
    fig.suptitle("Success Rate Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "heatmap_success")


def fig_replanning_comparison(df: pd.DataFrame) -> None:
    """Replanning frequency comparison across planners (Figure f)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

    for ax, diff in zip(axes, DIFFICULTY_ORDER):
        sub = df[df["difficulty"] == diff]
        means = []
        stds = []
        for pid in PLANNER_ORDER:
            ps = sub[sub["planner_id"] == pid]["replans"]
            means.append(ps.mean() if not ps.empty else 0)
            stds.append(ps.std() if not ps.empty else 0)

        x = np.arange(len(PLANNER_ORDER))
        colors = [PLANNER_COLORS[p] for p in PLANNER_ORDER]
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=3, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(p) for p in PLANNER_ORDER], rotation=30, fontsize=8)
        ax.set_title(f"{diff.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Replans" if diff == "easy" else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Replanning Frequency by Difficulty", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, "replanning_comparison")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze UAVBench v2 paper results.")
    p.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path (default: {DEFAULT_INPUT_CSV})",
    )
    p.add_argument(
        "--table-dir", type=str, default=DEFAULT_TABLE_DIR,
        help=f"Output directory for LaTeX tables (default: {DEFAULT_TABLE_DIR})",
    )
    p.add_argument(
        "--fig-dir", type=str, default=DEFAULT_FIG_DIR,
        help=f"Output directory for figures (default: {DEFAULT_FIG_DIR})",
    )
    return p.parse_args()


def main() -> None:
    global TABLE_DIR, FIG_DIR  # noqa: PLW0603
    args = _parse_args()
    input_csv = args.input
    TABLE_DIR = args.table_dir
    FIG_DIR = args.fig_dir

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found. Run run_v2_paper_experiments.py first.")
        sys.exit(1)

    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(input_csv)
    n_episodes = len(df)
    n_scenarios = df["scenario_id"].nunique()
    n_planners = df["planner_id"].nunique()
    n_seeds = df["seed"].nunique()

    print(f"UAVBench v2 Paper Analysis")
    print(f"  Episodes:  {n_episodes}")
    print(f"  Scenarios: {n_scenarios}")
    print(f"  Planners:  {n_planners}")
    print(f"  Seeds:     {n_seeds}")
    print()

    # --- Tables ---
    print("Generating LaTeX tables...")
    _save_tex(gen_table_per_scenario(df), "planner_per_scenario")
    _save_tex(gen_table_by_difficulty(df), "planner_by_difficulty")
    _save_tex(gen_table_exclusion(df), "exclusion_rate")
    _save_tex(gen_table_significance(df), "significance_matrix")

    # --- Figures ---
    print("\nGenerating figures...")
    fig_boxplot_path_length(df)
    fig_bar_success_rate(df)
    fig_heatmap_success(df)
    fig_replanning_comparison(df)

    # --- Summary stats to stdout ---
    print("\n--- Summary Statistics ---")
    for diff in DIFFICULTY_ORDER:
        sub = df[df["difficulty"] == diff]
        print(f"\n  {diff.upper()}:")
        for pid in PLANNER_ORDER:
            ps = sub[sub["planner_id"] == pid]
            if ps.empty:
                continue
            sr = 100 * ps["success"].mean()
            pl = ps["path_length"].mean()
            ct = ps["computation_time_ms"].mean()
            rp = ps["replans"].mean()
            print(f"    {_label(pid):12s}  SR={sr:5.1f}%  PL={pl:7.0f}  T={ct:7.0f}ms  RP={rp:5.1f}")

    print(f"\nOutputs written to {TABLE_DIR}/ and {FIG_DIR}/")


if __name__ == "__main__":
    main()
