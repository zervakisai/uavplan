#!/usr/bin/env python3
"""Risk vs Mission Trade-off table — THE insight as a table.

Sorted by risk coefficient ascending (blind → tolerant → moderate → reactive → averse).
Shows detour%, Nav SR, Mission Score, and a human-readable Why column.

Output: outputs/paper_tables/risk_tradeoff.tex
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flare.visualization.labels import PLANNER_ORDER, PLANNER_SHORT

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "flare"
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"
OUT_DIR = ROOT / "outputs" / "paper_tables"


def _extract(filepath: Path, varname: str) -> float:
    source = filepath.read_text()
    match = re.search(rf"^{varname}\s*=\s*(.+)", source, re.MULTILINE)
    if not match:
        print(f"ERROR: Could not find {varname} in {filepath}")
        sys.exit(1)
    return float(ast.literal_eval(match.group(1).strip()))


# Risk attitude labels and sort order (ascending risk aversion)
RISK_ORDER = [
    ("astar", "Blind", None, 0.0),
    ("aggressive_replan", "Tolerant", r"$\beta{=}0.5$", None),
    ("incremental_astar", "Moderate", r"$\gamma{=}2.0$", None),
    ("apf", "Reactive", r"$\delta{=}3.0$", None),
    ("periodic_replan", "Averse", r"$\alpha{=}5.0$", None),
]

WHY_TEXT = {
    "astar": r"No detour $=$ fire collision",
    "aggressive_replan": r"Small detour, fast arrival $\rightarrow$ saves lives",
    "incremental_astar": r"Best navigator but medicine degrades en route",
    "apf": r"Repulsion traps in local minima near fire",
    "periodic_replan": r"Huge detour $=$ safe but casualties expire waiting",
}


def main() -> None:
    # Extract coefficients
    alpha = _extract(SRC / "planners" / "periodic_replan.py", "_RISK_ALPHA")
    beta = _extract(SRC / "planners" / "aggressive_replan.py", "_RISK_BETA")
    gamma = _extract(SRC / "planners" / "incremental_astar.py", "_RISK_GAMMA")
    delta = _extract(SRC / "planners" / "apf.py", "_RISK_DELTA")

    # Update RISK_ORDER with extracted values
    coeff_map = {
        "periodic_replan": (rf"$\alpha{{=}}{alpha}$", alpha),
        "aggressive_replan": (rf"$\beta{{=}}{beta}$", beta),
        "incremental_astar": (rf"$\gamma{{=}}{gamma}$", gamma),
        "apf": (rf"$\delta{{=}}{delta}$", delta),
        "astar": ("---", 0.0),
    }

    if not CSV.exists():
        print("ERROR: Run `python scripts/run_paper_experiments.py` first.")
        sys.exit(1)

    df = pd.read_csv(CSV)
    df = df[df["planner_id"] != "dstar_lite"].copy()

    if "infeasible" in df.columns:
        feasible = df[df["infeasible"] != True].copy()  # noqa: E712
    else:
        feasible = df.copy()

    # Compute stats
    stats = {}
    for pid in PLANNER_ORDER:
        ps = feasible[feasible["planner_id"] == pid]
        succ = ps[ps["success"] == True]  # noqa: E712
        sr = ps["success"].mean() * 100
        mean_steps = succ["executed_steps"].mean() if len(succ) > 0 else float("nan")
        stats[pid] = {"sr": sr, "mean_steps": mean_steps}

    # Normalized mission score
    mean_ms = (
        feasible.groupby(["scenario_id", "planner_id"])["mission_score"]
        .mean()
        .unstack("planner_id")
        .reindex(columns=PLANNER_ORDER)
    )
    scenario_max = mean_ms.max(axis=1)
    norm_ms = mean_ms.div(scenario_max.replace(0, 1), axis=0)
    avg_norm_ms = norm_ms.mean(axis=0)

    for pid in PLANNER_ORDER:
        stats[pid]["norm_ms"] = avg_norm_ms[pid]

    # Detour % relative to fastest successful planner
    # (A* has 0% SR so no successful steps to use as baseline)
    valid_steps = {p: stats[p]["mean_steps"] for p in PLANNER_ORDER
                   if not np.isnan(stats[p]["mean_steps"])}
    if valid_steps:
        baseline_steps = min(valid_steps.values())
    else:
        baseline_steps = float("nan")

    for pid in PLANNER_ORDER:
        s = stats[pid]["mean_steps"]
        if not np.isnan(s) and not np.isnan(baseline_steps) and baseline_steps > 0:
            stats[pid]["detour_pct"] = (s - baseline_steps) / baseline_steps * 100
        else:
            stats[pid]["detour_pct"] = float("nan")

    # Find best SR and best MS
    best_sr_pid = max(PLANNER_ORDER, key=lambda p: stats[p]["sr"])
    best_ms_pid = max(PLANNER_ORDER, key=lambda p: stats[p]["norm_ms"])

    # Build rows in risk-ascending order
    ordered_pids = [
        "astar", "aggressive_replan", "incremental_astar", "apf", "periodic_replan",
    ]
    risk_labels = {
        "astar": "Blind (A*)",
        "aggressive_replan": "Tolerant (Aggressive)",
        "incremental_astar": "Moderate (Incr.\\ A*)",
        "apf": "Reactive (APF)",
        "periodic_replan": "Averse (Periodic)",
    }

    print(f"\n{'Risk attitude':<25} {'Coeff':>8} {'Detour':>8} {'SR%':>6} {'MS':>6}")
    print("-" * 60)

    tex_rows = []
    for pid in ordered_pids:
        s = stats[pid]
        cl, cv = coeff_map[pid]

        if np.isnan(s["detour_pct"]):
            detour_str = "---"
        elif abs(s["detour_pct"]) < 0.5:
            detour_str = "0\\%"
        else:
            detour_str = f"+{s['detour_pct']:.0f}\\%"
        sr_str = f"{s['sr']:.0f}"
        ms_str = f"{s['norm_ms']:.2f}"

        # Star markers
        if pid == best_sr_pid:
            sr_str = r"$\star$ " + sr_str
        if pid == best_ms_pid:
            ms_str = r"$\star$ " + ms_str

        # Row coloring
        row_prefix = ""
        if pid == "aggressive_replan":
            row_prefix = r"\rowcolor{green!8}"
        elif pid == "incremental_astar":
            row_prefix = r"\rowcolor{blue!8}"

        line = (
            f"{row_prefix}"
            f"{risk_labels[pid]} & {cl} & {detour_str} & {sr_str} & {ms_str} & "
            f"{WHY_TEXT[pid]} \\\\"
        )
        tex_rows.append(line)

        print(f"{risk_labels[pid].replace(chr(92), ''):<25} {cl:>8} "
              f"{s['detour_pct']:>+7.0f}% {s['sr']:6.0f} {s['norm_ms']:6.2f}")

    tex = r"""\begin{table}[t]
\centering
\caption{Risk--mission trade-off. Sorted by risk aversion (ascending).
  $\star$ for best SR lands on Incr.\ A* ($\gamma{=}2.0$); $\star$ for
  best mission score lands on Aggressive ($\beta{=}0.5$). They are in
  \emph{different rows}---that is the paper's finding.}
\label{tab:risk_tradeoff}
\footnotesize
\begin{tabular}{@{}lcrlrp{3.8cm}@{}}
\toprule
Risk attitude & Coeff. & Detour & SR\% & MS & Why \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\vspace{2pt}
\begin{flushleft}
\scriptsize Detour = $(\bar{t}_p - \bar{t}_{\min}) / \bar{t}_{\min} \times 100$
over successful episodes, relative to fastest successful planner. MS = normalized mission score.
\end{flushleft}
\end{table}"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "risk_tradeoff.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
