#!/usr/bin/env python3
"""Generate planner characteristics comparison table (Task 2).

Reads all_episodes.csv and computes per-planner statistics including
the ranking inversion (Nav_Rank - Mission_Rank).

Output: outputs/paper_tables/planner_characteristics.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from flare.visualization.labels import PLANNER_ORDER, PLANNER_SHORT

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"
OUT_DIR = ROOT / "outputs" / "paper_tables"


def main() -> None:
    if not CSV.exists():
        print("ERROR: Run `python scripts/run_paper_experiments.py` first.")
        sys.exit(1)

    df = pd.read_csv(CSV)
    # Exclude dstar_lite alias rows
    df = df[df["planner_id"] != "dstar_lite"].copy()

    # Filter feasible episodes
    if "infeasible" in df.columns:
        feasible = df[df["infeasible"] != True].copy()  # noqa: E712
    else:
        feasible = df.copy()

    # Detect column names dynamically
    steps_col = next(
        (c for c in df.columns if "step" in c.lower() and "plan" not in c.lower()
         and "service" not in c.lower() and "fire" not in c.lower()),
        "executed_steps",
    )
    ms_col = next((c for c in df.columns if "mission_score" in c.lower()), None)

    print(f"Using steps column: {steps_col}")
    print(f"Using mission score column: {ms_col}")
    print()

    rows = []
    for pid in PLANNER_ORDER:
        ps = feasible[feasible["planner_id"] == pid]
        sr = ps["success"].mean() * 100
        succ = ps[ps["success"] == True]  # noqa: E712
        avg_steps = succ[steps_col].mean() if len(succ) > 0 else float("nan")
        avg_replans = ps["replans"].mean()

        if ms_col:
            avg_ms = ps[ms_col].mean()
        else:
            avg_ms = float("nan")

        rows.append({
            "pid": pid,
            "label": PLANNER_SHORT[pid],
            "sr": sr,
            "avg_steps": avg_steps,
            "avg_replans": avg_replans,
            "avg_ms": avg_ms,
        })

    # Compute ranks
    # Nav rank: by SR descending (higher is better → rank 1)
    sr_sorted = sorted(rows, key=lambda r: -r["sr"])
    for rank, r in enumerate(sr_sorted, 1):
        r["nav_rank"] = rank

    # Mission rank: by avg_ms descending (higher is better → rank 1)
    ms_sorted = sorted(rows, key=lambda r: -r["avg_ms"])
    for rank, r in enumerate(ms_sorted, 1):
        r["ms_rank"] = rank

    # Delta rank
    for r in rows:
        r["delta"] = r["nav_rank"] - r["ms_rank"]

    # Print summary
    print(f"{'Planner':<15} {'SR%':>6} {'Steps':>7} {'Replans':>8} {'MS':>7} "
          f"{'NavR':>5} {'MsR':>5} {'Delta':>6}")
    print("-" * 65)
    for r in rows:
        print(f"{r['label']:<15} {r['sr']:6.1f} {r['avg_steps']:7.0f} "
              f"{r['avg_replans']:8.1f} {r['avg_ms']:7.3f} "
              f"{r['nav_rank']:5d} {r['ms_rank']:5d} {r['delta']:+6d}")

    # Generate LaTeX
    tex_rows = []
    for r in rows:
        delta_str = f"{r['delta']:+d}"
        # Bold if delta != 0 (ranking inversion)
        if r["delta"] != 0:
            line = (
                f"\\textbf{{{r['label']}}} & \\textbf{{{r['sr']:.1f}}} & "
                f"\\textbf{{{r['avg_steps']:.0f}}} & \\textbf{{{r['avg_replans']:.1f}}} & "
                f"\\textbf{{{r['avg_ms']:.3f}}} & "
                f"\\textbf{{{r['nav_rank']}}} & \\textbf{{{r['ms_rank']}}} & "
                f"\\textbf{{{delta_str}}} \\\\"
            )
        else:
            line = (
                f"{r['label']} & {r['sr']:.1f} & "
                f"{r['avg_steps']:.0f} & {r['avg_replans']:.1f} & "
                f"{r['avg_ms']:.3f} & "
                f"{r['nav_rank']} & {r['ms_rank']} & "
                f"{delta_str} \\\\"
            )
        tex_rows.append(line)

    tex = r"""\begin{table}[t]
\centering
\caption{Planner performance comparison across all scenarios.
  $\Delta$~Rank = Nav.\ Rank $-$ Mission Rank: positive means better
  navigator than rescuer (ranking inversion). Bold rows indicate
  planners where navigation and mission rankings diverge.}
\label{tab:planner_characteristics}
\footnotesize
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
Planner & SR\% & Steps & Replans & MS & Nav.R & Ms.R & $\Delta$ Rank \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\vspace{2pt}
\begin{flushleft}
\scriptsize SR = feasible success rate (\%), Steps = mean steps (successes only),
Replans = mean replans, MS = mean normalized mission score.
\end{flushleft}
\end{table}"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "planner_characteristics.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
