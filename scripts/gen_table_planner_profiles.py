#!/usr/bin/env python3
"""Planner Profiles — the Rosetta Stone table.

Merges risk_profile.tex + planner_characteristics.tex into ONE table.
One row per planner: family, risk coeff, replan trigger, SR%, MS,
Nav Rank, Mission Rank, Delta Rank with star symbols.

Output: outputs/paper_tables/planner_profiles.tex
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

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


# Planner metadata (static properties)
PLANNER_META = {
    "astar": {
        "family": "Static",
        "coeff_label": "---",
        "cost_formula": r"$w(c) = 1$",
        "replan": "Never",
    },
    "periodic_replan": {
        "family": "Adaptive",
        "coeff_label": None,  # filled dynamically
        "cost_formula": None,
        "replan": "Every 6 steps",
    },
    "aggressive_replan": {
        "family": "Adaptive",
        "coeff_label": None,
        "cost_formula": None,
        "replan": "On obstacle change",
    },
    "incremental_astar": {
        "family": "Adaptive",
        "coeff_label": None,
        "cost_formula": None,
        "replan": "On path blocked",
    },
    "apf": {
        "family": "Reactive",
        "coeff_label": None,
        "cost_formula": None,
        "replan": "Every step (gradient)",
    },
}


def main() -> None:
    # --- Extract coefficients from source ---
    alpha = _extract(SRC / "planners" / "periodic_replan.py", "_RISK_ALPHA")
    beta = _extract(SRC / "planners" / "aggressive_replan.py", "_RISK_BETA")
    gamma = _extract(SRC / "planners" / "incremental_astar.py", "_RISK_GAMMA")
    delta = _extract(SRC / "planners" / "apf.py", "_RISK_DELTA")

    PLANNER_META["periodic_replan"]["coeff_label"] = rf"$\alpha{{=}}{alpha}$"
    PLANNER_META["periodic_replan"]["cost_formula"] = rf"$1 + {alpha} \cdot r(c)$"
    PLANNER_META["aggressive_replan"]["coeff_label"] = rf"$\beta{{=}}{beta}$"
    PLANNER_META["aggressive_replan"]["cost_formula"] = rf"$1 + {beta} \cdot r(c)$"
    PLANNER_META["incremental_astar"]["coeff_label"] = rf"$\gamma{{=}}{gamma}$"
    PLANNER_META["incremental_astar"]["cost_formula"] = rf"$1 + {gamma} \cdot r(c)$"
    PLANNER_META["apf"]["coeff_label"] = rf"$\delta{{=}}{delta}$"
    PLANNER_META["apf"]["cost_formula"] = rf"$F_{{\text{{rep}}}} \propto {delta} \cdot r(c)$"

    print(f"Coefficients: alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}")

    # --- Load results ---
    if not CSV.exists():
        print("ERROR: Run `python scripts/run_paper_experiments.py` first.")
        sys.exit(1)

    df = pd.read_csv(CSV)
    df = df[df["planner_id"] != "dstar_lite"].copy()

    if "infeasible" in df.columns:
        feasible = df[df["infeasible"] != True].copy()  # noqa: E712
    else:
        feasible = df.copy()

    # --- Compute per-planner stats ---
    rows = []
    for pid in PLANNER_ORDER:
        ps = feasible[feasible["planner_id"] == pid]
        sr = ps["success"].mean() * 100

        # Normalized mission score (per-scenario normalization, then averaged)
        mean_ms_by_scen = (
            feasible.groupby(["scenario_id", "planner_id"])["mission_score"]
            .mean()
            .unstack("planner_id")
            .reindex(columns=PLANNER_ORDER)
        )
        scenario_max = mean_ms_by_scen.max(axis=1)
        norm_ms = mean_ms_by_scen.div(scenario_max.replace(0, 1), axis=0)
        avg_norm_ms = norm_ms.mean(axis=0)

        rows.append({
            "pid": pid,
            "label": PLANNER_SHORT[pid],
            "sr": sr,
            "ms": avg_norm_ms[pid],
        })

    # --- Compute ranks ---
    sr_sorted = sorted(rows, key=lambda r: -r["sr"])
    for rank, r in enumerate(sr_sorted, 1):
        r["nav_rank"] = rank

    ms_sorted = sorted(rows, key=lambda r: -r["ms"])
    for rank, r in enumerate(ms_sorted, 1):
        r["ms_rank"] = rank

    for r in rows:
        r["delta"] = r["nav_rank"] - r["ms_rank"]

    # --- Print summary ---
    print(f"\n{'Planner':<15} {'SR%':>6} {'NormMS':>7} {'NavR':>5} {'MsR':>5} {'Delta':>6}")
    print("-" * 50)
    for r in rows:
        stars = abs(r["delta"]) * "*"
        print(f"{r['label']:<15} {r['sr']:6.1f} {r['ms']:7.3f} "
              f"{r['nav_rank']:5d} {r['ms_rank']:5d} {r['delta']:+6d} {stars}")

    # --- Generate LaTeX ---
    tex_rows = []
    for r in rows:
        pid = r["pid"]
        meta = PLANNER_META[pid]
        delta_val = r["delta"]
        n_stars = abs(delta_val)
        stars_tex = r"\star" * n_stars if n_stars > 0 else ""
        if delta_val != 0:
            delta_str = f"${delta_val:+d}$ ${stars_tex}$"
        else:
            delta_str = "$0$"

        # Row color for inversions
        row_prefix = ""
        if delta_val != 0:
            row_prefix = r"\rowcolor{yellow!12}"

        line = (
            f"{row_prefix}"
            f"{r['label']} & {meta['family']} & {meta['coeff_label']} & "
            f"{meta['replan']} & "
            f"{r['sr']:.1f} & {r['ms']:.2f} & "
            f"{r['nav_rank']} & {r['ms_rank']} & "
            f"{delta_str} \\\\"
        )
        tex_rows.append(line)

    tex = r"""\begin{table*}[t]
\centering
\caption{Planner profiles. The $\Delta$ column reveals the paper's central finding:
  the best navigator (Incr.\ A*, rank \#1) ranks only \#4 on mission effectiveness.
  The best rescuer (Aggressive, mission rank \#1) ranks \#4 on navigation.
  $\star$ symbols scale with $|\Delta|$: one star per rank shift.}
\label{tab:planner_profiles}
\footnotesize
\begin{tabular}{@{}llclrrrrc@{}}
\toprule
Planner & Family & Risk Coeff. & Replan trigger & SR\% & MS & Nav.R & Ms.R & $\Delta$ Rank \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}
\vspace{2pt}
\begin{flushleft}
\scriptsize $r(c) \in [0,1]$ from fire/smoke/traffic/debris proximity.
$\Delta$~Rank = Nav.\ Rank $-$ Mission Rank.
$\star$ = ranking inversion.
SR = feasible success rate, MS = normalized mission score (per-scenario normalization, then averaged).
\end{flushleft}
\end{table*}"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "planner_profiles.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
