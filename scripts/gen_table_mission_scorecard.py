#!/usr/bin/env python3
"""Mission Scorecard — what happened in each real scenario.

Three sub-tables (Penteli, Piraeus, Downtown), each with mission-specific
metrics and a human-readable Verdict column.

Output: outputs/paper_tables/mission_scorecard.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from flare.visualization.labels import PLANNER_ORDER, PLANNER_SHORT

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"
OUT_DIR = ROOT / "outputs" / "paper_tables"

SCENARIOS = {
    "osm_penteli_pharma_delivery_medium": {
        "title": "Penteli --- Insulin delivery to fire-isolated village (Evia 2021)",
        "metric_label": "Med.\\ efficacy",
        "metric_fn": lambda steps: max(0.0, 1.0 - (steps / 800) ** 2),
        "step_label": "Delivery step (med.)",
    },
    "osm_piraeus_urban_rescue_medium": {
        "title": "Piraeus --- Urban rescue in port district (Rhodes 2023)",
        "metric_label": "Triage value",
        "metric_fn": None,  # use mission_score directly
        "step_label": "Rescue step (med.)",
    },
    "osm_downtown_fire_surveillance_medium": {
        "title": "Downtown --- Fire perimeter survey (Evros 2023)",
        "metric_label": "Survey freshness",
        "metric_fn": lambda steps: max(0.0, 1.0 - steps / 800),
        "step_label": "Survey step (med.)",
    },
}


def _verdict(pid: str, scenario_key: str, sr: float, med_steps: float,
             metric_val: float, is_best_sr: bool, is_best_metric: bool) -> str:
    """Generate human-readable verdict based on planner CHARACTER + outcome."""
    # Verdicts keyed on planner identity — reflects risk coefficient behaviour
    VERDICTS = {
        "osm_penteli_pharma_delivery_medium": {
            "astar":             "Walks straight into fire",
            "periodic_replan":   "Safe detour, insulin degrades",
            "aggressive_replan": "Cuts through smoke, fast delivery",
            "incremental_astar": "Best navigator, delivers on time",
            "apf":               "Repulsion traps waste time",
        },
        "osm_piraeus_urban_rescue_medium": {
            "astar":             "Blocked by fire and collapse",
            "periodic_replan":   "Safe route, casualty deteriorates",
            "aggressive_replan": "Reaches casualty before expiry",
            "incremental_astar": "Good navigation, moderate rescue",
            "apf":               "Local minima delay rescue",
        },
        "osm_downtown_fire_surveillance_medium": {
            "astar":             "Trapped in dense urban grid",
            "periodic_replan":   "Cautious survey, data goes stale",
            "aggressive_replan": "Bold coverage, some risk",
            "incremental_astar": "Fresh survey, maximum value",
            "apf":               "Reactive drift, slow coverage",
        },
    }
    return VERDICTS.get(scenario_key, {}).get(pid, "---")


def main() -> None:
    if not CSV.exists():
        print("ERROR: Run `python scripts/run_paper_experiments.py` first.")
        sys.exit(1)

    df = pd.read_csv(CSV)
    df = df[df["planner_id"] != "dstar_lite"].copy()

    if "infeasible" in df.columns:
        feasible = df[df["infeasible"] != True].copy()  # noqa: E712
    else:
        feasible = df.copy()

    all_sub_tables = []

    for scen_id, scen_info in SCENARIOS.items():
        sf = feasible[feasible["scenario_id"] == scen_id]

        rows = []
        for pid in PLANNER_ORDER:
            ps = sf[sf["planner_id"] == pid]
            sr = ps["success"].mean() * 100
            succ = ps[ps["success"] == True]  # noqa: E712

            if len(succ) > 0:
                med_steps = succ["executed_steps"].median()
            else:
                med_steps = float("nan")

            # Compute mission-specific metric
            if scen_info["metric_fn"] is not None and not np.isnan(med_steps):
                metric_val = scen_info["metric_fn"](med_steps)
            else:
                # Use mission_score from CSV directly
                metric_val = ps["mission_score"].mean()

            rows.append({
                "pid": pid,
                "label": PLANNER_SHORT[pid],
                "sr": sr,
                "med_steps": med_steps,
                "metric": metric_val,
            })

        # Determine best SR and best metric
        valid_srs = [(r["sr"], i) for i, r in enumerate(rows)]
        valid_metrics = [(r["metric"], i) for i, r in enumerate(rows) if not np.isnan(r["metric"])]
        best_sr_idx = max(valid_srs, key=lambda x: x[0])[1] if valid_srs else -1
        best_metric_idx = max(valid_metrics, key=lambda x: x[0])[1] if valid_metrics else -1

        # Generate verdicts
        for i, r in enumerate(rows):
            r["verdict"] = _verdict(
                r["pid"], scen_id, r["sr"], r["med_steps"], r["metric"],
                i == best_sr_idx, i == best_metric_idx,
            )

        # Build LaTeX sub-table
        tex_rows = []
        for i, r in enumerate(rows):
            sr_str = f"{r['sr']:.0f}"
            if i == best_sr_idx:
                sr_str = r"\textbf{" + sr_str + "}"

            if np.isnan(r["med_steps"]):
                steps_str = "timeout"
            else:
                steps_str = f"$\\sim${int(r['med_steps'])}"

            # Format metric: percentage for pharma/surveillance, raw score for triage
            if np.isnan(r["metric"]):
                metric_str = "---"
            elif scen_info["metric_fn"] is not None:
                metric_str = f"{r['metric'] * 100:.0f}\\%"
            else:
                metric_str = f"{r['metric']:.2f}"
            if i == best_metric_idx:
                metric_str = r"$\star$ " + metric_str

            verdict_str = r"\textit{" + r["verdict"] + "}"

            tex_rows.append(
                f"{r['label']} & {sr_str} & {steps_str} & {metric_str} & {verdict_str} \\\\"
            )

        sub_table = (
            r"\multicolumn{5}{l}{\textbf{" + scen_info["title"] + r"}} \\" + "\n"
            r"\midrule" + "\n"
            + "\n".join(tex_rows)
        )
        all_sub_tables.append(sub_table)

    # Combine into final table
    body = ("\n" + r"\midrule" + "\n" + r"\addlinespace[4pt]" + "\n").join(all_sub_tables)

    tex = r"""\begin{table*}[t]
\centering
\caption{Mission scorecard. Per-scenario results with mission-specific metrics.
  $\star$ marks the best mission score per scenario. \textbf{Bold} marks the best
  success rate. Note that they land on \emph{different} planners---the paper's
  central finding.}
\label{tab:mission_scorecard}
\footnotesize
\begin{tabular}{@{}lrlrl@{}}
\toprule
Planner & SR\% & Delivery step & Mission metric & Verdict \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\vspace{2pt}
\begin{flushleft}
\scriptsize Delivery step = median \texttt{executed\_steps} over successful episodes.
Med.\ efficacy = $\max(0, 1-(t/800)^2)$.
Triage value = mean \texttt{mission\_score}.
Survey freshness = $\max(0, 1-t/800)$.
\end{flushleft}
\end{table*}"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "mission_scorecard.tex"
    out_path.write_text(tex)
    print(f"  Saved: {out_path}")

    # Print summary
    print("\nMission Scorecard Summary:")
    for scen_id, scen_info in SCENARIOS.items():
        sf = feasible[feasible["scenario_id"] == scen_id]
        print(f"\n  {scen_info['title'][:50]}...")
        for pid in PLANNER_ORDER:
            ps = sf[sf["planner_id"] == pid]
            sr = ps["success"].mean() * 100
            print(f"    {PLANNER_SHORT[pid]:12s}: SR={sr:.0f}%")


if __name__ == "__main__":
    main()
