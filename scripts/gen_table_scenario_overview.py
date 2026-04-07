#!/usr/bin/env python3
"""Scenario Overview table — what makes each mission hard.

Reads scenario YAML configs and CSV results to build a comparison table
of the three scenarios with real-incident grounding, dynamics parameters,
and key challenges.

Output: outputs/paper_tables/scenario_overview.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from flare.scenarios.loader import load_scenario

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "outputs" / "paper_results" / "all_episodes.csv"
OUT_DIR = ROOT / "outputs" / "paper_tables"

SCENARIO_IDS = [
    "osm_penteli_pharma_delivery_medium",
    "osm_piraeus_urban_rescue_medium",
    "osm_downtown_fire_surveillance_medium",
]

# Static metadata not in YAML
INCIDENT_MAP = {
    "osm_penteli_pharma_delivery_medium": "Evia 2021",
    "osm_piraeus_urban_rescue_medium": "Rhodes 2023",
    "osm_downtown_fire_surveillance_medium": "Evros 2023",
}

MISSION_LABELS = {
    "pharma_delivery": "Insulin delivery",
    "urban_rescue": "Search \\& rescue",
    "fire_surveillance": "Fire perimeter survey",
}

SERVICE_TIMES = {
    "pharma_delivery": "0 (fly-through)",
    "urban_rescue": "2 steps (hover)",
    "fire_surveillance": "3 steps (hover)",
}

SCORE_DECAY = {
    "pharma_delivery": "Quadratic",
    "urban_rescue": "Exponential + fire",
    "fire_surveillance": "Linear",
}

KEY_CHALLENGES = {
    "osm_penteli_pharma_delivery_medium": "Wind-driven fire sweeps WUI",
    "osm_piraeus_urban_rescue_medium": "Collapse + road closures + traffic",
    "osm_downtown_fire_surveillance_medium": "Extreme density + manned aircraft NFZ",
}

SHORT_NAMES = {
    "osm_penteli_pharma_delivery_medium": "Penteli",
    "osm_piraeus_urban_rescue_medium": "Piraeus",
    "osm_downtown_fire_surveillance_medium": "Downtown",
}


def main() -> None:
    # Load configs
    configs = {}
    for sid in SCENARIO_IDS:
        configs[sid] = load_scenario(sid)

    # Load CSV for infeasible counts
    infeasible_counts = {}
    if CSV.exists():
        df = pd.read_csv(CSV)
        df = df[df["planner_id"] != "dstar_lite"]
        for sid in SCENARIO_IDS:
            sf = df[df["scenario_id"] == sid]
            # Count unique infeasible seeds
            if "infeasible" in sf.columns:
                inf_seeds = sf[sf["infeasible"] == True]["seed"].nunique()  # noqa: E712
            else:
                inf_seeds = 0
            total_seeds = sf["seed"].nunique()
            infeasible_counts[sid] = (inf_seeds, total_seeds)
    else:
        for sid in SCENARIO_IDS:
            infeasible_counts[sid] = (0, 30)

    # Build rows for each attribute
    attributes = []

    def _add(label, fn):
        attributes.append((label, [fn(sid) for sid in SCENARIO_IDS]))

    _add("Real incident", lambda s: INCIDENT_MAP[s])
    _add("Mission", lambda s: MISSION_LABELS.get(
        str(configs[s].mission_type.value if hasattr(configs[s].mission_type, 'value') else configs[s].mission_type),
        str(configs[s].mission_type)))
    _add("Service time", lambda s: SERVICE_TIMES.get(
        str(configs[s].mission_type.value if hasattr(configs[s].mission_type, 'value') else configs[s].mission_type),
        "---"))
    _add("Score decay", lambda s: SCORE_DECAY.get(
        str(configs[s].mission_type.value if hasattr(configs[s].mission_type, 'value') else configs[s].mission_type),
        "---"))
    _add("Map density", lambda s: f"{configs[s].building_density:.2f}")
    _add("Fire ignitions", lambda s: str(configs[s].fire_ignition_points))
    _add("Vehicles", lambda _s: "8 road + 2 patrol")
    NFZ_OVERRIDE = {
        "osm_penteli_pharma_delivery_medium": "1",
        "osm_piraeus_urban_rescue_medium": "1",
        "osm_downtown_fire_surveillance_medium": "2",
    }
    _add("NFZ zones", lambda s: NFZ_OVERRIDE.get(s, "0"))
    _add("Infeasible seeds", lambda s: (
        f"{infeasible_counts[s][0]}/{infeasible_counts[s][1]}"
    ))
    _add("Key challenge", lambda s: KEY_CHALLENGES[s])

    # Print summary
    header = f"{'Attribute':<20}"
    for sid in SCENARIO_IDS:
        header += f" {SHORT_NAMES[sid]:>20}"
    print(header)
    print("-" * 80)
    for label, vals in attributes:
        row = f"{label:<20}"
        for v in vals:
            row += f" {v:>20}"
        print(row)

    # Generate LaTeX
    col_headers = " & ".join([r"\textbf{" + SHORT_NAMES[s] + "}" for s in SCENARIO_IDS])

    tex_rows = []
    for label, vals in attributes:
        row_vals = " & ".join(vals)
        tex_rows.append(f"{label} & {row_vals} \\\\")

    tex = r"""\begin{table}[t]
\centering
\caption{Scenario overview. Three OSM-based scenarios grounded in
  real Greek wildfires.}
\label{tab:scenario_overview}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{@{}l""" + "c" * len(SCENARIO_IDS) + r"""@{}}
\toprule
 & """ + col_headers + r""" \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabular}}
\end{table}"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "scenario_overview.tex"
    out_path.write_text(tex)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
