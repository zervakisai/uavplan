#!/usr/bin/env python3
"""Generate ready-to-paste paper tables for the revision (markdown + LaTeX booktabs).

  Table 1 (regenerated): planner profiles — full 5-planner suite (Aggressive restored),
           baseline (ρ=None diagonal) feasible SR, normalised mission score M, ranks.
  Table S (new): per-planner ρ-sensitivity across all 8 swept planners / 5 paradigms.

Writes outputs/rho_sweep/analysis/paper_tables.md
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = "outputs/rho_sweep/rho_sweep.csv"
CSV_NEW = "outputs/rho_sweep/rho_sweep_new.csv"
OUT = "outputs/rho_sweep/analysis/paper_tables.md"
SCEN = ["osm_penteli_pharma_delivery_medium", "osm_piraeus_urban_rescue_medium",
        "osm_downtown_fire_surveillance_medium"]

META = {  # planner: (label, family, reported ρ, replan trigger)
    "astar": ("A*", "Graph search (static)", "0", "none"),
    "periodic_replan": ("Periodic", "Soft cost-inflation", "5", "every 6 steps"),
    "aggressive_replan": ("Aggressive", "Soft cost-inflation", "0.5", "mask change"),
    "incremental_astar": ("Incr. A*", "Soft cost-inflation", "2", "path blocked"),
    "apf": ("APF", "Potential field", "3", "per-step gradient"),
    "chance_constrained": ("Chance-Constr.", "Hard threshold", "swept", "mask change"),
    "cvar_conservative": ("CVaR", "Worst-case / CVaR", "swept", "mask change"),
    "risk_sensitive": ("Risk-Sensitive", "Exponential / entropic", "swept", "mask change"),
    "risk_aware_rrt": ("RRT*", "Sampling", "swept", "mask change"),
}
BASE5 = ["astar", "periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
SWEEP8 = ["periodic_replan", "aggressive_replan", "incremental_astar",
          "chance_constrained", "cvar_conservative", "risk_sensitive",
          "risk_aware_rrt", "apf"]


def _fsr(s):
    f = s[~s["infeasible"].astype(bool)]
    return f["success"].astype(float).mean() if len(f) else np.nan


def main():
    base = pd.read_csv(CSV)
    base = base[base["risk_rho"].astype(str) == "none"]
    fac = pd.read_csv(CSV)
    if os.path.exists(CSV_NEW):
        fac = pd.concat([fac, pd.read_csv(CSV_NEW)], ignore_index=True)
    fac = fac[fac["risk_rho"].astype(str) != "none"].copy()
    fac["rho"] = fac["risk_rho"].astype(float)

    # --- Table 1: aggregate feasible SR + normalised M over the 5 planners ---
    sr = {p: np.mean([_fsr(base[(base.scenario_id == s) & (base.planner_id == p)])
                      for s in SCEN]) for p in BASE5}
    normms = {}
    for p in BASE5:
        vals = []
        for s in SCEN:
            sub = base[base.scenario_id == s]
            mx = max(sub[sub.planner_id == q]["mission_score"].mean() for q in BASE5) or 1.0
            vals.append(sub[sub.planner_id == p]["mission_score"].mean() / mx if mx else 0.0)
        normms[p] = float(np.mean(vals))
    navr = {p: r + 1 for r, p in enumerate(sorted(BASE5, key=lambda k: -sr[k]))}
    msr = {p: r + 1 for r, p in enumerate(sorted(BASE5, key=lambda k: -normms[k]))}

    lines = ["# Regenerated paper tables (revision)\n"]
    lines.append("## Table 1. Planner profiles and ranking discrepancy "
                 "(baseline configuration; Aggressive Replan restored).\n")
    lines.append("| Planner | Family | ρ | Replan trigger | Feas. SR (%) | M (norm.) | Nav. rank | Miss. rank |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in BASE5:
        lab, fam, rho, trig = META[p]
        lines.append(f"| {lab} | {fam} | {rho} | {trig} | {100*sr[p]:.1f} | "
                     f"{normms[p]:.2f} | {navr[p]} | {msr[p]} |")
    nav_best = min(navr, key=navr.get); ms_best = min(msr, key=msr.get)
    lines.append(f"\n*Ranking inversion: navigation-best = **{META[nav_best][0]}** "
                 f"(rank 1 SR) ≠ mission-best = **{META[ms_best][0]}** (rank 1 M).*\n")

    # --- Table S: per-planner ρ-sensitivity, 8 planners ---
    lines.append("## Table S (new). Per-planner ρ-sensitivity "
                 "(factorial sweep; 3 scenarios × 30 seeds per ρ).\n")
    lines.append("| Planner | Paradigm | Spearman(ρ, SR) | Spearman(ρ, M) | M at ρ=0 → ρ=10 |")
    lines.append("|---|---|---|---|---|")
    srows = []
    for p in SWEEP8:
        d = fac[fac.planner_id == p]
        feas = d[~d["infeasible"].astype(bool)]
        rs, ps = spearmanr(feas["rho"], feas["success"].astype(float))
        rm, pm = spearmanr(d["rho"], d["mission_score"].astype(float))
        m0 = d[d.rho == 0]["mission_score"].mean(); m10 = d[d.rho == 10]["mission_score"].mean()
        srows.append((p, rs, ps, rm, pm, m0, m10))
    srows.sort(key=lambda t: t[3])  # most negative ρ~M first
    for p, rs, ps, rm, pm, m0, m10 in srows:
        star = "*" if pm < 0.05 else ""
        lines.append(f"| {META[p][0]} | {META[p][1]} | {rs:+.2f} | {rm:+.2f}{star} | "
                     f"{m0:.2f} → {m10:.2f} |")
    lines.append("\n*\\* p < 0.05. ρ erodes mission value strongly in the CVaR and "
                 "Incremental-A* planners, moderately in the exponential planner, and "
                 "negligibly in the potential-field, hard-threshold and frequent-replan "
                 "planners — i.e. the effect of ρ is paradigm-dependent.*\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWritten: {OUT}")


if __name__ == "__main__":
    main()
