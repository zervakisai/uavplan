#!/usr/bin/env python3
"""Analyse the unified-ρ factorial sweep for the reviewer response.

Produces, from outputs/rho_sweep/rho_sweep.csv:
  (0) Baseline SR/termination reproduction check (risk_rho=None diagonal).
  (1) Comment 5 — per-scenario feasible-SR ranking vs mission-score ranking
      (5 planners, baseline) + aggregate ranking inversion, under the corrected
      fire-coupled scoring. Flags per-scenario agreement vs disagreement.
  (2) Comment 1 — factorial ρ-response: SR(ρ) and MS(ρ) per planner per scenario,
      and, at each matched ρ, the SR-ranking vs MS-ranking of the swept planners
      → whether the inversion persists when algorithmic structure is held fixed.
  (3) Spearman(ρ, SR) and Spearman(ρ, MS) pooled over the sweep.
  (4) Comment 2 — mission-model sensitivity by post-hoc re-scoring from
      task_events_json (vary κ, λ-scale, decay horizon T) → whether the
      argmax-SR vs argmax-MS ordering flips under plausible alternate models.

Writes tables to outputs/rho_sweep/analysis/.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from flare.metrics.compute import (
    medication_efficacy, surveillance_value, triage_value,
)

CSV = "outputs/rho_sweep/rho_sweep.csv"
CSV_NEW = "outputs/rho_sweep/rho_sweep_new.csv"  # additional risk-paradigm planners
OUT = "outputs/rho_sweep/analysis"
SWEEP_PLANNERS = ["periodic_replan", "aggressive_replan", "incremental_astar", "apf"]
ALL_PLANNERS = ["astar"] + SWEEP_PLANNERS
SCENARIOS = {
    "osm_penteli_pharma_delivery_medium": "Penteli (pharma)",
    "osm_piraeus_urban_rescue_medium": "Piraeus (SAR)",
    "osm_downtown_fire_surveillance_medium": "Downtown (survey)",
}
# Expected baseline termination mix (memory, 2026-04-22 bit-identical re-run).
EXPECTED_BASELINE_TERM = {
    "success": 168, "fire_caught": 98, "vehicle_collision": 90,
    "goal_stall": 58, "infeasible": 35, "debris_caught": 1,
}


def feasible_sr(sub: pd.DataFrame) -> float:
    f = sub[~sub["infeasible"].astype(bool)]
    return float(f["success"].astype(bool).mean()) if len(f) else float("nan")


def mean_ms(sub: pd.DataFrame) -> float:
    return float(sub["mission_score"].mean()) if len(sub) else float("nan")


def _rank_desc(d: dict) -> dict:
    """Rank keys by value descending → {key: rank(1=best)}."""
    order = sorted(d, key=lambda k: (-d[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def report_baseline_reproduction(base: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("(0) BASELINE (risk_rho=None) SR / TERMINATION REPRODUCTION")
    print("=" * 74)
    print(f"baseline rows: {len(base)} (expect 450 = 5 planners x 3 scen x 30 seeds)")
    term = base["termination_reason"].value_counts().to_dict()
    print(f"{'termination':20s} {'got':>5s} {'expected':>9s}")
    for k in sorted(set(term) | set(EXPECTED_BASELINE_TERM)):
        print(f"{k:20s} {term.get(k,0):5d} {EXPECTED_BASELINE_TERM.get(k,0):9d}")
    match = all(term.get(k, 0) == v for k, v in EXPECTED_BASELINE_TERM.items())
    print(f"  → termination mix matches old run: {match}")


def per_scenario_ranking(base: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 74)
    print("(1) COMMENT 5 — per-scenario feasible-SR vs mission-score ranking (baseline)")
    print("=" * 74)
    rows = []
    for sid, label in SCENARIOS.items():
        s = base[base["scenario_id"] == sid]
        sr = {p: feasible_sr(s[s["planner_id"] == p]) for p in ALL_PLANNERS}
        ms = {p: mean_ms(s[s["planner_id"] == p]) for p in ALL_PLANNERS}
        sr_rank, ms_rank = _rank_desc(sr), _rank_desc(ms)
        nav_best = min(sr_rank, key=sr_rank.get)
        ms_best = min(ms_rank, key=ms_rank.get)
        agree = nav_best == ms_best
        print(f"\n{label}  [{sid}]")
        print(f"  {'planner':18s} {'feasSR%':>8s} {'navR':>5s} {'MS':>8s} {'msR':>5s}")
        for p in ALL_PLANNERS:
            print(f"  {p:18s} {100*sr[p]:8.1f} {sr_rank[p]:5d} {ms[p]:8.3f} {ms_rank[p]:5d}")
        print(f"  nav-best={nav_best}  mission-best={ms_best}  "
              f"→ {'AGREE (no inversion)' if agree else 'INVERSION'}")
        for p in ALL_PLANNERS:
            rows.append({"scenario": label, "planner": p, "feasible_sr": sr[p],
                         "mission_score": ms[p], "nav_rank": sr_rank[p],
                         "ms_rank": ms_rank[p], "nav_best": nav_best,
                         "ms_best": ms_best, "inversion": not agree})
    # Aggregate: normalised MS averaged across scenarios (paper Table 1 method)
    print("\n--- AGGREGATE (normalised MS averaged across scenarios) ---")
    agg_sr, agg_ms = {}, {}
    for p in ALL_PLANNERS:
        agg_sr[p] = np.nanmean([feasible_sr(base[(base.scenario_id == sid) &
                     (base.planner_id == p)]) for sid in SCENARIOS])
        norm = []
        for sid in SCENARIOS:
            s = base[base.scenario_id == sid]
            mx = max(mean_ms(s[s.planner_id == q]) for q in ALL_PLANNERS) or 1.0
            norm.append(mean_ms(s[s.planner_id == p]) / mx if mx else 0.0)
        agg_ms[p] = float(np.mean(norm))
    sr_rank, ms_rank = _rank_desc(agg_sr), _rank_desc(agg_ms)
    print(f"  {'planner':18s} {'feasSR%':>8s} {'navR':>5s} {'normMS':>8s} {'msR':>5s}")
    for p in ALL_PLANNERS:
        print(f"  {p:18s} {100*agg_sr[p]:8.1f} {sr_rank[p]:5d} {agg_ms[p]:8.3f} {ms_rank[p]:5d}")
    nav_best, ms_best = min(sr_rank, key=sr_rank.get), min(ms_rank, key=ms_rank.get)
    print(f"  aggregate nav-best={nav_best}  mission-best={ms_best} "
          f"→ {'AGREE' if nav_best==ms_best else 'INVERSION'}")
    return pd.DataFrame(rows)


def per_planner_sensitivity(fac: pd.DataFrame) -> pd.DataFrame:
    """Headline for Comment 1: HOW MUCH does ρ affect each planner (SR and MS)?"""
    print("\n" + "=" * 74)
    print("(2b) PER-PLANNER ρ-SENSITIVITY — how much ρ affects each planner")
    print("=" * 74)
    order = ["periodic_replan", "aggressive_replan", "incremental_astar", "apf",
             "chance_constrained", "cvar_conservative", "risk_sensitive", "risk_aware_rrt"]
    planners = [p for p in order if p in set(fac.planner_id)]
    planners += [p for p in fac.planner_id.unique() if p not in planners]
    rhos = sorted(fac["risk_rho"].astype(float).unique())
    hdr = (f"{'planner':20s} | {'Spear(ρ,SR)':>15s} | {'Spear(ρ,MS)':>15s} | "
           f"{'SR span':>17s} | {'MS span':>17s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for p in planners:
        d = fac[fac.planner_id == p].copy(); d["rho"] = d["risk_rho"].astype(float)
        feas = d[~d["infeasible"].astype(bool)]
        if feas["success"].astype(bool).nunique() > 1:
            rs, ps = spearmanr(feas["rho"], feas["success"].astype(float))
        else:
            rs, ps = float("nan"), float("nan")
        rm, pm = spearmanr(d["rho"], d["mission_score"].astype(float))
        sr_by = [feasible_sr(d[d.rho == r]) for r in rhos]
        ms_by = [mean_ms(d[d.rho == r]) for r in rhos]
        srspan = f"{100*min(sr_by):.0f}->{100*max(sr_by):.0f}%(D{100*(max(sr_by)-min(sr_by)):.0f})"
        msspan = f"{min(ms_by):.2f}->{max(ms_by):.2f}(D{max(ms_by)-min(ms_by):.2f})"
        print(f"{p:20s} | {rs:+7.3f}(p={ps:.1g}) | {rm:+7.3f}(p={pm:.1g}) | {srspan:>17s} | {msspan:>17s}")
        rows.append({"planner": p, "spearman_rho_sr": rs, "p_sr": ps,
                     "spearman_rho_ms": rm, "p_ms": pm,
                     "sr_min": min(sr_by), "sr_max": max(sr_by),
                     "ms_min": min(ms_by), "ms_max": max(ms_by),
                     "ms_span": max(ms_by) - min(ms_by)})
    print("\nMEAN MS BY ρ:")
    print(f"{'planner':20s} | " + " | ".join(f"ρ={r:>5.0f}" for r in rhos))
    for p in planners:
        d = fac[fac.planner_id == p].copy(); d["rho"] = d["risk_rho"].astype(float)
        print(f"{p:20s} | " + " | ".join(f"{mean_ms(d[d.rho==r]):8.2f}" for r in rhos))
    return pd.DataFrame(rows)


def factorial_rho_response(fac: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 74)
    print("(2) COMMENT 1 — factorial ρ-response + inversion persistence at matched ρ")
    print("=" * 74)
    rhos = sorted(fac["risk_rho"].astype(float).unique())
    rows = []
    for sid, label in SCENARIOS.items():
        s = fac[fac["scenario_id"] == sid]
        print(f"\n{label}")
        print(f"  {'ρ':>4s} | " + " | ".join(f"{p[:10]:>10s}" for p in SWEEP_PLANNERS)
              + "   || SR-best / MS-best  → verdict")
        for rho in rhos:
            r = s[s["risk_rho"].astype(float) == rho]
            sr = {p: feasible_sr(r[r.planner_id == p]) for p in SWEEP_PLANNERS}
            ms = {p: mean_ms(r[r.planner_id == p]) for p in SWEEP_PLANNERS}
            nav_best = max(sr, key=lambda k: (sr[k] if sr[k] == sr[k] else -1))
            ms_best = max(ms, key=lambda k: (ms[k] if ms[k] == ms[k] else -1))
            inv = nav_best != ms_best
            cells = " | ".join(f"{100*sr[p]:4.0f}/{ms[p]:4.2f}" for p in SWEEP_PLANNERS)
            print(f"  {rho:4.0f} | {cells}   || {nav_best[:8]}/{ms_best[:8]} "
                  f"→ {'INV' if inv else 'agree'}")
            for p in SWEEP_PLANNERS:
                rows.append({"scenario": label, "rho": rho, "planner": p,
                             "feasible_sr": sr[p], "mission_score": ms[p],
                             "nav_best": nav_best, "ms_best": ms_best, "inversion": inv})
    return pd.DataFrame(rows)


def spearman_report(fac: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("(3) SPEARMAN(ρ, SR) and SPEARMAN(ρ, MS) pooled over the sweep")
    print("=" * 74)
    for sid, label in list(SCENARIOS.items()) + [(None, "ALL scenarios")]:
        s = fac if sid is None else fac[fac.scenario_id == sid]
        # per-episode: rho vs success(0/1), rho vs mission_score
        feas = s[~s["infeasible"].astype(bool)]
        rho = feas["risk_rho"].astype(float)
        if feas["success"].astype(bool).nunique() > 1:
            rs, ps = spearmanr(rho, feas["success"].astype(float))
        else:
            rs, ps = float("nan"), float("nan")
        rm, pm = spearmanr(rho, feas["mission_score"].astype(float))
        print(f"  {label:22s}  ρ~SR: ρ_s={rs:+.3f} (p={ps:.3g})   "
              f"ρ~MS: ρ_s={rm:+.3f} (p={pm:.3g})")


def _events_from_json(js: str) -> list[dict]:
    try:
        evs = json.loads(js) if js else []
    except Exception:
        evs = []
    return [{"type": "task_completed", "step_idx": e.get("step_idx", 2000),
             "weight": e.get("weight", 1.0), "d_fire": e.get("d_fire", 999.0)}
            for e in evs]


def _rescore(row, kappa=5.0, lam=1.0, horizon=800) -> float:
    evs = _events_from_json(row["task_events_json"])
    mt = row["mission_type"]
    if mt == "urban_rescue":
        return triage_value(evs, kappa=kappa, lambda_scale=lam)
    if mt == "pharma_delivery":
        step = evs[0]["step_idx"] if evs else 2000
        return medication_efficacy(step, decay_horizon=horizon, max_steps=2000)
    if mt == "fire_surveillance":
        return surveillance_value(evs, decay_horizon=horizon, max_steps=2000)
    return 0.0


def mission_sensitivity(base: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("(4) COMMENT 2 — mission-model sensitivity (post-hoc re-score, baseline)")
    print("=" * 74)
    grids = {
        "urban_rescue κ": [dict(kappa=k) for k in (0.0, 2.0, 5.0, 10.0)],
        "urban_rescue λ-scale": [dict(lam=l) for l in (0.5, 1.0, 2.0)],
        "pharma/survey horizon T": [dict(horizon=h) for h in (400, 800, 1200, 2000)],
    }
    for gname, variants in grids.items():
        print(f"\n  {gname}: does argmax-MS planner (per scenario) change?")
        for sid, label in SCENARIOS.items():
            s = base[base.scenario_id == sid].copy()
            line = []
            for v in variants:
                s["_ms"] = s.apply(lambda r: _rescore(r, **v), axis=1)
                ms = {p: float(s[s.planner_id == p]["_ms"].mean()) for p in ALL_PLANNERS}
                best = max(ms, key=ms.get)
                tag = ",".join(f"{k}={val}" for k, val in v.items())
                line.append(f"{tag}:{best[:8]}")
            print(f"    {label:20s} " + "  ".join(line))


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(CSV)
    if os.path.exists(CSV_NEW):
        df = pd.concat([df, pd.read_csv(CSV_NEW)], ignore_index=True)
    df["is_baseline"] = df["risk_rho"].astype(str) == "none"
    base = df[df.is_baseline]
    fac = df[~df.is_baseline].copy()
    print(f"Loaded {len(df)} rows  (baseline={len(base)}, factorial={len(fac)})")
    print(f"factorial planners: {sorted(fac.planner_id.unique())}")

    report_baseline_reproduction(base)
    t1 = per_scenario_ranking(base)
    t1.to_csv(os.path.join(OUT, "per_scenario_ranking.csv"), index=False)
    ts = per_planner_sensitivity(fac)
    ts.to_csv(os.path.join(OUT, "per_planner_sensitivity.csv"), index=False)
    t2 = factorial_rho_response(fac)
    t2.to_csv(os.path.join(OUT, "factorial_rho_response.csv"), index=False)
    spearman_report(fac)
    mission_sensitivity(base)
    print(f"\nTables written to {OUT}/")


if __name__ == "__main__":
    main()
