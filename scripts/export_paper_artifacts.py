#!/usr/bin/env python3
"""Run paper protocol benchmarks and export publication artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from uavbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from uavbench.cli.benchmark import run_dynamic_episode
from uavbench.scenarios.registry import get_scenario_metadata, list_scenarios_by_track


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _scenario_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for p in sorted(Path("src/uavbench/scenarios/configs").glob("*.yaml")):
        checksums[p.stem] = hashlib.sha256(p.read_bytes()).hexdigest()
    return checksums


def _bootstrap_ci(values: list[float], confidence: float = 0.95, n_bootstrap: int = 2000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])

    rng = random.Random(42)
    means: list[float] = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(alpha * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1.0 - alpha) * n_bootstrap))
    return float(means[lo_idx]), float(means[hi_idx])


def _run_track(
    scenario_ids: list[str],
    planner_ids: list[str],
    seeds: list[int],
    output_dir: Path,
) -> None:
    runner = BenchmarkRunner(
        BenchmarkConfig(
            scenario_ids=scenario_ids,
            planner_ids=planner_ids,
            seeds=seeds,
            output_dir=output_dir,
            save_jsonl=True,
            save_csv=True,
            verbose=False,
        )
    )
    runner.run()


def _write_tex_table(csv_path: Path, out_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l r r}\\n")
        f.write("\\hline\\n")
        f.write("Scenario & Planner & Success(\\%) & PathLen \\\\\\n")
        f.write("\\hline\\n")
        for r in rows:
            success_pct = 100.0 * float(r.get("success_rate", 0.0))
            path_len = r.get("path_length_mean", "nan")
            f.write(f"{r['scenario_id']} & {r['planner_id']} & {success_pct:.1f} & {path_len} \\\\\\n")
        f.write("\\hline\\n")
        f.write("\\end{tabular}\\n")


def _paired_stats(agg_csv: Path, baseline: str = "astar") -> dict[str, Any]:
    rows = list(csv.DictReader(agg_csv.read_text(encoding="utf-8").splitlines()))
    by_planner: dict[str, dict[str, float]] = {}
    for r in rows:
        by_planner.setdefault(r["planner_id"], {})[r["scenario_id"]] = float(r["success_rate"])
    if baseline not in by_planner:
        return {}
    base = by_planner[baseline]

    stats: dict[str, Any] = {}
    for planner, vals in by_planner.items():
        if planner == baseline:
            continue
        common = sorted(set(base).intersection(vals))
        if not common:
            continue
        x = [vals[sid] for sid in common]
        y = [base[sid] for sid in common]
        diffs = [a - b for a, b in zip(x, y)]
        mean_diff = sum(diffs) / max(len(diffs), 1)
        gt = sum(1 for d in diffs if d > 0)
        lt = sum(1 for d in diffs if d < 0)
        effect = (gt - lt) / max((gt + lt), 1)

        p_value = None
        test_name = "wilcoxon_unavailable"
        try:
            from scipy.stats import wilcoxon  # type: ignore

            stat = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
            p_value = float(stat.pvalue)
            test_name = "wilcoxon"
        except Exception:
            n = gt + lt
            if n > 0:
                z = abs(gt - n / 2.0) / math.sqrt(max(n * 0.25, 1e-9))
                p_value = float(min(1.0, 2.0 * math.exp(-0.717 * z - 0.416 * z * z)))
            else:
                p_value = 1.0
            test_name = "sign_test_approx"

        stats[planner] = {
            "n_pairs": len(common),
            "mean_success_diff_vs_astar": round(mean_diff, 4),
            "effect_size_rank_biserial": round(effect, 4),
            "test": test_name,
            "p_value": round(float(p_value), 6),
        }
    return stats


def _mission_effect_sizes(agg_csv: Path, baseline: str = "astar") -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(agg_csv.read_text(encoding="utf-8").splitlines()))
    by_mission: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        sid = r["scenario_id"]
        meta = get_scenario_metadata(sid)
        mission = meta.mission_type.value if meta is not None else "unknown"
        by_mission[mission][r["planner_id"]][sid] = float(r["success_rate"])

    out: dict[str, dict[str, float]] = {}
    for mission, planner_map in by_mission.items():
        if baseline not in planner_map:
            continue
        base = planner_map[baseline]
        mission_out: dict[str, float] = {}
        for planner_id, vals in planner_map.items():
            if planner_id == baseline:
                continue
            common = sorted(set(base).intersection(vals))
            if not common:
                continue
            diffs = [vals[s] - base[s] for s in common]
            mission_out[planner_id] = round(sum(diffs) / len(diffs), 4)
        if mission_out:
            out[mission] = mission_out
    return out


def _run_variant_rows(
    scenario_ids: list[str],
    planners: list[str],
    seeds: list[int],
    variant: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in scenario_ids:
        for pid in planners:
            for seed in seeds:
                res = run_dynamic_episode(sid, pid, seed=seed, protocol_variant=variant)
                rows.append(
                    {
                        "scenario_id": sid,
                        "planner_id": pid,
                        "seed": int(seed),
                        "variant": variant,
                        "success": 1.0 if bool(res.get("success", False)) else 0.0,
                        "path_length": float(res.get("path_length", 0.0)),
                        "total_replans": float(res.get("total_replans", 0.0)),
                        "initial_budget_exceeded": 1.0 if bool(res.get("initial_budget_exceeded", False)) else 0.0,
                        "replan_budget_violations": float(res.get("replan_budget_violations", 0.0)),
                        "dynamic_block_hits": float(res.get("dynamic_block_hits", 0.0)),
                        "risk_exposure_integral": float(res.get("risk_exposure_integral", 0.0)),
                        "time_to_recover_after_break": float(res.get("time_to_recover_after_break", float("nan"))),
                        "replan_latency_after_break": float(res.get("replan_latency_after_break", float("nan"))),
                        "interaction_fire_nfz_overlap_ratio": float(res.get("interaction_fire_nfz_overlap_ratio", 0.0)),
                        "interaction_fire_road_closure_rate": float(res.get("interaction_fire_road_closure_rate", 0.0)),
                        "interaction_congestion_risk_corr": float(res.get("interaction_congestion_risk_corr", 0.0)),
                        "dynamic_block_entropy": float(res.get("dynamic_block_entropy", 0.0)),
                        "interdiction_hit_rate": float(res.get("interdiction_hit_rate", 0.0)),
                    }
                )
    return rows


def _aggregate_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["scenario_id"], row["planner_id"], row["variant"])].append(row)

    out: list[dict[str, Any]] = []
    for (sid, pid, variant), rs in sorted(grouped.items()):
        succ = [float(r["success"]) for r in rs]
        path_vals = [float(r["path_length"]) for r in rs if float(r["success"]) > 0.0]
        replans = [float(r["total_replans"]) for r in rs]
        budget_flags = [float(r["initial_budget_exceeded"]) for r in rs]
        repl_budget = [float(r["replan_budget_violations"]) for r in rs]
        dyn_hits = [float(r["dynamic_block_hits"]) for r in rs]
        risk_int = [float(r["risk_exposure_integral"]) for r in rs]
        recover_vals = [float(r["time_to_recover_after_break"]) for r in rs if not math.isnan(float(r["time_to_recover_after_break"]))]
        lat_vals = [float(r["replan_latency_after_break"]) for r in rs if not math.isnan(float(r["replan_latency_after_break"]))]
        ci_lo, ci_hi = _bootstrap_ci(succ, confidence=0.95)
        out.append(
            {
                "scenario_id": sid,
                "planner_id": pid,
                "variant": variant,
                "num_seeds": len(rs),
                "success_rate": round(sum(succ) / max(len(succ), 1), 6),
                "success_rate_ci_low": round(ci_lo, 6),
                "success_rate_ci_high": round(ci_hi, 6),
                "path_length_mean": round(sum(path_vals) / max(len(path_vals), 1), 4) if path_vals else float("nan"),
                "replans_mean": round(sum(replans) / max(len(replans), 1), 4),
                "dynamic_block_hits_mean": round(sum(dyn_hits) / max(len(dyn_hits), 1), 4),
                "risk_exposure_integral_mean": round(sum(risk_int) / max(len(risk_int), 1), 6),
                "time_to_recover_after_break_mean": round(sum(recover_vals) / max(len(recover_vals), 1), 6) if recover_vals else float("nan"),
                "replan_latency_after_break_mean": round(sum(lat_vals) / max(len(lat_vals), 1), 6) if lat_vals else float("nan"),
                "budget_violation_rate": round(sum(budget_flags) / max(len(budget_flags), 1), 6),
                "replan_budget_violations_mean": round(sum(repl_budget) / max(len(repl_budget), 1), 4),
                "interaction_fire_nfz_overlap_ratio_mean": round(
                    sum(float(r["interaction_fire_nfz_overlap_ratio"]) for r in rs) / max(len(rs), 1), 6
                ),
                "interaction_fire_road_closure_rate_mean": round(
                    sum(float(r["interaction_fire_road_closure_rate"]) for r in rs) / max(len(rs), 1), 6
                ),
                "interaction_congestion_risk_corr_mean": round(
                    sum(float(r["interaction_congestion_risk_corr"]) for r in rs) / max(len(rs), 1), 6
                ),
                "dynamic_block_entropy_mean": round(
                    sum(float(r["dynamic_block_entropy"]) for r in rs) / max(len(rs), 1), 6
                ),
                "interdiction_hit_rate_mean": round(
                    sum(float(r["interdiction_hit_rate"]) for r in rs) / max(len(rs), 1), 6
                ),
            }
        )
    return out


def _write_rows_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        else:
            writer = csv.DictWriter(f, fieldnames=["scenario_id", "planner_id"])
        writer.writeheader()
        writer.writerows(rows)


def _export_dynamic_event_views(
    scenario_ids: list[str],
    planners: list[str],
    seeds: list[int],
    replan_csv: Path,
    interaction_csv: Path,
) -> None:
    replan_csv.parent.mkdir(parents=True, exist_ok=True)
    repl_rows: list[dict[str, Any]] = []
    inter_rows: list[dict[str, Any]] = []
    for sid in scenario_ids:
        for pid in planners:
            for seed in seeds:
                res = run_dynamic_episode(sid, pid, seed=seed, protocol_variant="default")
                repl_rows.append(
                    {
                        "scenario": sid,
                        "planner": pid,
                        "seed": seed,
                        "total_replans": res.get("total_replans", 0),
                        "planner_total_replans_raw": res.get("planner_total_replans_raw", 0),
                        "replan_budget_violations": res.get("replan_budget_violations", 0),
                        "forced_replans_triggered": res.get("forced_replans_triggered", 0),
                        "interdictions_triggered": res.get("interdictions_triggered", 0),
                        "interdiction_hit_rate": res.get("interdiction_hit_rate", 0.0),
                        "termination_reason": res.get("termination_reason", "unknown"),
                    }
                )
                inter_rows.append(
                    {
                        "scenario": sid,
                        "planner": pid,
                        "seed": seed,
                        "dynamic_blocks": res.get("total_dynamic_blocks", 0),
                        "fire_blocks": res.get("fire_blocks", 0),
                        "traffic_blocks": res.get("traffic_blocks", 0),
                        "intruder_blocks": res.get("intruder_blocks", 0),
                        "dynamic_nfz_blocks": res.get("dynamic_nfz_blocks", 0),
                        "interaction_fire_nfz_overlap_ratio": res.get("interaction_fire_nfz_overlap_ratio", 0.0),
                        "interaction_fire_road_closure_rate": res.get("interaction_fire_road_closure_rate", 0.0),
                        "interaction_congestion_risk_corr": res.get("interaction_congestion_risk_corr", 0.0),
                        "dynamic_block_entropy": res.get("dynamic_block_entropy", 0.0),
                        "interdiction_hit_rate": res.get("interdiction_hit_rate", 0.0),
                        "dynamic_block_hits": res.get("dynamic_block_hits", 0.0),
                        "risk_exposure_integral": res.get("risk_exposure_integral", 0.0),
                        "time_to_recover_after_break": res.get("time_to_recover_after_break", float("nan")),
                        "replan_latency_after_break": res.get("replan_latency_after_break", float("nan")),
                        "reachability_failed_before_relax": res.get("reachability_failed_before_relax", False),
                        "corridor_fallback_used": res.get("corridor_fallback_used", False),
                        "feasible_after_guardrail": res.get("feasible_after_guardrail", True),
                        "success": res.get("success", False),
                    }
                )

    _write_rows_csv(repl_rows, replan_csv)
    _write_rows_csv(inter_rows, interaction_csv)


def _write_ablation_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l r r r}\\n")
        f.write("\\hline\\n")
        f.write("Variant & Planner & Success(\\%) & HitRate & Entropy \\\\\\n")
        f.write("\\hline\\n")
        for r in rows:
            f.write(
                f"{r['variant']} & {r['planner_id']} & {100.0 * float(r['success_rate']):.1f} & "
                f"{float(r.get('interdiction_hit_rate_mean', 0.0)):.2f} & "
                f"{float(r.get('dynamic_block_entropy_mean', 0.0)):.2f} \\\\\\n"
            )
        f.write("\\hline\\n")
        f.write("\\end{tabular}\\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UAVBench paper artifacts")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..N-1)")
    parser.add_argument("--output-root", type=Path, default=Path("results/paper"))
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    static_scenarios = list_scenarios_by_track("static")
    dynamic_scenarios = list_scenarios_by_track("dynamic")

    static_planners = ["astar", "theta_star"]
    dynamic_planners = [
        "astar",
        "theta_star",
        "dstar_lite",
        "ad_star",
        "dwa",
        "mppi",
    ]

    static_dir = output_root / "static_run"
    dynamic_dir = output_root / "dynamic_run"
    _run_track(static_scenarios, static_planners, seeds, static_dir)
    _run_track(dynamic_scenarios, dynamic_planners, seeds, dynamic_dir)

    static_csv = static_dir / "aggregates.csv"
    dynamic_csv = dynamic_dir / "aggregates.csv"
    static_out = output_root / "static_track_aggregates.csv"
    dynamic_out = output_root / "dynamic_track_aggregates.csv"
    static_out.write_text(static_csv.read_text(encoding="utf-8"), encoding="utf-8")
    dynamic_out.write_text(dynamic_csv.read_text(encoding="utf-8"), encoding="utf-8")

    _export_dynamic_event_views(
        dynamic_scenarios,
        dynamic_planners,
        seeds,
        output_root / "replan_events.csv",
        output_root / "interaction_metrics.csv",
    )

    _write_tex_table(static_out, output_root / "tables/main_results.tex")
    _write_tex_table(dynamic_out, output_root / "tables/dynamic_results.tex")
    stats = _paired_stats(dynamic_out, baseline="astar")
    stats["effect_size_by_mission"] = _mission_effect_sizes(dynamic_out, baseline="astar")
    (output_root / "stats_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    ablation_summary_rows: list[dict[str, Any]] = []
    if not args.skip_ablations:
        ablation_variants = [
            "no_interactions",
            "no_forced_breaks",
            "no_guardrail",
            "risk_only",
            "blocking_only",
        ]
        ablation_root = output_root / "ablations"
        for variant in ablation_variants:
            trial_rows = _run_variant_rows(dynamic_scenarios, dynamic_planners, seeds, variant)
            agg_rows = _aggregate_variant_rows(trial_rows)
            _write_rows_csv(trial_rows, ablation_root / f"{variant}_episodes.csv")
            _write_rows_csv(agg_rows, ablation_root / f"{variant}_aggregates.csv")
            _write_tex_table(ablation_root / f"{variant}_aggregates.csv", output_root / "tables" / f"ablation_{variant}.tex")

            planner_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in agg_rows:
                planner_group[row["planner_id"]].append(row)
            for planner_id, rows in planner_group.items():
                ablation_summary_rows.append(
                    {
                        "variant": variant,
                        "planner_id": planner_id,
                        "success_rate": round(sum(float(r["success_rate"]) for r in rows) / max(len(rows), 1), 6),
                        "interdiction_hit_rate_mean": round(
                            sum(float(r["interdiction_hit_rate_mean"]) for r in rows) / max(len(rows), 1), 6
                        ),
                        "dynamic_block_entropy_mean": round(
                            sum(float(r["dynamic_block_entropy_mean"]) for r in rows) / max(len(rows), 1), 6
                        ),
                    }
                )

    if ablation_summary_rows:
        _write_rows_csv(ablation_summary_rows, output_root / "ablation_summary.csv")
        _write_ablation_table(ablation_summary_rows, output_root / "tables/ablation_interactions.tex")

    manifest = {
        "git_commit": _git_commit(),
        "python": sys.version,
        "seeds": seeds,
        "tracks": {
            "static": static_scenarios,
            "dynamic": dynamic_scenarios,
        },
        "scenario_checksums": _scenario_checksums(),
        "stats_file": str(output_root / "stats_summary.json"),
        "protocol_invariants": [
            "same_snapshot_per_plan_call",
            "same_plan_budget_per_call",
            "same_replanning_triggers_and_limits",
            "shared_collision_checker",
            "seed_controls_all_dynamics",
        ],
    }
    (output_root / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved paper artifacts in: {output_root}")


if __name__ == "__main__":
    main()
