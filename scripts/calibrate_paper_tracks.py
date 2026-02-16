#!/usr/bin/env python3
"""Calibrate static/dynamic tracks for paper protocol targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml

from uavbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from uavbench.scenarios.loader import load_scenario
from uavbench.scenarios.registry import list_scenarios, list_scenarios_by_track


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _scenario_path(sid: str) -> Path:
    return Path("src/uavbench/scenarios/configs") / f"{sid}.yaml"


def _evaluate(
    scenario_ids: list[str],
    planners: list[str],
    seeds: list[int],
) -> dict[tuple[str, str], float]:
    runner = BenchmarkRunner(
        BenchmarkConfig(
            scenario_ids=[],
            planner_ids=[],
            seeds=[],
            save_jsonl=False,
            save_csv=False,
            verbose=False,
        )
    )
    out: dict[tuple[str, str], float] = {}
    for sid in scenario_ids:
        for pid in planners:
            ok = 0
            total = 0
            for seed in seeds:
                ep = runner._run_episode(sid, pid, seed)
                ok += 1 if ep.success else 0
                total += 1
            out[(sid, pid)] = ok / max(total, 1)
    return out


def _apply_adjustments(
    baseline: dict[tuple[str, str], float],
    adaptive: dict[tuple[str, str], float],
    static_scenarios: list[str],
    dynamic_scenarios: list[str],
) -> int:
    changed = 0
    for sid in static_scenarios:
        p = _scenario_path(sid)
        data = _load_yaml(p)
        astar_sr = baseline[(sid, "astar")]
        theta_sr = baseline[(sid, "theta_star")]
        if min(astar_sr, theta_sr) < 1.0:
            data["building_density"] = max(0.05, float(data.get("building_density", 0.3)) - 0.03)
            data["wind_level"] = "none"
            changed += 1
            _save_yaml(p, data)

    for sid in dynamic_scenarios:
        p = _scenario_path(sid)
        data = _load_yaml(p)
        base_sr = 0.5 * (baseline[(sid, "astar")] + baseline[(sid, "theta_star")])
        ad_sr = adaptive[(sid, "adaptive_astar")]
        if base_sr > 0.05:
            data["num_nfz_zones"] = min(7, int(data.get("num_nfz_zones", 4)) + 1)
            data["nfz_expansion_rate"] = min(1.6, float(data.get("nfz_expansion_rate", 1.0)) + 0.1)
            data["num_emergency_vehicles"] = min(20, int(data.get("num_emergency_vehicles", 10)) + 2)
            data["fire_ignition_points"] = min(14, int(data.get("fire_ignition_points", 0)) + (2 if data.get("enable_fire") else 0))
            changed += 1
            _save_yaml(p, data)
        elif ad_sr < 0.25:
            data["num_nfz_zones"] = max(3, int(data.get("num_nfz_zones", 4)) - 1)
            data["nfz_expansion_rate"] = max(0.8, float(data.get("nfz_expansion_rate", 1.0)) - 0.1)
            changed += 1
            _save_yaml(p, data)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate UAVBench paper tracks")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (starting at 0)")
    parser.add_argument("--max-iters", type=int, default=10, help="Maximum calibration iterations")
    parser.add_argument("--apply", action="store_true", help="Apply adjustments to YAML configs")
    parser.add_argument("--output", type=Path, default=Path("results/paper/calibration_summary.csv"))
    args = parser.parse_args()

    static_scenarios = list_scenarios_by_track("static")
    dynamic_scenarios = list_scenarios_by_track("dynamic")
    seeds = list(range(args.seeds))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for it in range(1, args.max_iters + 1):
        baseline = _evaluate(list_scenarios(), ["astar", "theta_star"], seeds)
        adaptive = _evaluate(dynamic_scenarios, ["adaptive_astar"], seeds)

        static_ok = all(baseline[(sid, "astar")] == 1.0 and baseline[(sid, "theta_star")] == 1.0 for sid in static_scenarios)
        dyn_base = [
            0.5 * (baseline[(sid, "astar")] + baseline[(sid, "theta_star")])
            for sid in dynamic_scenarios
        ]
        dyn_ad = [adaptive[(sid, "adaptive_astar")] for sid in dynamic_scenarios]
        dyn_base_mean = sum(dyn_base) / max(len(dyn_base), 1)
        dyn_ad_mean = sum(dyn_ad) / max(len(dyn_ad), 1)

        rows.append(
            {
                "iteration": it,
                "static_all_success": static_ok,
                "dynamic_baseline_mean": round(dyn_base_mean, 4),
                "dynamic_adaptive_mean": round(dyn_ad_mean, 4),
            }
        )

        print(
            f"[iter {it}] static_ok={static_ok} "
            f"dynamic_baseline_mean={dyn_base_mean:.4f} "
            f"dynamic_adaptive_mean={dyn_ad_mean:.4f}"
        )

        if static_ok and dyn_base_mean <= 0.05 and dyn_ad_mean >= 0.25:
            print("Calibration targets reached.")
            break

        if not args.apply:
            print("Dry run mode: stopping after first evaluation.")
            break

        changed = _apply_adjustments(baseline, adaptive, static_scenarios, dynamic_scenarios)
        print(f"Applied adjustments to {changed} scenarios.")
        if changed == 0:
            break

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "static_all_success", "dynamic_baseline_mean", "dynamic_adaptive_mean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved calibration summary: {args.output}")


if __name__ == "__main__":
    main()
