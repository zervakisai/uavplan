#!/usr/bin/env python3
"""UAVBench readiness checklist verifier."""

from dataclasses import fields
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavbench.scenarios.schema import MissionType
from uavbench.metrics.comprehensive import EpisodeMetrics


def main() -> None:
    checks = []

    # 1) mission coverage
    checks.append(("6+ mission types", len(MissionType) >= 6, f"found {len(MissionType)}"))

    # 2) metric coverage
    metric_count = len(fields(EpisodeMetrics))
    checks.append(("25+ per-episode metrics", metric_count >= 25, f"found {metric_count}"))

    # 3) CLI coverage
    cli_src = Path("src/uavbench/cli/benchmark.py").read_text(encoding="utf-8")
    option_count = len(re.findall(r"add_argument\(", cli_src))
    checks.append(("20+ CLI options", option_count >= 20, f"found {option_count}"))

    # 4) collision termination support
    urban_src = Path("src/uavbench/envs/urban.py").read_text(encoding="utf-8")
    has_collision_termination = "terminate_on_collision" in urban_src and "collision_terminated" in urban_src
    checks.append(("collision termination", has_collision_termination, "terminate_on_collision + collision_terminated hooks"))

    # 5) reproducibility metadata artifact
    runner_src = Path("src/uavbench/benchmark/runner.py").read_text(encoding="utf-8")
    has_metadata = "run_metadata.json" in runner_src
    checks.append(("repro metadata artifact", has_metadata, "run_metadata.json export"))

    print("UAVBench readiness checklist")
    print("=" * 40)
    failures = 0
    for label, ok, detail in checks:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{tag}] {label}: {detail}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
