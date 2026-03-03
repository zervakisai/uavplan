"""UAVBench v2 CLI entry point (RU-2).

Exactly ONE CLI. No forks, no duplicates.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m uavbench``."""
    parser = argparse.ArgumentParser(
        prog="uavbench",
        description="UAVBench v2 — UAV navigation benchmark",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run subcommand (RU-5) ---
    run_parser = sub.add_parser("run", help="Run benchmark episodes")
    run_parser.add_argument("--scenarios", type=str, default="gov_fire_delivery_easy")
    run_parser.add_argument("--planners", type=str, default="astar")
    run_parser.add_argument("--trials", type=int, default=1)
    run_parser.add_argument("--seed-base", type=int, default=0)
    run_parser.add_argument("--output-dir", type=str, default="outputs/v2")

    args = parser.parse_args(argv)

    if args.command == "run":
        from uavbench.benchmark.runner import run_episode

        scenarios = [s.strip() for s in args.scenarios.split(",")]
        planners = [p.strip() for p in args.planners.split(",")]

        for scenario_id in scenarios:
            for planner_id in planners:
                for trial in range(args.trials):
                    seed = args.seed_base + trial
                    result = run_episode(
                        scenario_id=scenario_id,
                        planner_id=planner_id,
                        seed=seed,
                    )
                    status = "OK" if result.metrics["success"] else "FAIL"
                    print(
                        f"[{status}] {scenario_id} / {planner_id} / seed={seed} "
                        f"steps={result.metrics['executed_steps_len']}"
                    )
    elif args.command is None:
        parser.print_help()
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)
