"""Generate 2 demo GIFs: dynamic planners on 2 different scenarios."""

import sys
import time
from pathlib import Path

from uavbench.cli.benchmark import run_dynamic_episode

OUT = Path("outputs/demo_gifs")
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("gov_civil_protection_medium", "aggressive_replan", 1),
    ("gov_maritime_domain_medium",  "periodic_replan",   0),
]

for scenario, planner, seed in RUNS:
    gif_path = OUT / f"{scenario}_{planner}_s{seed}.gif"
    print(f"Running {scenario} / {planner} / seed={seed} ...", flush=True)
    t0 = time.perf_counter()
    result = run_dynamic_episode(
        scenario, planner, seed=seed,
        render_gif=str(gif_path),
        render_dpi=100,
    )
    elapsed = time.perf_counter() - t0
    success = result.get("success", False)
    path_len = result.get("path_length", 0)
    replans = result.get("total_replans", 0)
    term = result.get("termination_reason", "?")
    print(
        f"  done in {elapsed:.0f}s | success={success} | "
        f"path_len={path_len} | replans={replans} | {term}",
        flush=True,
    )
    print(f"  GIF: {gif_path}", flush=True)

print("\nAll done.", flush=True)
