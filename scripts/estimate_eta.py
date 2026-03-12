#!/usr/bin/env python3
"""Quick ETA estimator — times 1 episode per planner×scenario, projects total."""

import time
import sys

from uavbench.planners import PLANNERS
from uavbench.scenarios.registry import list_scenarios
from uavbench.benchmark.runner import run_episode

CANONICAL = ["astar", "periodic_replan", "aggressive_replan", "dstar_lite", "apf"]
scenarios = list_scenarios()
n_seeds_main = 30
workers = 6

print("=== UAVBench ETA Estimator ===\n")
print(f"Planners:  {len(CANONICAL)}")
print(f"Scenarios: {len(scenarios)} — {', '.join(s.replace('osm_','') for s in scenarios)}")
print(f"Seed:      0 (single sample)\n")

times: dict[str, dict[str, float]] = {}
total_sample = 0.0

for scn in scenarios:
    times[scn] = {}
    scn_short = scn.replace("osm_", "").replace("_medium", "")
    for pid in CANONICAL:
        t0 = time.perf_counter()
        try:
            run_episode(scn, pid, seed=99)  # seed 99 to avoid cache overlap
            elapsed = time.perf_counter() - t0
            status = "OK"
        except Exception as e:
            elapsed = time.perf_counter() - t0
            status = f"ERR: {e}"
        times[scn][pid] = elapsed
        total_sample += elapsed
        print(f"  {scn_short:>30s} / {pid:<20s}  {elapsed:6.1f}s  {status}")
    print()

# --- Summary table ---
print("=" * 72)
print(f"{'':>30s}  {'astar':>6s} {'period':>6s} {'aggres':>6s} {'dstar':>6s} {'apf':>6s}")
print("-" * 72)
for scn in scenarios:
    scn_short = scn.replace("osm_", "").replace("_medium", "")
    vals = "  ".join(f"{times[scn].get(p, 0):5.1f}s" for p in CANONICAL)
    print(f"{scn_short:>30s}  {vals}")

# --- ETA projection ---
# The main run uses 6 planners (including incremental_astar alias) × 3 scenarios × 30 seeds
# with 6 parallel workers. Each worker handles one (scenario, planner) block of 30 seeds.
n_blocks = 6 * len(scenarios)  # 18 blocks (6 planners × 3 scenarios)
# Average time per episode from our sample
avg_per_episode = total_sample / (len(CANONICAL) * len(scenarios))
# Each block = 30 seeds sequential
avg_block_time = avg_per_episode * n_seeds_main
# With 6 workers, blocks run in waves of 6
n_waves = -(-n_blocks // workers)  # ceil division
estimated_total = n_waves * avg_block_time

# Per-planner projection (useful for understanding which planner is slow)
planner_avgs = {}
for pid in CANONICAL:
    planner_avgs[pid] = sum(times[scn][pid] for scn in scenarios) / len(scenarios)

print()
print("=" * 72)
print("ETA PROJECTION")
print("=" * 72)
print(f"  Avg episode time:       {avg_per_episode:.1f}s")
print(f"  Blocks (scn×planner):   {n_blocks} ({n_seeds_main} seeds each)")
print(f"  Workers:                {workers}")
print(f"  Waves:                  {n_waves}")
print(f"  Est. block time:        {avg_block_time/60:.1f}m")
print(f"  Est. total wall time:   {estimated_total/60:.0f}m ({estimated_total/3600:.1f}h)")
print()
print("  Per-planner avg episode time:")
for pid in CANONICAL:
    print(f"    {pid:<20s}  {planner_avgs[pid]:.1f}s")
print()

# Check how long the main run has been going
import subprocess
try:
    result = subprocess.run(
        ["ps", "-o", "etime=", "-p", "95978"],
        capture_output=True, text=True, timeout=5,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"  Main run elapsed:       {result.stdout.strip()}")
        print(f"  Est. remaining:         ~{max(0, estimated_total/60 - 20):.0f}m (rough)")
except Exception:
    pass
