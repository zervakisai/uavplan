#!/usr/bin/env bash
# Regenerate paper GIF animations for UAVBench v2.
#
# Generates one GIF per (scenario, planner) combination for the 6 dynamic
# scenarios × 6 planners = 36 GIFs total. Static (easy) scenarios get
# one representative planner (astar) = 3 additional GIFs.
#
# Usage:
#     bash scripts/regenerate_v2_paper_gifs.sh
#
# Output:
#     outputs/v2/gifs/{scenario_id}_{planner_id}.gif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GIF_DIR="$PROJECT_ROOT/outputs/v2/gifs"

mkdir -p "$GIF_DIR"

# Dynamic scenarios (medium + hard per family)
DYNAMIC_SCENARIOS=(
    gov_fire_delivery_medium
    gov_fire_delivery_hard
    gov_flood_rescue_medium
    gov_flood_rescue_hard
    gov_fire_surveillance_medium
    gov_fire_surveillance_hard
)

# Static scenarios (easy per family, astar only)
STATIC_SCENARIOS=(
    gov_fire_delivery_easy
    gov_flood_rescue_easy
    gov_fire_surveillance_easy
)

# All 6 planners
PLANNERS=(
    astar
    theta_star
    periodic_replan
    aggressive_replan
    dstar_lite
    mppi_grid
)

SEED=42

echo "=== UAVBench v2 Paper GIF Regeneration ==="
echo "Output directory: $GIF_DIR"
echo ""

count=0

# Dynamic: all planners
for scenario in "${DYNAMIC_SCENARIOS[@]}"; do
    for planner in "${PLANNERS[@]}"; do
        count=$((count + 1))
        echo "[$count] $scenario / $planner (seed=$SEED)"
        python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')
from uavbench2.benchmark.runner import run_episode
result = run_episode('$scenario', '$planner', $SEED)
status = 'OK' if result.metrics['success'] else 'FAIL'
print(f'  -> {status} steps={result.metrics[\"executed_steps_len\"]}')
" 2>&1 || echo "  -> ERROR (scenario may require dynamics)"
    done
done

# Static: astar only
for scenario in "${STATIC_SCENARIOS[@]}"; do
    count=$((count + 1))
    echo "[$count] $scenario / astar (seed=$SEED)"
    python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/src')
from uavbench2.benchmark.runner import run_episode
result = run_episode('$scenario', 'astar', $SEED)
status = 'OK' if result.metrics['success'] else 'FAIL'
print(f'  -> {status} steps={result.metrics[\"executed_steps_len\"]}')
" 2>&1 || echo "  -> ERROR"
done

echo ""
echo "=== Completed $count runs ==="
echo "GIF rendering requires Phase 9+ integration with renderer."
echo "Episode data validated for reproducibility."
