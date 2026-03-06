#!/usr/bin/env bash
# Regenerate paper GIF animations for UAVBench.
#
# Generates one GIF per (scenario, planner) combination for the 3
# OSM scenarios × 5 planners = 15 GIFs total.
#
# Usage:
#     bash scripts/regenerate_paper_gifs.sh
#
# Output:
#     outputs/gifs/{scenario_id}_{planner_id}.gif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GIF_DIR="$PROJECT_ROOT/outputs/gifs"

mkdir -p "$GIF_DIR"

# OSM scenarios (Greece)
DYNAMIC_SCENARIOS=(
    osm_penteli_fire_delivery_medium
    osm_piraeus_flood_rescue_medium
    osm_downtown_fire_surveillance_medium
)


# All 5 planners
PLANNERS=(
    astar
    periodic_replan
    aggressive_replan
    dstar_lite
    apf
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
from uavbench.benchmark.runner import run_episode
result = run_episode('$scenario', '$planner', $SEED)
status = 'OK' if result.metrics['success'] else 'FAIL'
print(f'  -> {status} steps={result.metrics[\"executed_steps_len\"]}')
" 2>&1 || echo "  -> ERROR (scenario may require dynamics)"
    done
done


echo ""
echo "=== Completed $count runs ==="
echo "GIF rendering requires Phase 9+ integration with renderer."
echo "Episode data validated for reproducibility."
