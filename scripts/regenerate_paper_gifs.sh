#!/usr/bin/env bash
# UAVBench — Paper GIF Regeneration Script
# Regenerates hard_gifs for all 6 paper planners × 3 hard scenarios
# Run from project root. Each episode takes 5-30 minutes depending on map size.
#
# Why regenerate?
#   1. All existing GIFs predate semantic hardening (2026-02-24)
#   2. HUD tokens FORCED BLOCK: ACTIVE / CLEARED now wired (benchmark.py fixed)
#   3. ad_star + dstar_lite GIFs use deprecated planners → must be replaced
#   4. astar, theta_star, incremental_dstar_lite, grid_mppi have no hard_gifs at all
#
# Usage: bash scripts/regenerate_paper_gifs.sh
#        Or selectively: bash scripts/regenerate_paper_gifs.sh periodic_replan
#        (only runs the specified planner across all 3 scenarios)

set -e
mkdir -p outputs/hard_gifs

SCENARIOS=(
    "gov_civil_protection_hard"
    "gov_maritime_domain_hard"
    "gov_critical_infrastructure_hard"
)
PLANNERS=(
    "astar"
    "theta_star"
    "periodic_replan"
    "aggressive_replan"
    "incremental_dstar_lite"
    "grid_mppi"
)

# Filter to specific planner if passed as argument
FILTER="${1:-}"

for scenario in "${SCENARIOS[@]}"; do
    for planner in "${PLANNERS[@]}"; do
        if [ -n "$FILTER" ] && [ "$planner" != "$FILTER" ]; then
            continue
        fi
        out="outputs/hard_gifs/${scenario}_${planner}.gif"
        echo "=== $(date -u +%H:%M:%S) Generating: $scenario × $planner ==="
        python -m uavbench.cli.benchmark \
            --scenarios "$scenario" \
            --planners "$planner" \
            --track dynamic \
            --with-dynamics \
            --paper-protocol \
            --deterministic \
            --trials 1 \
            --seed-base 0 \
            --render-gif "$out" \
            --render-dpi 120
        echo "    -> $out"
    done
done

echo ""
echo "=== Regeneration complete ==="
echo "Deleting deprecated artifacts..."
rm -f \
    outputs/hard_gifs/gov_civil_protection_hard_ad_star.gif \
    outputs/hard_gifs/gov_civil_protection_hard_dstar_lite.gif \
    outputs/hard_gifs/gov_critical_infrastructure_hard_ad_star.gif \
    outputs/hard_gifs/gov_critical_infrastructure_hard_dstar_lite.gif \
    outputs/hard_gifs/gov_maritime_domain_hard_ad_star.gif \
    outputs/hard_gifs/gov_maritime_domain_hard_dstar_lite.gif \
    outputs/test_basemap.gif
echo "Done."
