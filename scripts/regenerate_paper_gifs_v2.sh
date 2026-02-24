#!/usr/bin/env bash
# regenerate_paper_gifs_v2.sh — Post-semantic-hardening GIF regeneration
# Date: 2026-02-24
# Context: All GIFs must be regenerated to include:
#   - FORCED BLOCK: ACTIVE / CLEARED HUD tokens
#   - PLAN:Nwp / STALE HUD badges
#   - Blue planned_path overlay (correctly wired)
#   - P1 calibration (gnss_noise=0, comms_dropout=0, constraint_latency=0)
#
# Usage:
#   chmod +x scripts/regenerate_paper_gifs_v2.sh
#   bash scripts/regenerate_paper_gifs_v2.sh          # full run (18 GIFs)
#   bash scripts/regenerate_paper_gifs_v2.sh --quick   # only periodic_replan (3 GIFs)
#
# Estimated time: ~1.5-6 hours for full; ~15-30 min for --quick

set -euo pipefail
OUTDIR="outputs/hard_gifs"
mkdir -p "$OUTDIR"

SCENARIOS=(
  gov_civil_protection_hard
  gov_maritime_domain_hard
  gov_critical_infrastructure_hard
)

if [[ "${1:-}" == "--quick" ]]; then
  PLANNERS=(periodic_replan)
  echo "=== Quick mode: periodic_replan only (3 GIFs) ==="
else
  PLANNERS=(
    astar
    theta_star
    periodic_replan
    aggressive_replan
    incremental_dstar_lite
    grid_mppi
  )
  echo "=== Full mode: 6 planners x 3 scenarios (18 GIFs) ==="
fi

# Step 1: Delete deprecated GIFs
echo ""
echo "--- Deleting deprecated GIFs (ad_star, dstar_lite) ---"
for dep in "$OUTDIR"/*ad_star*.gif "$OUTDIR"/*dstar_lite*.gif; do
  if [[ -f "$dep" ]]; then
    echo "  rm $dep"
    rm "$dep"
  fi
done

# Step 2: Regenerate
echo ""
echo "--- Regenerating paper GIFs ---"
TOTAL=$(( ${#SCENARIOS[@]} * ${#PLANNERS[@]} ))
COUNT=0

for scenario in "${SCENARIOS[@]}"; do
  for planner in "${PLANNERS[@]}"; do
    COUNT=$((COUNT + 1))
    GIF="$OUTDIR/${scenario}_${planner}.gif"
    echo ""
    echo "[$COUNT/$TOTAL] $scenario x $planner -> $GIF"
    python -m uavbench.cli.benchmark \
      --scenarios "$scenario" \
      --planners "$planner" \
      --track dynamic \
      --with-dynamics \
      --paper-protocol \
      --deterministic \
      --trials 1 \
      --seed-base 0 \
      --render-gif "$GIF" \
      --render-dpi 120 \
      2>&1 | tail -3
    echo "  Done: $(ls -lh "$GIF" 2>/dev/null | awk '{print $5}')"
  done
done

echo ""
echo "=== Regeneration complete: $COUNT GIFs ==="
ls -lh "$OUTDIR"/*.gif | grep -v ad_star | grep -v dstar_lite
