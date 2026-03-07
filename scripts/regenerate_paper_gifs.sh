#!/usr/bin/env bash
# Regenerate paper GIF animations for UAVBench.
#
# Generates animated GIFs showing all 3 OSM scenarios x 5 planners = 15 GIFs
# plus 3 OSM overview GIFs (one per scenario with aggressive_replan).
#
# Usage:
#     bash scripts/regenerate_paper_gifs.sh
#
# Output:
#     outputs/demo_gifs/{scenario_id}_{planner_id}_s42.gif

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== UAVBench v2 Paper GIF Regeneration ==="
echo "Using gen_demo_gifs.py for consistent rendering pipeline."
echo ""

# Full set: all scenarios x all planners
cd "$PROJECT_ROOT"
python3 scripts/gen_demo_gifs.py --fps 10 --skip-frames 3

echo ""
echo "=== OSM overview GIFs (one per scenario) ==="
python3 scripts/gen_demo_gifs.py --osm --fps 10 --skip-frames 3

echo ""
echo "=== Complete ==="
echo "GIFs: outputs/demo_gifs/"
