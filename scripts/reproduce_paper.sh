#!/usr/bin/env bash
# reproduce_paper.sh — Full reproducibility pipeline for UAVBench paper.
#
# Usage:
#   bash scripts/reproduce_paper.sh [--seeds N] [--skip-tests]
#
# This script:
#   1. Installs package in editable mode
#   2. Runs full test suite (contract + unit + integration)
#   3. Runs paper experiments (5 planners × 3 scenarios × 30 seeds = 450 episodes)
#   4. Generates analysis (LaTeX tables + figures)
#   5. Exports reproducibility artifacts
#   6. Writes pip freeze lock file

set -euo pipefail

SEEDS=30
SKIP_TESTS=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --seeds) SEEDS="$2"; shift 2;;
    --skip-tests) SKIP_TESTS=true; shift;;
    *) echo "Unknown option: $1"; exit 1;;
  esac
done

echo "=== UAVBench Paper Reproduction Pipeline ==="
echo "  Seeds per (planner, scenario): $SEEDS"
echo "  Skip tests: $SKIP_TESTS"
echo ""

# 1. Install
echo "--- Step 1: Install package ---"
pip install -e . --quiet
echo "  Done."

# 2. Lock dependencies
echo ""
echo "--- Step 2: Lock dependencies ---"
pip freeze > outputs/pip_freeze.txt
echo "  Saved: outputs/pip_freeze.txt"

# 3. Tests
if [ "$SKIP_TESTS" = false ]; then
  echo ""
  echo "--- Step 3: Run test suite ---"
  pytest tests/ -q --tb=short
  echo "  All tests passed."
else
  echo ""
  echo "--- Step 3: Tests SKIPPED ---"
fi

# 4. Paper experiments
echo ""
echo "--- Step 4: Run paper experiments (${SEEDS} seeds) ---"
python scripts/run_paper_experiments.py --seeds "$SEEDS" --output outputs/paper_results/all_episodes.csv
echo "  Results: outputs/paper_results/all_episodes.csv"

# 5. Analysis
echo ""
echo "--- Step 5: Generate analysis ---"
python scripts/analyze_paper_results.py \
  --input outputs/paper_results/all_episodes.csv \
  --table-dir outputs/paper_tables \
  --fig-dir outputs/paper_figures
echo "  Tables: outputs/paper_tables/"
echo "  Figures: outputs/paper_figures/"

# 6. Artifacts
echo ""
echo "--- Step 6: Export artifacts ---"
python scripts/export_artifacts.py
echo "  Artifacts: outputs/"

echo ""
echo "=== Reproduction complete ==="
echo "Outputs:"
echo "  outputs/paper_results/all_episodes.csv  — raw episode data"
echo "  outputs/paper_tables/*.tex              — LaTeX tables"
echo "  outputs/paper_figures/*.{png,pdf}       — figures"
echo "  outputs/pip_freeze.txt                  — dependency lock"
echo "  outputs/determinism_hashes.json         — DC-2 verification"
echo "  outputs/repro_manifest.json             — reproducibility manifest"
