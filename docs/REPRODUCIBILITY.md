# REPRODUCIBILITY.md — UAVBench Reproduction Guide

**Date:** 2025-02-20  
**Codebase:** HEAD (post-sprint-3)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10+ | 3.12+ |
| OS | Linux / macOS | Ubuntu 22.04+ |
| RAM | 4 GB | 8 GB |
| Disk | 500 MB | 2 GB (with results) |
| GPU | Not required | Not required |

---

## 2. Installation

```bash
# Clone repository
git clone https://github.com/zervakisai/uavbench.git
cd uavbench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package (editable mode)
pip install -e ".[dev,viz]"

# Verify installation
python -m pytest tests/ -q --tb=short
```

Expected output: `268 passed` (≈37 s).

---

## 3. Reproduce Paper Results

### 3.1 Smoke Test (5 minutes)

```bash
uavbench --mode nav --track static \
  --planners astar,periodic_replan \
  --trials 3 --save-csv \
  --output-dir results/smoke/
```

### 3.2 Navigation Track (full)

```bash
python scripts/paper_best_paper_validation.py \
  --seeds 50 \
  --output-dir results/paper/
```

### 3.3 Mission Track (full)

```bash
uavbench --mode mission --track all \
  --planners astar,theta_star,periodic_replan,aggressive_replan,greedy_local,grid_mppi,incremental_dstar_lite \
  --trials 50 --paper-protocol --save-csv --save-json \
  --output-dir results/paper_mission/
```

### 3.4 Ablation Study

```bash
for variant in no_interactions no_forced_breaks no_guardrail risk_only blocking_only; do
  uavbench --mode nav --track dynamic \
    --planners astar,periodic_replan,incremental_dstar_lite \
    --trials 50 --protocol-variant $variant --save-csv \
    --output-dir results/ablations/$variant/
done
```

---

## 4. Determinism Contract

UAVBench guarantees **bit-identical** episode trajectories for the same (scenario, planner, seed) triple.

**Verification:**

```bash
# Run twice with same seed
uavbench --scenarios gov_civil_protection_easy --planners astar --trials 1 --seed-base 42 --save-json --output-dir /tmp/run1
uavbench --scenarios gov_civil_protection_easy --planners astar --trials 1 --seed-base 42 --save-json --output-dir /tmp/run2

# Compare outputs
diff /tmp/run1/*.json /tmp/run2/*.json
# Expected: no differences
```

---

## 5. Key Configuration Files

| File | Purpose |
|---|---|
| `pyproject.toml` | Dependencies, version bounds, pytest config |
| `src/uavbench/scenarios/configs/*.yaml` | 9 scenario definitions (3 missions × 3 difficulties) |
| `src/uavbench/planners/__init__.py` | Planner registry (11 keys) |

---

## 6. Validating a Release

```bash
python tools/validate_release.py
```

This script verifies:
- All 9 scenario YAMLs parse without error
- All 11 planners instantiate correctly
- Realism knobs are at top-level (not nested under `extra:`)
- Test suite passes
- Key documentation files exist

---

## 7. Known Platform Notes

- **macOS ARM (Apple Silicon):** Fully supported. NumPy/SciPy use Accelerate.
- **Windows:** Untested. POSIX paths in tests may need adjustment.
- **Docker:** See `Dockerfile` in repository root.
