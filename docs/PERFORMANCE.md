# Performance Benchmarks

Results from running the full 20-scenario pack with A* planner on Apple Silicon (M-series).

## Scenario Pack Results (A*, 1 trial)

| Scenario | SR | Path | Opt | Smooth | Risk | Plan (ms) |
|----------|---:|-----:|----:|-------:|-----:|----------:|
| wildfire_easy | 100% | 212 | 1.00 | 0.77 | 0.00 | 11.6 |
| wildfire_medium | 100% | 286 | 1.00 | 0.80 | 0.00 | 17.3 |
| wildfire_hard | 100% | 372 | 0.99 | 0.88 | 0.00 | 27.0 |
| emergency_easy | 100% | 679 | 0.73 | 0.57 | 36.92 | 81.8 |
| emergency_medium | 100% | 501 | 0.97 | 0.68 | 11.93 | 14.6 |
| emergency_hard | 100% | 313 | 0.88 | 0.66 | 5.80 | 12.0 |
| port_easy | 0% | - | - | - | - | 0.1 |
| port_medium | 100% | 428 | 0.90 | 0.85 | 2.34 | 6.0 |
| port_hard | 100% | 702 | 1.00 | 0.75 | 3.46 | 150.9 |
| crisis_hard | 100% | 593 | 0.99 | 0.82 | 17.92 | 60.6 |
| sar_easy | 100% | 385 | 1.00 | 0.82 | 5.80 | 61.5 |
| sar_medium | 100% | 346 | 1.00 | 0.80 | 0.00 | 32.6 |
| sar_hard | 100% | 504 | 1.00 | 0.90 | 0.00 | 79.2 |
| infrastructure_easy | 0% | - | - | - | - | 302.7 |
| infrastructure_medium | 0% | - | - | - | - | 0.0 |
| infrastructure_hard | 100% | 645 | 0.98 | 0.61 | 26.65 | 53.8 |
| border_easy | 100% | 399 | 1.00 | 0.90 | 2.27 | 21.8 |
| border_medium | 100% | 586 | 1.00 | 0.93 | 0.23 | 146.6 |
| border_hard | 100% | 407 | 1.00 | 0.88 | 0.00 | 64.0 |
| comms_denied_hard | 100% | 375 | 0.89 | 0.58 | 7.81 | 29.9 |

**Columns:** SR = success rate, Path = path length (steps), Opt = path optimality (manhattan/actual), Smooth = path smoothness, Risk = cumulative risk exposure, Plan = A* planning time.

## Key Observations

- **A* near-optimal on open terrain:** Penteli scenarios (wildfire, SAR, border) achieve optimality > 0.99
- **Downtown scenarios harder:** Dense urban grids reduce optimality (emergency_easy: 0.73) and smoothness
- **Seed-dependent failures:** port_easy, infrastructure_easy/medium show 0% on single trial but reach 60-100% over 5 trials — start/goal placement in dense areas
- **Planning time scales with grid complexity:** Open terrain (Penteli) < 80ms, dense urban (Downtown) can reach 300ms for exhaustive search
- **Risk exposure concentrated in downtown:** Emergency and infrastructure scenarios traverse high-risk areas

## Tile Statistics

| Tile | Dimensions | Free Cells | Buildings | Roads | NFZ |
|------|-----------|------------|-----------|-------|-----|
| downtown | 500x500 | ~180k | ~45k | ~25k | ~5k |
| penteli | 500x500 | ~230k | ~12k | ~8k | ~2k |
| piraeus | 500x500 | ~195k | ~38k | ~18k | ~8k |

## Runtime Performance

### Dynamic Episode Step Time (gov_civil_protection_hard, seed=42)

The env.step() function was optimized in v2.0 with **26× wall-clock speedup**:

| Component | Before (ms/step) | After (ms/step) | Speedup |
|-----------|------------------:|------------------:|--------:|
| NFZ/TFR zones | 380.1 | 12.5 | 30× |
| BFS reachability | 210.9 | 1.9 | 110× |
| Fire spread | 2.8 | 2.7 | 1.0× |
| Interaction engine | 2.7 | 2.6 | 1.0× |
| Population risk | 1.1 | 1.0 | 1.1× |
| **Total step** | **599.5** | **22.6** | **26.5×** |

**Optimizations applied (zero algorithmic/fairness changes):**
1. **NFZ fire-cluster centroids**: `scipy.ndimage.label` + `np.bincount` vectorized centroids (replaces per-cluster `np.where` loop)
2. **BFS reachability**: `scipy.ndimage.label` connected-component check (replaces pure-Python deque BFS on 500×500 grid)
3. **Fire proximity & vehicle near-miss**: Vectorized numpy slices (replaces nested Python loops)
4. **Dynamic state cache**: Per-step memoization of `get_dynamic_state()` (called multiple times per step)
5. **Topology-change detection**: Guardrail skips BFS when blocking mask unchanged

### Static Operations

| Operation | Time |
|-----------|------|
| Load .npz tile | ~2ms |
| A* planning (typical) | 10-80ms |
| A* planning (worst case, dense grid) | ~300ms |
| Fire model step | ~0.5ms |
| Traffic model step | ~0.1ms |
| Full 20-scenario pack (1 trial each) | ~1.5s |
| Full 20-scenario pack (5 trials each) | ~7s |

## Memory Usage

| Component | Approximate Size |
|-----------|-----------------|
| Single .npz tile (disk) | ~2MB |
| Loaded tile (memory) | ~6MB (5 arrays x 500x500) |
| A* planner (peak) | ~50MB (visited set for 250k cells) |
| Fire model state | ~1.5MB (3 state arrays) |
| Traffic model state | ~0.1MB |

## Reproducing These Results

```bash
# Install
pip install -e ".[viz]"

# ── Scenario pack (static planners) ──
python tools/benchmark_scenario_pack.py --planners astar --trials 1 --output outputs/
python tools/benchmark_scenario_pack.py --planners astar --trials 5 --seed-base 42 --output outputs/

# ── Paper artifact export ──
# FAST_DEV: quick iteration (seeds=1, horizon=500, no ablations)
python scripts/export_paper_artifacts.py --mode FAST_DEV

# CAMERA_READY: full publication run (seeds=30, horizon=2000, all ablations)
python scripts/export_paper_artifacts.py --mode CAMERA_READY

# Parallel execution (4 workers)
python scripts/export_paper_artifacts.py --mode CAMERA_READY --parallel 4

# Custom settings
python scripts/export_paper_artifacts.py --seeds 5 --episode-horizon 1000 --output-root results/custom
```
