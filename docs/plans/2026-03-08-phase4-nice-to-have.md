# Phase 4: Nice-to-Have Upgrades (10-16) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 7 polish upgrades to UAVBench Paper #1: energy budget, fire-spawned tasks, APF wind-aware momentum, fog split-screen, risk heatmap overlay, failure autopsy GIF, trajectory heatmap.

**Architecture:** Each upgrade is independent. Energy budget adds a constraint dimension to the runner loop. Fire-spawned tasks extend fire_ca with event emission consumed by TriageMission. APF momentum smooths oscillation. Visualization upgrades add new overlay/rendering modes and scripts.

**Tech Stack:** Python 3.10+, numpy, scipy, matplotlib, imageio, PIL

---

### Task 1: Energy Budget (Upgrade 10)

**Files:**
- Modify: `src/uavbench/scenarios/schema.py` — add `energy_budget` field
- Modify: `src/uavbench/envs/base.py` — add `ENERGY_DEPLETED` termination reason
- Modify: `src/uavbench/benchmark/runner.py` — track energy, terminate when depleted
- Modify: `src/uavbench/metrics/compute.py` — add energy metrics
- Test: `tests/test_energy_budget.py`

**Implementation:**
- ScenarioConfig: `energy_budget: float = 0.0` (0=unlimited)
- TerminationReason: add `ENERGY_DEPLETED = "energy_depleted"`
- Runner: track `energy_remaining`, costs: move=1.0, stay=0.3, replan=2.0
- When energy_remaining <= 0 and budget > 0: terminate with ENERGY_DEPLETED
- Metrics: `energy_consumed`, `energy_remaining_pct`
- Backward compat: energy_budget=0 means no energy tracking (default)

### Task 2: Fire-Spawned Tasks (Upgrade 11)

**Files:**
- Modify: `src/uavbench/dynamics/fire_ca.py` — add `pop_events()` method
- Modify: `src/uavbench/missions/triage.py` — add `inject_casualty()` method
- Test: `tests/test_fire_spawned_tasks.py`

**Implementation:**
- FireSpreadModel: `_events: list[dict]` accumulator
- In step(), when a cell transitions UNBURNED→BURNING and landuse is building (2/3): emit `{"type": "building_fire", "x": x, "y": y, "step": step_count}`
- `pop_events() → list[dict]`: returns and clears pending events
- TriageMission.inject_casualty(xy, severity, step): spawn new casualty at runtime

### Task 3: APF Directional Repulsion + Momentum (Upgrade 12)

**Files:**
- Modify: `src/uavbench/planners/apf.py` — add wind bias + momentum
- Test: `tests/test_apf_momentum.py`

**Implementation:**
- Wind-aware repulsion: when wind_speed > 0, bias repulsive gradient toward upwind
  `wind_bias = wind_speed * cos(cell_angle - wind_dir)` added to repulsive potential
- Momentum: `plan()` returns smoothed path where each step blends 0.7*previous_direction + 0.3*gradient
- Only active when config has wind_speed > 0 (backward compat)

### Task 4: Risk Heatmap Viz Layer (Upgrade 14)

**Files:**
- Modify: `src/uavbench/visualization/overlays.py` — add `draw_risk_heatmap()`
- Modify: `src/uavbench/visualization/renderer.py` — call overlay at Z=3.2
- Modify: `src/uavbench/benchmark/runner.py` — pass cost_map in frame_state
- Test: `tests/test_risk_heatmap_viz.py`

**Implementation:**
- `draw_risk_heatmap(frame, cost_map, cell, alpha=0.4)`:
  green(0) → yellow(0.5) → red(1.0) translucent overlay
- Only renders when `state.get("cost_map")` is present
- Legend entry: "RISK" with gradient swatch

### Task 5: Fog Split-Screen Rendering (Upgrade 13)

**Files:**
- Modify: `src/uavbench/visualization/renderer.py` — add `render_fog_comparison()`
- Create: `scripts/gen_fog_comparison.py`
- Test: `tests/test_fog_split_screen.py`

**Implementation:**
- `render_fog_comparison(heightmap, state, dyn_state, fog_state)`:
  renders two frames side-by-side — fog-filtered (left) vs ground truth (right)
- Divider line + labels "Agent View" / "Ground Truth"
- Script runs episode with fog, captures paired frames, outputs GIF

### Task 6: Failure Autopsy GIF (Upgrade 15)

**Files:**
- Create: `scripts/gen_failure_autopsy.py`

**Implementation:**
- Run episode; if failed, extract last 20 frames
- Zoom 3x on agent's final position (crop + upscale)
- Annotate each frame with step, distance_to_goal, reject_reasons
- Output zoomed GIF of failure sequence

### Task 7: Trajectory Heatmap Overlay (Upgrade 16)

**Files:**
- Create: `scripts/gen_trajectory_heatmap.py`

**Implementation:**
- Run N seeds (default 30) per planner, collect all trajectories
- For each planner: count cell visit frequency across all seeds
- Normalize to [0,1], apply viridis colormap as translucent overlay on basemap
- 5-panel figure (one per planner), IEEE two-column width, 300 DPI PNG + PDF
