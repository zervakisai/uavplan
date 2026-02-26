# UAVBench v2 — Visual Truth Specification

This document defines the visualization contracts: what MUST appear on screen,
when, and in what order. It is the source of truth for contracts VC-1, VC-2, VC-3.

---

## 1. Renderer Modes

| Mode | Purpose | Layers Enabled | HUD |
|------|---------|---------------|-----|
| `paper_min` | Clean figures for paper (high-DPI PNG) | Base map, fire, smoke, forced blocks, trajectory, path, UAV, start/goal | Minimal: step counter, planner name |
| `ops_full` | Full operational rendering (animated GIF) | All 12+ layers | Full HUD, timeline bar, cartographic overlay |

Both modes enforce the same visual truth contracts (VC-1, VC-2, VC-3).

---

## 2. Layer Z-Order Stack

Layers are rendered bottom-to-top. Higher z-index draws on top of lower.
No layer may overwrite a higher-priority layer.

| Z | Layer | Style | Notes |
|---|-------|-------|-------|
| 1 | Base map (terrain, buildings, roads) | Buildings opaque, roads grey, ground light | Always present |
| 1.5 | Grid lines | `#D0D0D0`, linewidth=0.3, alpha=0.5 | `ops_full` only |
| 2 | Risk heatmap | YlOrRd colormap, alpha=0.30, bilinear | Non-blocking; informational |
| 3 | Smoke | Grey, alpha=0.25–0.35 | Threshold >= 0.3 for blocking |
| 4 | Fire | Red-orange, alpha=0.65 | CA states: BURNING shown |
| 4.9 | Static no-fly zones | Crimson `#CC2222`, alpha=0.35 | Permanent; baked from config |
| 5 | Dynamic restriction zones | Per-type colors (see zone table) | Hatch patterns per mission |
| 6 | Traffic closures | Yellow `#FFFF00` diagonal stripes | From interaction engine |
| 7 | Entities (vehicles, intruders) | Mission-specific icons | Traffic, intruders, moving target |
| 8 | Forced block markers | `#1A1A1A` X-scatter, alpha=0.65 | Interdiction cells |
| 9 | Trajectory + planned path | See path rendering rules below | Core visual truth layer |
| 10 | UAV icon + safety bubble | Blue/orange/red by safety state | Always present |
| 11 | Event annotations | Replan flash, corridor label | Transient overlays |
| 12 | HUD | Monospace text box, timeline bar | Top layer |

---

## 3. Planned Path Rendering Rules (VC-1, VC-2)

### VC-1: Path Visibility

**Invariant**: If `plan_len > 1`, the planned path overlay MUST contain visible pixels.

| Condition | Rendering |
|-----------|-----------|
| `plan_len > 1` | Dashed cyan `#4FC3F7` line, linewidth=1.8, with black outline (linewidth=3.5, alpha=0.6) |
| `plan_len == 1` | Single waypoint marker (no line) |
| `plan_len == 0` | No path rendered; NO_PLAN badge shown |

**Test**: Frame pixel analysis confirms cyan-family pixels present when `plan_len > 1`.

### VC-2: Plan Status Badges

**Invariant**: If the plan is missing or stale, the HUD MUST show a badge.

| Condition | Badge Text | Color |
|-----------|-----------|-------|
| `plan_len <= 1` | `NO PLAN` | Red `#FF4444` |
| `plan_age_steps > 2 * replan_every_steps` | `STALE PLAN ({plan_reason})` | Orange `#FF8C00` |
| `plan_len > 1 AND plan_age_steps <= threshold` | `PLAN: {plan_len}wp` | White (normal) |

The `plan_reason` string explains why the plan is stale or absent:
- `"path_blocked"` — next cell on path is now blocked
- `"goal_unreachable"` — BFS from current position cannot reach goal
- `"planner_timeout"` — planner exceeded budget
- `"initial"` — no plan computed yet

**Test**: HUD text extraction confirms badge presence under trigger conditions.

### Path Ghost (Replan Flash)

When a replan occurs, the OLD path is shown as a ghost for 3 frames:
- Style: dashed `#FF6666`, alpha=0.35
- Z-order: 9 (same as path layer, rendered first)
- Duration: 3 frames, then removed

---

## 4. Forced Block Lifecycle Rendering (VC-3)

Each forced interdiction has a lifecycle: `TRIGGERED → ACTIVE → CLEARED`.
The HUD and map rendering reflect all states.

### Map Rendering

| Lifecycle State | Map Style | HUD Badge |
|----------------|-----------|-----------|
| `TRIGGERED` | Cells flash amber for 2 frames | — |
| `ACTIVE` | X-markers at z=8, `#1A1A1A`, alpha=0.65 | `FORCED BLOCK: ACTIVE` (red) |
| `CLEARED` | X-markers fade out over 3 frames | `FORCED BLOCK: CLEARED` (green) |

### HUD Badge Rules

| Condition | Badge | Color |
|-----------|-------|-------|
| Any block in ACTIVE state | `FORCED BLOCK: ACTIVE` | Red `#FF0000` |
| Most recent block CLEARED, no active blocks | `FORCED BLOCK: CLEARED` | Green `#00CC44` |
| No blocks ever triggered | (no badge) | — |

### Lifecycle Timing

```
Step event_t1:   Block TRIGGERED → logged in events with step_idx
Step event_t1+1: Block ACTIVE → cells blocked on map, HUD badge
  ...
Step guardrail clears: Block CLEARED → HUD badge, cells fade
```

**Test**: Frame sequence at event_t1 shows lifecycle transitions; HUD badge text matches block state.

---

## 5. HUD Token Specification

The HUD is rendered as a semi-transparent text box at z=12.
Background: `#0A0F1A`, alpha=0.82. Border color varies by guardrail depth.

### Row 1: Identity

```
SCN: {scenario_id}  |  MSN: {mission_type.upper()}  |  TRK: {track.upper()}
```

### Row 2: Planner

```
PLN: {planner_name}  |  MOD: {mode_label}
```

### Row 3: Live Metrics

```
T: {step}  |  Dt: {elapsed_s:.1f}s  |  REP: {replans}  |  RISK: {risk:.2f}  |  HITS: {block_hits}  |  {guardrail_badge}  |  {feasibility_badge}  |  {plan_badge}  |  {block_badge}
```

### Token Definitions

| Token | Source Field | Format | Update Rate |
|-------|-------------|--------|-------------|
| `T` | `step_idx` | Integer | Every step |
| `Dt` | `step_idx * dt_s` | `{:.1f}s` | Every step |
| `REP` | `replans` | Integer | On replan |
| `RISK` | `risk_cost` | `{:.2f}` | Every step |
| `HITS` | `dynamic_block_hits` | Integer | On hit |
| `PLAN` | `plan_len` | `{n}wp` or badge | On replan |
| `FEAS` / `INFEAS` | `feasible_after_guardrail` | Boolean badge | On guardrail check |
| `COMM` | `1 - comms_dropout_prob` | `{:.0f}%` | Config-static |

### Guardrail Depth Symbols

| Depth | Symbol | Color |
|-------|--------|-------|
| 0 | G0 (circle) | White |
| 1 | G1 (bar) | Yellow `#FFD700` |
| 2 | G2 (triangle) | Orange `#FF8C00` |
| 3 | G3 (diamond) | Red `#FF0000` |

---

## 6. Event Timeline Bar

A horizontal progress bar at the bottom of the frame (z=24).

### Structure

```
|START ──── t1 ──── t2 ──── END|
         ▼         ▼
     replans   guardrails
```

### Elements

| Element | Style | Z |
|---------|-------|---|
| Background | `#111827`, alpha=0.85 | 24 |
| Progress fill | `#00DDFF`, alpha=0.30 | 24.1 |
| t1/t2 tick lines | `#334155` | 24.2 |
| t1/t2 labels | `#E8E8E8`, fontsize=5 | 24.3 |
| Replan markers | Downward triangle `#FFAA00` | 24.4 |
| Guardrail markers | Upward triangle `#FF2D00` | 24.4 |

### Labels

- Left edge: `START`
- Right edge: `END`
- At `event_t1 / max_steps` fraction: `t1`
- At `event_t2 / max_steps` fraction: `t2`

---

## 7. Mission-Specific Overlays

### POI Icons

| Mission | Icon | Halo |
|---------|------|------|
| Civil Protection | FIRE / CAMERA | White 2px |
| Maritime Domain | SHIP / ANCHOR | White 2px |
| Critical Infrastructure | BUILDING / INSPECTION | White 2px |

### Start/Goal Markers

| Marker | Style | Z |
|--------|-------|---|
| Start | Green `#00CC44` circle, s=max(180, W*0.6) | 9.6 |
| Goal | Gold `#FFD700` star, s=marker*1.2 | 9.6 |

### Trajectory

| Element | Style | Z |
|---------|-------|---|
| Trajectory outline | White, linewidth=4.5, alpha=0.7 | 9.4 |
| Trajectory | Blue `#0066FF`, linewidth=2.5 | 9.5 |

### UAV Icon

| Safety State | Color |
|-------------|-------|
| Safe | Blue `#0066FF` |
| Caution | Orange `#FF8C00` |
| Danger | Red `#FF0000` |

Safety state derived from: `safe` if no nearby hazards, `caution` if within risk buffer, `danger` if adjacent to blocking cell.

---

## 8. Cartographic Overlay (`ops_full` only)

| Element | Style | Position |
|---------|-------|----------|
| Scale bar | `{m} m / {km} km` | Bottom-left |
| North arrow | N with arrow | Top-right |
| Coordinate readout | `E: {easting} | N: {northing}` | Bottom |
| Attribution | `[CITE] ODbL` | Bottom-right |
| Version pin | `UAVBench v{version} @ {sha[:7]}` | Bottom-right |

---

## 9. Deterministic Rendering (VZ-3)

**Invariant**: Same episode data → same GIF file bytes.

Requirements:
- No timestamp-dependent rendering (no `datetime.now()` in pixel data)
- Fixed font rendering (no system-dependent font fallbacks)
- Fixed colormap discretization (256-step, not float-interpolated)
- Fixed figure DPI (150 for ops_full, 300 for paper_min)
- Fixed random state (no animation jitter from unseeded noise)

**Test**: Two GIF exports from identical episode data produce identical SHA-256 hashes.

---

## 10. Contract Compliance Checklist

| Contract | Requirement | Specification Section |
|----------|------------|----------------------|
| VC-1 | Planned path visible if `plan_len > 1` | Section 3 (Path Visibility) |
| VC-2 | NO_PLAN/STALE badge when plan missing/stale | Section 3 (Plan Status Badges) |
| VC-3 | Forced block lifecycle rendered | Section 4 (Forced Block Lifecycle) |
| VZ-1 | paper_min and ops_full modes | Section 1 (Renderer Modes) |
| VZ-2 | 12-layer z-order stack | Section 2 (Layer Z-Order Stack) |
| VZ-3 | Deterministic GIF export | Section 9 (Deterministic Rendering) |
