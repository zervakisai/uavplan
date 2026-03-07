# UAVBench — Visual Truth Specification

This document defines the visualization contracts: what MUST appear on screen,
when, and in what order. It is the source of truth for contracts VC-1 and VC-2.

---

## 1. Renderer Modes

| Mode | Purpose | Layers Enabled | HUD |
|------|---------|---------------|-----|
| `paper_min` | Clean figures for paper (high-DPI PNG) | Base map, fire, smoke, trajectory, path, UAV, start/goal | Minimal: step counter, planner name |
| `ops_full` | Full operational rendering (animated GIF) | All 12+ layers | Full HUD, timeline bar, cartographic overlay |

Both modes enforce the same visual truth contracts (VC-1, VC-2).

---

## 2. Layer Z-Order Stack

Layers are rendered bottom-to-top. Higher z-index draws on top of lower.
No layer may overwrite a higher-priority layer.

| Z | Layer | Style | Notes |
|---|-------|-------|-------|
| 1 | Base map (terrain, buildings, roads) | Buildings opaque, roads grey, ground light | Always present |
| 3.5 | Smoke | Grey `#A0A0A0` (160,160,160), ~40% opacity | Threshold >= 0.5 for blocking |
| 3.8 | Fire buffer zone | Light orange (255,180,100), 30% opacity + dots | Safety exclusion ring |
| 4 | Fire | Red-orange (255,80,20), 80% opacity | CA states: BURNING shown |
| 5 | Dynamic NFZ (restriction zones) | Purple `#CC79A7` (Okabe-Ito), 50% + hatching | Hatch patterns per mission |
| 6 | Traffic closures/occupancy | Orange `#E69F00` (Okabe-Ito), ~40% opacity | From interaction engine |
| 9 | Trajectory trail | Blue `#0072B2` with white outline | Executed path |
| 9.5 | Planned path | Cyan `#56B4E9` dashed with black outline | VC-1 core layer |
| 9.6 | Start / Goal markers | Green `#009E73` / Yellow `#F0E442` | Okabe-Ito palette |
| 10 | UAV icon | Blue `#0072B2` disc + white ring + rotor cross | Always present |
| 12 | HUD | Monospace text box, badges | Top layer |
| 13 | Color legend bar | 10-swatch legend at frame bottom | Both modes |

---

## 3. Planned Path Rendering Rules (VC-1, VC-2)

### VC-1: Path Visibility

**Invariant**: If `plan_len > 1`, the planned path overlay MUST contain visible pixels.

| Condition | Rendering |
|-----------|-----------|
| `plan_len > 1` | Dashed cyan `#56B4E9` line (Okabe-Ito sky blue), with black outline |
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

## 4. (Removed — VC-3 Forced Block Lifecycle)

VC-3 has been removed. The abstract forced block system (ForcedBlockManager,
BlockLifecycle enum) has been replaced by physical interdiction mechanisms:
fire corridor closures and vehicle roadblocks. These are visible through the
existing fire and traffic overlays and do not require a separate lifecycle
rendering layer.

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
| Start | Bluish green `#009E73` circle (Okabe-Ito) | 9.6 |
| Goal | Yellow `#F0E442` circle (Okabe-Ito) | 9.6 |

### Trajectory

| Element | Style | Z |
|---------|-------|---|
| Trajectory outline | White, linewidth=4.5, alpha=0.7 | 9.4 |
| Trajectory | Blue `#0072B2` (Okabe-Ito), linewidth=2.5 | 9.5 |

### UAV Icon

| Safety State | Color |
|-------------|-------|
| Safe | Blue `#0072B2` (Okabe-Ito) |
| Caution | Orange `#E69F00` (Okabe-Ito) |
| Danger | Red `#D55E00` (Okabe-Ito vermillion) |

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
| VC-3 | (Removed — interdictions now physical fire/traffic events) | Section 4 (historical note) |
| VZ-1 | paper_min and ops_full modes | Section 1 (Renderer Modes) |
| VZ-2 | 12-layer z-order stack | Section 2 (Layer Z-Order Stack) |
| VZ-3 | Deterministic GIF export | Section 9 (Deterministic Rendering) |
