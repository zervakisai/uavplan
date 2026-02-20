# Stakeholder Visualization Specification

**Version**: 1.0  
**Status**: Implemented  
**Date**: 2025-01-XX  

## Overview

Stakeholder-ready demo visualization system for three Greek government agencies,
designed for **at-a-glance comprehension** (<5 seconds) by non-technical audiences.

### Target Agencies

| Code | Agency | Greek Name | Mission | Tile |
|------|--------|------------|---------|------|
| M1 | ΓΓΠΠ (GSCP) | Γενική Γραμματεία Πολιτικής Προστασίας | Wildfire Monitoring | `penteli` |
| M2 | ΛΣ-ΕΛΑΚΤ (HCG) | Λιμενικό Σώμα – Ελληνική Ακτοφυλακή | Maritime SAR | `piraeus` |
| M3 | ΥΠΕΘΑ (MoD) | Υπουργείο Εθνικής Άμυνας | Infrastructure Inspection | `downtown` |

---

## Task 1 — Visual Spec (4-Pane Layout)

### Layout

```
┌─────────────────────────────┬──────────────┐
│                             │  METRICS     │
│      MAP VIEWPORT           │  (KPIs,      │
│      (75% width)            │   gauges)    │
│                             │              │
├─────────────────────────────┼──────────────┤
│  TIMELINE BAR               │  LEGEND      │
│  (events, phase markers)    │  (icons)     │
└─────────────────────────────┴──────────────┘
```

- **Output**: 1920×1080 pixels (19.2" × 10.8" @ 100 DPI)
- **Map Viewport**: 75% width, 83% height — the main visual
- **Metrics Panel**: 25% width, 83% height — live KPIs
- **Timeline Bar**: 75% width, 17% height — event chronology
- **Legend Panel**: 25% width, 17% height — icon key

### Implementation

`src/uavbench/visualization/stakeholder_renderer.py` — `StakeholderRenderer` class.

---

## Task 2 — Icon System

### Architecture

Path-based vector icons rendered as `matplotlib.patches.PathPatch` artists.
All icons defined as unit-square (0–1) path vertices — resolution-independent,
no external assets, works headlessly with Agg backend.

### Icon Registry (20 icons)

| IconID | Shape | Default Colour | Usage |
|--------|-------|---------------|-------|
| `uav` | Triangle (top-down) | `#0066FF` | UAV position |
| `fire` | Flame teardrop | `#FF2D00` | Fire perimeter point |
| `ship` | Vessel silhouette | `#4488FF` | Vessel / boat |
| `building` | House with roof | `#5A5A5A` | Building / site |
| `waypoint` | Circle (12-gon) | `#00CC44` | Pending waypoint |
| `waypoint_done` | Circle (completed) | `#00CC44` (α=0.5) | Completed waypoint |
| `nfz` | Octagon (stop-sign) | `#FF00FF` | No-fly zone |
| `person` | Person silhouette | `#FFD700` | Person / victim |
| `distress` | Diamond | `#FF0044` | SOS / distress |
| `anchor` | Anchor T-shape | `#334155` | Port / anchorage |
| `camera` | Camera rectangle | `#00DDFF` | Surveillance point |
| `shield` | Shield shape | `#3388FF` | Protected zone |
| `alert` | Warning triangle | `#FF6B35` | Alert / hazard |
| `home` | House silhouette | `#00CC44` | Home base |
| `wind` | Arrow (right) | `#888888` | Wind direction |
| `radio` | Antenna | `#3388FF` | Comms tower |
| `inspection` | Magnifying glass | `#00DDFF` | Inspection point |
| `patrol` | Crosshair circle | `#4488FF` | Patrol waypoint |
| `hazard` | Hexagon | `#FF6B35` | Hazard zone |
| `corridor` | Double-headed arrow | `#00E5FF` | Emergency corridor |

### Rendering Rules

1. **Size**: configurable in grid-cell units (default 10)
2. **Status colours**: pending (full α) → active (pulse glow) → completed (α=0.4)
3. **Labels**: optional text below icon, configurable offset and font size
4. **Rotation**: CCW degrees for heading-dependent icons (UAV, wind)
5. **Z-order**: icons render at Z5 (POIs), Z6 (entities), Z8 (UAV)

### Implementation

`src/uavbench/visualization/icons/library.py` — `IconLibrary` class.

---

## Task 3 — OSM Basemap

### Tile Dataset

Three offline-rasterised tiles from OpenStreetMap via `osmnx`:

| Tile | Centre (lat/lon) | Coverage | Description |
|------|------------------|----------|-------------|
| `penteli` | 38.0807°N, 23.8335°E | 1.5 km² | Wildland-Urban Interface (WUI) |
| `downtown` | 37.9756°N, 23.7348°E | 1.5 km² | Athens city centre (Syntagma) |
| `piraeus` | 37.9432°N, 23.6432°E | 1.5 km² | Piraeus port area |

### Raster Specification

- **Grid**: 500 × 500 cells
- **Resolution**: 3.0 m/pixel
- **CRS**: EPSG:32634 (UTM zone 34N)
- **Format**: `.npz` with arrays:
  - `heightmap` (float32) — building heights
  - `roads_mask` (bool) — road network
  - `landuse_map` (int8, 0–4) — 0=other, 1=residential, 2=commercial, 3=vegetation, 4=water
  - `risk_map` (float32, 0–1) — population risk
  - `nfz_mask` (bool) — no-fly zones
  - `roads_graph_nodes` (float64 Nx2) — road graph nodes
  - `roads_graph_edges` (float64 Ex3) — road graph edges

### Basemap Rendering

`build_basemap(tile)` produces an RGB [H, W, 3] float32 image:
1. Landuse background (5 classes, distinct colours)
2. Roads (primary fill + secondary edge via erosion)
3. Buildings (fill + 1px outline edge via erosion)

### Cartographic Overlay (Z10)

- **Scale bar**: 200m (or 100m if too wide), bottom-left
- **North arrow**: top-left with "N" label
- **Coordinate readout**: lat/lon at bottom-right
- **OSM attribution**: "© OpenStreetMap contributors" (mandatory)

---

## Task 4 — Dynamic Overlays

### Architecture

Three overlay classes, one per mission type.  Each produces a dict of kwargs
for `StakeholderRenderer.render_frame()`.

### M1: `CivilProtectionOverlay`

| Layer | Description | Z-order |
|-------|-------------|---------|
| Fire perimeter | Expanding mask with wind bias + sinusoidal flicker | Z3 |
| Smoke drift | Dilated fire shifted downwind with oscillation | Z3 |
| POIs | Fire icons (perimeter) + camera icons (tasks) | Z5 |
| Vehicles | Emergency vehicle entities with trails | Z6 |

Fire expansion: `scipy.ndimage.binary_dilation` + wind-biased roll.

### M2: `MaritimeDomainOverlay`

| Layer | Description | Z-order |
|-------|-------------|---------|
| Vessel tracks | Moving ship icons with 30-step trail | Z6 |
| Distress beacon | Pulsing diamond icon at distress position | Z9 |
| Patrol POIs | Anchor + ship icons at waypoints | Z5 |

Vessels follow circular patrol paths with configurable radius.

### M3: `CriticalInfraOverlay`

| Layer | Description | Z-order |
|-------|-------------|---------|
| Inspection sites | Building icons at fixed sites | Z5 |
| Restriction zones | Dynamic circular NFZ masks | Z4 |
| Task POIs | Inspection (magnifier) icons | Z5 |

### Implementation

`src/uavbench/visualization/overlays.py` — `CivilProtectionOverlay`,
`MaritimeDomainOverlay`, `CriticalInfraOverlay` classes.

Factory: `create_overlay(mission_type, grid_shape, **kwargs)`.

---

## Task 5 — Exportable Demo Pack

### Pack Contents

```
demo_packs/{mission}_{difficulty}/
  ├── thumbnail.png        # First-frame thumbnail
  ├── episode.mp4          # 1080p video (libx264, CRF 18)
  ├── episode.gif          # Lightweight preview (≤150 frames)
  ├── keyframes/           # Event-driven high-DPI PNGs
  │   ├── keyframe_0000.png
  │   ├── keyframe_0100.png
  │   └── keyframe_0199.png
  ├── metadata.json        # Full episode metadata
  ├── episode_log.jsonl    # Step-by-step engine log
  └── summary.txt          # Human-readable one-page summary
```

### MP4 Specification

- Codec: H.264 (libx264)
- Resolution: 1920×1080
- CRF: 18 (visually lossless)
- Preset: slow (quality)
- Frame rate: 10 FPS
- Colour space: yuv420p

### Metadata JSON

```json
{
  "mission_type": "civil_protection",
  "profile": {
    "name": "Wildfire Monitoring",
    "name_el": "Παρακολούθηση Πυρκαγιάς",
    "agency": "GSCP (ΓΓΠΠ)",
    "tile_id": "penteli"
  },
  "tile": {
    "center_latlon": [38.0807, 23.8335],
    "resolution_m": 3.0,
    "crs": "EPSG:32634"
  },
  "total_frames": 200,
  "events": [...],
  "attribution": "Map data © OpenStreetMap contributors"
}
```

### Implementation

`src/uavbench/visualization/demo_pack.py` — `export_demo_pack()` function.

---

## Task 6 — Implementation Plan

### Step 1: Icon System ✅

**Files**: `src/uavbench/visualization/icons/__init__.py`, `icons/library.py`  
**What**: 20 path-based vector icons + `IconLibrary` stamping API  
**Validation**: Import, stamp each icon on a test axes, verify no exceptions

### Step 2: Stakeholder Renderer ✅

**Files**: `src/uavbench/visualization/stakeholder_renderer.py`  
**What**: `TileData` loader, `build_basemap()`, `StakeholderRenderer` with 4-pane
layout and 11-layer z-order  
**Validation**: Load penteli tile, render one frame, export PNG, verify 1920×1080

### Step 3: Dynamic Overlays ✅

**Files**: `src/uavbench/visualization/overlays.py`  
**What**: `CivilProtectionOverlay`, `MaritimeDomainOverlay`, `CriticalInfraOverlay`
with `create_overlay()` factory  
**Validation**: Create each overlay, call `compute(step=0)`, verify output dict keys

### Step 4: Demo Pack Exporter ✅

**Files**: `src/uavbench/visualization/demo_pack.py`  
**What**: `export_demo_pack()` producing MP4, GIF, keyframes, metadata, summary  
**Validation**: Full export pipeline, verify all files exist

### Step 5: Demo Script ✅

**Files**: `scripts/demo_stakeholder.py`  
**What**: CLI tool running all 3 missions with greedy tour path and overlay animation  
**Validation**: `python scripts/demo_stakeholder.py --preview`

### Step 6: Tests ✅

**Files**: `tests/test_stakeholder_viz.py`  
**What**: Unit tests for icons, renderer, overlays, demo pack  
**Validation**: `pytest tests/test_stakeholder_viz.py -v`

### Step 7: Design Document ✅

**Files**: `docs/STAKEHOLDER_VIS_SPEC.md` (this file)

---

## Safety & Dual-Use Constraints

1. **No tactical symbology** — icons are civilian/humanitarian pictograms
2. **No weapons or targeting** — shield icon is for "protected zone", not defence
3. **ISR terminology** — used as "infrastructure surveillance & reconnaissance",
   not military intelligence
4. **OSM attribution** — mandatory on every frame and in metadata
5. **Non-tactical colour palette** — no military green/brown/black schemes

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `src/uavbench/visualization/icons/__init__.py` | 15 | Icon system public API |
| `src/uavbench/visualization/icons/library.py` | ~360 | 20 path-based icons + IconLibrary |
| `src/uavbench/visualization/stakeholder_renderer.py` | ~700 | 4-pane renderer + tile loader |
| `src/uavbench/visualization/overlays.py` | ~340 | 3 mission overlay classes |
| `src/uavbench/visualization/demo_pack.py` | ~180 | Demo pack exporter |
| `scripts/demo_stakeholder.py` | ~300 | CLI demo runner |
| `docs/STAKEHOLDER_VIS_SPEC.md` | this | Design specification |
| `tests/test_stakeholder_viz.py` | ~200 | Unit tests |
