# tools/osm_pipeline/

Offline preprocessing scripts that convert OpenStreetMap data into
numpy tile files (.npz) for UAVBench realistic urban scenarios.

## Dependencies

These scripts require heavy geospatial libraries that are NOT needed
at runtime. Install them via:

```bash
pip install -e ".[pipeline]"
```

This installs: osmnx, geopandas, shapely, rasterio, scipy.

## Pipeline Overview

1. **fetch** -- Query OSM via osmnx for buildings, roads, and POIs within a bounding box
2. **rasterize** -- Convert vector geometries to fixed-resolution grid arrays
3. **tile** -- Split large areas into fixed-size tiles suitable for UAVBench environments
4. **export** -- Save processed tiles as .npz files with metadata

## Output Format

Each `.npz` tile contains:
- `heightmap`: float32 [H, W] -- building heights (meters)
- `road_mask`: bool [H, W] -- road cells
- `building_mask`: bool [H, W] -- building footprint cells
- `metadata`: dict -- bbox, CRS, resolution, timestamp

## Usage

```bash
# Fetch and process a tile for central Athens
python -m tools.osm_pipeline.fetch \
    --bbox 37.97,23.72,37.98,23.73 \
    --resolution 1.0 \
    --out data/maps/athens_center.npz
```

## Environment

- Tested on macOS (Apple Silicon / M1)
- Python >= 3.10
- numpy-only at runtime; geospatial libs only here
