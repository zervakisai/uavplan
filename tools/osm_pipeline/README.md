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

1. **fetch** -- Query OSM via osmnx for buildings, roads, landuse, amenities, and NFZ candidates
2. **rasterize** -- Convert .geojson layers to numpy arrays (.npz)

## Available Tiles

| Tile ID | Center | Description |
|---------|--------|-------------|
| `penteli` | (38.08, 23.83) | Wildfire-urban interface north of Athens |
| `downtown` | (37.98, 23.73) | Dense urban core around Syntagma |
| `piraeus` | (37.94, 23.64) | Port and maritime zone |

Each tile covers ~1.5km x 1.5km (500x500 grid at ~3m resolution).

## Fetch Output (per tile)

Saved to `data/maps/{tile_id}/`:
- `buildings.geojson` -- building footprints with height tags
- `roads.geojson` -- road network edges
- `landuse.geojson` -- landuse and natural area polygons
- `amenities.geojson` -- hospitals, schools, critical infrastructure
- `nfz.geojson` -- airports, military zones
- `fetch_meta.json` -- bbox, CRS, resolution, timestamp

## Rasterize Output (per tile)

Saved to `data/maps/{tile_id}.npz`:
- `heightmap` -- float32 [500, 500] building heights in meters
- `roads_mask` -- bool [500, 500] road coverage
- `landuse_map` -- int8 [500, 500] (0=empty, 1=forest, 2=urban, 3=industrial, 4=water)
- `risk_map` -- float32 [500, 500] normalized [0,1] population/infrastructure density
- `nfz_mask` -- bool [500, 500] no-fly zones
- `roads_graph_nodes` -- float64 [N, 2] UTM coordinates
- `roads_graph_edges` -- float64 [E, 3] (node_i, node_j, length_m)

Metadata: `data/maps/{tile_id}/{tile_id}_raster_meta.json`

## Usage

```bash
# List available tiles
python -m tools.osm_pipeline.fetch --list

# Fetch a single tile
python -m tools.osm_pipeline.fetch --tile downtown --output data/maps/

# Fetch all 3 Athens tiles
python -m tools.osm_pipeline.fetch --tile all --output data/maps/

# Rasterize a tile (requires fetched .geojson data)
python -m tools.osm_pipeline.rasterize --tile downtown

# Rasterize all tiles
python -m tools.osm_pipeline.rasterize --tile all

# Full pipeline: fetch + rasterize
python -m tools.osm_pipeline fetch --tile all
python -m tools.osm_pipeline rasterize --tile all
```

## Environment

- Tested on macOS (Apple Silicon / M1)
- Python >= 3.10
- numpy-only at runtime; geospatial libs only here
