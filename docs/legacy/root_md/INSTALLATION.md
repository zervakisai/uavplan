# Installation

## Basic Install

```bash
git clone https://github.com/uavbench/uavbench.git
cd uavbench
pip install -e .
```

## With Visualization

```bash
pip install -e ".[viz]"
```

## With OSM Pipeline (tile generation)

```bash
pip install -e ".[pipeline]"
```

## All Dependencies

```bash
pip install -e ".[all]"
```

## Verify Installation

```bash
# Check CLI works
uavbench --help

# Run a quick benchmark (synthetic map, no tiles needed)
uavbench --scenarios urban_easy --planners astar --trials 1
```

## Tile Generation

OSM tiles are pre-rasterized `.npz` files in `data/maps/`. They are gitignored due to size. To generate them:

```bash
# Requires [pipeline] extras
pip install -e ".[pipeline]"

# Fetch and rasterize all 3 Athens tiles
for tile in downtown penteli piraeus; do
    python -m tools.osm_pipeline.fetch --tile $tile
    python -m tools.osm_pipeline.rasterize --tile $tile
done
```

Each tile produces a 500x500 grid at 3m/pixel containing:
- `heightmap` — building heights in meters
- `roads_mask` — boolean road network
- `landuse_map` — land use categories (int8)
- `risk_map` — risk heatmap [0, 1]
- `nfz_mask` — no-fly zones (boolean)
- `roads_graph_nodes` / `roads_graph_edges` — road network graph

## Verify Tiles

```bash
python -c "
import numpy as np
for tile in ['downtown', 'penteli', 'piraeus']:
    d = np.load(f'data/maps/{tile}.npz')
    print(f'{tile}: {list(d.keys())}, shape={d[\"heightmap\"].shape}')
"
```

## Troubleshooting

**ModuleNotFoundError: matplotlib**
```bash
pip install -e ".[viz]"
```

**OSM tile not found**
```bash
# Generate the missing tile
python -m tools.osm_pipeline.fetch --tile downtown
python -m tools.osm_pipeline.rasterize --tile downtown
```

**ffmpeg not found (MP4 export)**
```bash
# macOS
brew install ffmpeg
# Ubuntu
sudo apt install ffmpeg
```
