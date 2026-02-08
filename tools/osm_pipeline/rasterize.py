"""Rasterize OSM GeoJSON layers into numpy arrays.

Converts the vector .geojson files produced by fetch.py into fixed-size
numpy grids suitable for UAVBench environments.

Usage:
    python -m tools.osm_pipeline.rasterize --tile downtown --input data/maps/ --output data/maps/
    python -m tools.osm_pipeline.rasterize --tile all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter
from shapely.geometry import mapping

from tools.osm_pipeline.fetch import TILES

log = logging.getLogger(__name__)

# Target CRS for rasterization (UTM zone 34N covers Athens)
TARGET_CRS = "EPSG:32634"

# ---------------------------------------------------------------------------
# Height estimation defaults
# ---------------------------------------------------------------------------

FLOOR_HEIGHT_M = 3.0  # meters per floor

# Fallback heights by OSM building type tag
BUILDING_HEIGHT_DEFAULTS: dict[str, float] = {
    "residential": 12.0,
    "apartments": 18.0,
    "house": 9.0,
    "detached": 9.0,
    "commercial": 12.0,
    "retail": 9.0,
    "office": 15.0,
    "industrial": 8.0,
    "warehouse": 8.0,
    "church": 15.0,
    "cathedral": 25.0,
    "chapel": 10.0,
    "mosque": 15.0,
    "school": 12.0,
    "university": 15.0,
    "hospital": 18.0,
    "public": 12.0,
    "government": 15.0,
    "palace": 20.0,
    "hotel": 18.0,
    "garage": 4.0,
    "shed": 3.0,
    "hut": 3.0,
    "ruins": 3.0,
}
DEFAULT_BUILDING_HEIGHT = 10.0  # fallback for "yes" or unknown types

# ---------------------------------------------------------------------------
# Road width by highway type (meters)
# ---------------------------------------------------------------------------

ROAD_WIDTHS: dict[str, float] = {
    "motorway": 12.0,
    "motorway_link": 8.0,
    "trunk": 10.0,
    "trunk_link": 7.0,
    "primary": 8.0,
    "primary_link": 6.0,
    "secondary": 7.0,
    "secondary_link": 5.0,
    "tertiary": 6.0,
    "tertiary_link": 5.0,
    "residential": 5.0,
    "living_street": 4.0,
    "unclassified": 5.0,
    "pedestrian": 3.0,
    "service": 3.0,
    "footway": 2.0,
    "cycleway": 2.0,
    "path": 1.5,
    "steps": 1.5,
    "track": 3.0,
}
DEFAULT_ROAD_WIDTH = 4.0

# ---------------------------------------------------------------------------
# Landuse category mapping
# ---------------------------------------------------------------------------

LANDUSE_CATEGORIES = {
    # Category 1: forest/woodland
    "forest": 1, "wood": 1,
    # Category 2: urban/residential
    "residential": 2, "retail": 2, "grass": 2, "education": 2,
    "orchard": 2, "allotments": 2, "recreation_ground": 2,
    "village_green": 2, "flowerbed": 2, "meadow": 2,
    # Category 3: industrial/commercial
    "industrial": 3, "commercial": 3, "construction": 3,
    "railway": 3, "quarry": 3, "landfill": 3, "port": 3,
    # Category 4: water
    "water": 4, "reservoir": 4, "basin": 4,
}

NATURAL_CATEGORIES = {
    "wood": 1, "scrub": 1, "heath": 1,
    "tree": 2, "tree_row": 2,  # urban vegetation
    "water": 4, "wetland": 4,
    "bare_rock": 0, "stone": 0, "sand": 0,
}

# Priority order for burning: lower priority first, higher overwrites
LANDUSE_PRIORITY = {0: 0, 2: 1, 3: 2, 1: 3, 4: 4}


# ---------------------------------------------------------------------------
# Grid setup
# ---------------------------------------------------------------------------

def setup_grid(meta: dict) -> tuple:
    """Build an affine transform and grid shape from fetch metadata.

    Returns:
        (transform, shape, target_crs) where transform is a rasterio Affine,
        shape is (height, width), and target_crs is the CRS string.
    """
    bbox_nsew = meta["bbox_nsew"]  # [north, south, east, west]
    north, south, east, west = bbox_nsew
    grid_size = meta["grid_size"]

    # Transform bbox corners to UTM
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    utm_west, utm_south = transformer.transform(west, south)
    utm_east, utm_north = transformer.transform(east, north)

    transform = from_bounds(utm_west, utm_south, utm_east, utm_north, grid_size, grid_size)
    shape = (grid_size, grid_size)

    return transform, shape, TARGET_CRS


def _reproject(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the target CRS if needed."""
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(target_crs)


# ---------------------------------------------------------------------------
# Layer rasterizers
# ---------------------------------------------------------------------------

def _parse_height(val) -> float | None:
    """Try to parse a height value from OSM tags."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        s = str(val).strip().rstrip("m").strip()
        return float(s)
    except (ValueError, TypeError):
        return None


def _estimate_building_height(row) -> float:
    """Estimate building height from OSM tags."""
    # Priority 1: explicit height tag
    h = _parse_height(row.get("height"))
    if h is not None and h > 0:
        return h

    # Priority 2: building:levels
    levels = _parse_height(row.get("building:levels"))
    if levels is not None and levels > 0:
        return levels * FLOOR_HEIGHT_M

    # Priority 3: fallback by building type
    btype = row.get("building", "yes")
    if isinstance(btype, str):
        return BUILDING_HEIGHT_DEFAULTS.get(btype, DEFAULT_BUILDING_HEIGHT)

    return DEFAULT_BUILDING_HEIGHT


def rasterize_buildings(
    gdf: gpd.GeoDataFrame, transform, shape: tuple[int, int]
) -> np.ndarray:
    """Convert building footprints to a heightmap [H, W] in meters."""
    heightmap = np.zeros(shape, dtype=np.float32)

    if gdf.empty:
        log.info("  Buildings: empty → zero heightmap")
        return heightmap

    # Filter to polygonal geometries only
    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask].copy()

    if gdf.empty:
        return heightmap

    # Estimate heights
    gdf["_height"] = gdf.apply(_estimate_building_height, axis=1)

    # Sort ascending so tallest buildings burn last (overwrite shorter)
    gdf = gdf.sort_values("_height", ascending=True)

    shapes = [(mapping(geom), h) for geom, h in zip(gdf.geometry, gdf["_height"])]
    heightmap = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0.0,
        dtype="float32",
    )

    log.info(
        "  Buildings: %d polygons → heightmap range [%.1f, %.1f] m",
        len(shapes), heightmap.min(), heightmap.max(),
    )
    return heightmap


def _get_highway_type(row) -> str:
    """Extract a single highway type string from a row."""
    val = row.get("highway", "")
    if isinstance(val, list):
        return val[0] if val else "residential"
    if isinstance(val, str):
        return val
    return "residential"


def rasterize_roads(
    gdf: gpd.GeoDataFrame, transform, shape: tuple[int, int]
) -> tuple[np.ndarray, dict]:
    """Convert road lines to a boolean mask and a routing graph."""
    roads_mask = np.zeros(shape, dtype=bool)
    graph: dict = {"nodes": [], "edges": []}

    if gdf.empty:
        log.info("  Roads: empty → zero mask")
        return roads_mask, graph

    # Filter to LineString/MultiLineString
    line_mask = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    gdf = gdf[line_mask].copy()

    if gdf.empty:
        return roads_mask, graph

    # Buffer each road by its type-dependent width
    def _buffer_road(row):
        hwy = _get_highway_type(row)
        width = ROAD_WIDTHS.get(hwy, DEFAULT_ROAD_WIDTH)
        return row.geometry.buffer(width / 2.0, cap_style="flat")

    buffered = gdf.apply(_buffer_road, axis=1)

    shapes = [(mapping(geom), 1) for geom in buffered if not geom.is_empty]
    if shapes:
        roads_mask = rasterize(
            shapes,
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

    # Build routing graph from road network
    # Collect unique nodes and edges with lengths
    node_map: dict[tuple[float, float], int] = {}
    nodes: list[tuple[float, float]] = []
    edges: list[tuple[int, int, float]] = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        length = row.get("length", geom.length)
        if isinstance(length, (list, np.ndarray)):
            length = length[0] if len(length) > 0 else geom.length

        coords = list(geom.coords) if geom.geom_type == "LineString" else []
        if len(coords) < 2:
            continue

        start = (round(coords[0][0], 2), round(coords[0][1], 2))
        end = (round(coords[-1][0], 2), round(coords[-1][1], 2))

        for pt in (start, end):
            if pt not in node_map:
                node_map[pt] = len(nodes)
                nodes.append(pt)

        edges.append((node_map[start], node_map[end], float(length)))

    graph = {"nodes": nodes, "edges": edges}

    log.info(
        "  Roads: %d segments → mask %d pixels, graph %d nodes / %d edges",
        len(gdf), int(roads_mask.sum()), len(nodes), len(edges),
    )
    return roads_mask, graph


def _classify_landuse(row) -> int:
    """Map a landuse/natural feature to a category integer."""
    lu = row.get("landuse")
    if isinstance(lu, str) and lu in LANDUSE_CATEGORIES:
        return LANDUSE_CATEGORIES[lu]

    nat = row.get("natural")
    if isinstance(nat, str) and nat in NATURAL_CATEGORIES:
        return NATURAL_CATEGORIES[nat]

    return 0  # unknown


def rasterize_landuse(
    gdf: gpd.GeoDataFrame, transform, shape: tuple[int, int]
) -> np.ndarray:
    """Convert landuse polygons to a categorical map [H, W]."""
    landuse_map = np.zeros(shape, dtype=np.int8)

    if gdf.empty:
        log.info("  Landuse: empty → zero map")
        return landuse_map

    # Filter to polygonal geometries
    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask].copy()

    if gdf.empty:
        return landuse_map

    gdf["_category"] = gdf.apply(_classify_landuse, axis=1)
    gdf = gdf[gdf["_category"] > 0]

    if gdf.empty:
        return landuse_map

    # Sort by priority (lowest first, highest overwrites)
    gdf["_priority"] = gdf["_category"].map(LANDUSE_PRIORITY)
    gdf = gdf.sort_values("_priority", ascending=True)

    shapes = [(mapping(geom), cat) for geom, cat in zip(gdf.geometry, gdf["_category"])]
    landuse_map = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int8",
    )

    counts = {i: int((landuse_map == i).sum()) for i in range(5) if (landuse_map == i).any()}
    log.info("  Landuse: %d features → categories %s", len(shapes), counts)
    return landuse_map


def rasterize_amenities(
    gdf: gpd.GeoDataFrame, transform, shape: tuple[int, int]
) -> np.ndarray:
    """Convert amenity locations to a risk density map [H, W] in [0, 1]."""
    risk_map = np.zeros(shape, dtype=np.float32)

    if gdf.empty:
        log.info("  Amenities: empty → zero risk map")
        return risk_map

    # Get centroids in pixel space
    from rasterio.transform import rowcol

    centroids = gdf.geometry.centroid
    for pt in centroids:
        try:
            row, col = rowcol(transform, pt.x, pt.y)
        except Exception:
            continue
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            risk_map[row, col] += 1.0

    if risk_map.max() == 0:
        return risk_map

    # Apply Gaussian blur (sigma ~7 pixels ≈ 20m at 3m resolution)
    risk_map = gaussian_filter(risk_map, sigma=7.0)

    # Normalize to [0, 1]
    max_val = risk_map.max()
    if max_val > 0:
        risk_map /= max_val

    log.info(
        "  Amenities: %d features → risk map range [%.3f, %.3f]",
        len(gdf), risk_map.min(), risk_map.max(),
    )
    return risk_map


def rasterize_nfz(
    gdf: gpd.GeoDataFrame, transform, shape: tuple[int, int]
) -> np.ndarray:
    """Convert no-fly zone polygons to a boolean mask [H, W]."""
    nfz_mask = np.zeros(shape, dtype=bool)

    if gdf.empty:
        log.info("  NFZ: empty → all-False mask")
        return nfz_mask

    poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[poly_mask]

    if gdf.empty:
        return nfz_mask

    shapes = [(mapping(geom), 1) for geom in gdf.geometry]
    nfz_mask = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)

    log.info("  NFZ: %d polygons → %d pixels", len(shapes), int(nfz_mask.sum()))
    return nfz_mask


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def rasterize_tile(
    tile_id: str,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Rasterize all layers for a tile and save as .npz.

    Args:
        tile_id: e.g. "downtown"
        input_dir: directory containing {tile_id}/ subdirectory with .geojson files
        output_dir: directory to write {tile_id}.npz into

    Returns:
        Path to the output .npz file.
    """
    tile_dir = input_dir / tile_id

    # Load metadata
    meta_path = tile_dir / "fetch_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    transform, shape, target_crs = setup_grid(meta)
    log.info(
        "Rasterizing tile '%s' — %dx%d @ %.1fm resolution",
        tile_id, shape[1], shape[0], meta["resolution_m"],
    )

    # Load and reproject each layer
    def _load(name: str) -> gpd.GeoDataFrame:
        path = tile_dir / f"{name}.geojson"
        gdf = gpd.read_file(path)
        return _reproject(gdf, target_crs)

    buildings_gdf = _load("buildings")
    roads_gdf = _load("roads")
    landuse_gdf = _load("landuse")
    amenities_gdf = _load("amenities")
    nfz_gdf = _load("nfz")

    # Rasterize each layer
    heightmap = rasterize_buildings(buildings_gdf, transform, shape)
    roads_mask, roads_graph = rasterize_roads(roads_gdf, transform, shape)
    landuse_map = rasterize_landuse(landuse_gdf, transform, shape)
    risk_map = rasterize_amenities(amenities_gdf, transform, shape)
    nfz_mask = rasterize_nfz(nfz_gdf, transform, shape)

    # Save .npz
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{tile_id}.npz"

    # Encode roads_graph as arrays (np.savez can't store dicts)
    graph_nodes = np.array(roads_graph["nodes"], dtype=np.float64) if roads_graph["nodes"] else np.empty((0, 2), dtype=np.float64)
    graph_edges = np.array(roads_graph["edges"], dtype=np.float64) if roads_graph["edges"] else np.empty((0, 3), dtype=np.float64)

    np.savez_compressed(
        npz_path,
        heightmap=heightmap,
        roads_mask=roads_mask,
        landuse_map=landuse_map,
        risk_map=risk_map,
        nfz_mask=nfz_mask,
        roads_graph_nodes=graph_nodes,
        roads_graph_edges=graph_edges,
    )

    # Write rasterization metadata alongside fetch metadata
    raster_meta = {
        "tile_id": tile_id,
        "shape": list(shape),
        "resolution_m": meta["resolution_m"],
        "target_crs": target_crs,
        "transform": list(transform)[:6],
        "layers": {
            "heightmap": {"dtype": "float32", "range": [float(heightmap.min()), float(heightmap.max())]},
            "roads_mask": {"dtype": "bool", "pixels": int(roads_mask.sum())},
            "landuse_map": {"dtype": "int8", "categories": {str(i): int((landuse_map == i).sum()) for i in range(5)}},
            "risk_map": {"dtype": "float32", "range": [float(risk_map.min()), float(risk_map.max())]},
            "nfz_mask": {"dtype": "bool", "pixels": int(nfz_mask.sum())},
            "roads_graph": {"nodes": len(roads_graph["nodes"]), "edges": len(roads_graph["edges"])},
        },
        "rasterized_at": datetime.now(timezone.utc).isoformat(),
    }
    raster_meta_path = tile_dir / f"{tile_id}_raster_meta.json"
    raster_meta_path.write_text(json.dumps(raster_meta, indent=2), encoding="utf-8")

    size_mb = npz_path.stat().st_size / (1024 * 1024)
    log.info("Tile '%s' → %s (%.2f MB)", tile_id, npz_path, size_mb)
    return npz_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize OSM GeoJSON layers into numpy arrays.",
    )
    parser.add_argument(
        "--tile",
        type=str,
        required=True,
        help="Tile ID (penteli, downtown, piraeus) or 'all'.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/maps"),
        help="Input directory with tile subdirectories (default: data/maps/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/maps"),
        help="Output directory for .npz files (default: data/maps/).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tile == "all":
        tile_ids = list(TILES.keys())
    else:
        tile_ids = [t.strip() for t in args.tile.split(",")]

    for tid in tile_ids:
        if tid not in TILES:
            print(f"ERROR: Unknown tile '{tid}'.", file=sys.stderr)
            sys.exit(1)

    for tid in tile_ids:
        rasterize_tile(tid, args.input, args.output)

    print(f"\nDone. Rasterized {len(tile_ids)} tile(s).")


if __name__ == "__main__":
    main()
