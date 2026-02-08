"""Fetch OSM data for Athens tiles.

Downloads buildings, roads, landuse, amenities, and no-fly zone candidates
from OpenStreetMap using osmnx, saving raw GeoDataFrames as .geojson files
for inspection and downstream rasterization.

Usage:
    python -m tools.osm_pipeline.fetch --tile downtown --output data/maps/
    python -m tools.osm_pipeline.fetch --tile all --output data/maps/
    python -m tools.osm_pipeline.fetch --list
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import osmnx as ox

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tile configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileConfig:
    """Configuration for a single geographic tile."""

    tile_id: str
    name: str
    center: tuple[float, float]  # (lat, lon)
    dist_m: float                # half-side distance in meters
    grid_size: int               # target raster resolution (pixels per side)
    description: str


TILES: dict[str, TileConfig] = {
    "penteli": TileConfig(
        tile_id="penteli",
        name="Penteli / Kifisia",
        center=(38.08, 23.83),
        dist_m=750.0,
        grid_size=500,
        description="Wildfire-urban interface zone north of Athens",
    ),
    "downtown": TileConfig(
        tile_id="downtown",
        name="Downtown Athens",
        center=(37.98, 23.73),
        dist_m=750.0,
        grid_size=500,
        description="Dense urban core around Syntagma / Monastiraki",
    ),
    "piraeus": TileConfig(
        tile_id="piraeus",
        name="Piraeus Port",
        center=(37.94, 23.64),
        dist_m=750.0,
        grid_size=500,
        description="Port and maritime zone south-west of Athens",
    ),
}

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def compute_bbox(
    center: tuple[float, float], dist_m: float
) -> tuple[float, float, float, float]:
    """Convert center (lat, lon) + half-side distance to (north, south, east, west).

    Uses a simple spherical approximation (good enough for ~1 km tiles).
    """
    lat, lon = center
    d_lat = dist_m / 111_320.0
    d_lon = dist_m / (111_320.0 * math.cos(math.radians(lat)))
    return (lat + d_lat, lat - d_lat, lon + d_lon, lon - d_lon)


# ---------------------------------------------------------------------------
# Individual fetch functions
# ---------------------------------------------------------------------------

def fetch_buildings(
    center: tuple[float, float], dist_m: float
) -> gpd.GeoDataFrame:
    """Fetch building footprints from OSM."""
    tags = {"building": True}
    log.info("  Fetching buildings...")
    gdf = ox.features_from_point(center, tags=tags, dist=dist_m)
    log.info("  -> %d building features", len(gdf))
    return gdf


def fetch_roads(
    center: tuple[float, float], dist_m: float
) -> gpd.GeoDataFrame:
    """Fetch road network edges as a GeoDataFrame."""
    log.info("  Fetching road network...")
    G = ox.graph_from_point(center, dist=dist_m, network_type="all")
    _nodes, edges = ox.graph_to_gdfs(G)
    log.info("  -> %d road segments", len(edges))
    return edges


def fetch_landuse(
    center: tuple[float, float], dist_m: float
) -> gpd.GeoDataFrame:
    """Fetch landuse and natural area polygons from OSM."""
    tags = {"landuse": True, "natural": True}
    log.info("  Fetching landuse / natural areas...")
    gdf = ox.features_from_point(center, tags=tags, dist=dist_m)
    log.info("  -> %d landuse features", len(gdf))
    return gdf


def fetch_amenities(
    center: tuple[float, float], dist_m: float
) -> gpd.GeoDataFrame:
    """Fetch amenities and critical infrastructure POIs for risk mapping."""
    tags = {
        "amenity": [
            "hospital", "clinic", "school", "university", "kindergarten",
            "fire_station", "police", "townhall",
        ],
        "building": [
            "hospital", "school", "university", "public",
            "government", "church", "cathedral",
        ],
    }
    log.info("  Fetching amenities / critical infrastructure...")
    gdf = ox.features_from_point(center, tags=tags, dist=dist_m)
    log.info("  -> %d amenity features", len(gdf))
    return gdf


def fetch_nfz(
    center: tuple[float, float], dist_m: float
) -> gpd.GeoDataFrame:
    """Fetch no-fly zone candidates (airports, military, critical infra)."""
    tags = {
        "aeroway": True,
        "military": True,
        "landuse": ["military"],
    }
    log.info("  Fetching no-fly zone candidates...")
    gdf = ox.features_from_point(center, tags=tags, dist=dist_m)
    log.info("  -> %d NFZ features", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Layer registry
# ---------------------------------------------------------------------------

# Each entry: (layer_name, fetch_function)
# Roads use a different osmnx path (graph-based), so they need special
# error handling compared to the tag-based layers.
LAYERS: list[tuple[str, object]] = [
    ("buildings", fetch_buildings),
    ("roads", fetch_roads),
    ("landuse", fetch_landuse),
    ("amenities", fetch_amenities),
    ("nfz", fetch_nfz),
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _save_gdf(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Save a GeoDataFrame to GeoJSON, handling empty frames."""
    if gdf.empty:
        # Write a valid empty FeatureCollection
        path.write_text(
            '{"type":"FeatureCollection","features":[]}',
            encoding="utf-8",
        )
    else:
        gdf.to_file(path, driver="GeoJSON")


def fetch_tile(tile: TileConfig, output_dir: Path) -> dict[str, Path]:
    """Fetch all layers for a tile and save as .geojson files.

    Returns a dict mapping layer names to their output file paths.
    """
    tile_dir = output_dir / tile.tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)

    bbox = compute_bbox(tile.center, tile.dist_m)
    resolution_m = (2.0 * tile.dist_m) / tile.grid_size

    log.info(
        "Fetching tile '%s' (%s) — center=(%.4f, %.4f), dist=%dm",
        tile.tile_id, tile.name, *tile.center, int(tile.dist_m),
    )

    results: dict[str, Path] = {}

    for layer_name, fetch_fn in LAYERS:
        out_path = tile_dir / f"{layer_name}.geojson"
        try:
            gdf = fetch_fn(tile.center, tile.dist_m)
            _save_gdf(gdf, out_path)
            results[layer_name] = out_path
        except Exception:
            log.exception("  FAILED to fetch layer '%s'", layer_name)
            # Write empty geojson so downstream doesn't break
            _save_gdf(gpd.GeoDataFrame(), out_path)
            results[layer_name] = out_path

    # Save metadata
    meta = {
        "tile_id": tile.tile_id,
        "name": tile.name,
        "center_latlon": list(tile.center),
        "dist_m": tile.dist_m,
        "bbox_nsew": list(bbox),
        "grid_size": tile.grid_size,
        "resolution_m": round(resolution_m, 4),
        "crs": "EPSG:4326",
        "osmnx_version": ox.__version__,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "description": tile.description,
    }
    meta_path = tile_dir / "fetch_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    results["meta"] = meta_path

    log.info("Tile '%s' saved to %s", tile.tile_id, tile_dir)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch OSM data for UAVBench Athens tiles.",
    )
    parser.add_argument(
        "--tile",
        type=str,
        help="Tile ID to fetch (penteli, downtown, piraeus) or 'all'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/maps"),
        help="Output directory (default: data/maps/).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_tiles",
        help="List available tiles and exit.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_tiles:
        print("Available tiles:")
        for tid, tc in TILES.items():
            res = (2.0 * tc.dist_m) / tc.grid_size
            print(
                f"  {tid:12s}  center=({tc.center[0]:.2f}, {tc.center[1]:.2f})  "
                f"{tc.grid_size}x{tc.grid_size} @ {res:.1f}m  — {tc.description}"
            )
        return

    if not args.tile:
        parser.error("--tile is required (use --list to see options)")

    # Determine which tiles to fetch
    if args.tile == "all":
        tile_ids = list(TILES.keys())
    else:
        tile_ids = [t.strip() for t in args.tile.split(",")]

    for tid in tile_ids:
        if tid not in TILES:
            print(f"ERROR: Unknown tile '{tid}'. Use --list to see options.", file=sys.stderr)
            sys.exit(1)

    # Fetch
    for tid in tile_ids:
        fetch_tile(TILES[tid], args.output)

    print(f"\nDone. Fetched {len(tile_ids)} tile(s) to {args.output}/")


if __name__ == "__main__":
    main()
