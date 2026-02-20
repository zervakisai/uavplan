#!/usr/bin/env python3
"""Export styled basemap gallery PNGs from OSM tiles.

Usage:
    python scripts/gallery_basemap.py

Outputs to: outputs/basemap_gallery/
    <tile>_full.png      — full tile (500×500)
    <tile>_center.png    — centre crop (200×200)
    <tile>_detail.png    — detail crop (100×100)
    metadata.json        — tile info + ODbL attribution
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uavbench.visualization.basemap_style import (
    DEFAULT_STYLE,
    build_styled_basemap,
)

TILES = ["penteli", "piraeus", "downtown"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "maps"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "basemap_gallery"


def _save_rgb(rgb: np.ndarray, path: Path, dpi: int = 150) -> None:
    """Save an RGB float32 [H,W,3] array as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta: dict[str, dict] = {}

    for tile_id in TILES:
        npz_path = DATA_DIR / f"{tile_id}.npz"
        if not npz_path.exists():
            print(f"⚠️  Skipping {tile_id} — {npz_path} not found")
            continue

        print(f"\n{'─'*50}")
        print(f"  Tile: {tile_id}")
        print(f"{'─'*50}")

        data = np.load(str(npz_path))
        heightmap = data.get("heightmap", np.zeros((500, 500), dtype=np.float32))
        roads_mask = data.get("roads_mask", np.zeros((500, 500), dtype=bool))
        landuse_map = data.get("landuse_map", np.zeros((500, 500), dtype=np.int8))

        H, W = heightmap.shape
        print(f"  Grid: {H}×{W}")
        print(f"  Buildings: {int(np.sum(heightmap > 0))} cells")
        print(f"  Roads: {int(np.sum(roads_mask))} cells")
        print(f"  Landuse classes: {sorted(set(landuse_map.ravel().tolist()))}")

        t0 = time.perf_counter()
        basemap = build_styled_basemap(
            heightmap,
            roads_mask,
            landuse_map=landuse_map,
            style=DEFAULT_STYLE,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Render time: {elapsed*1000:.0f} ms")

        # Full tile
        full_path = OUT_DIR / f"{tile_id}_full.png"
        _save_rgb(basemap, full_path)
        print(f"  ✅ {full_path.name}")

        # Centre crop (200×200)
        cy, cx = H // 2, W // 2
        half = 100
        crop_center = basemap[
            max(cy - half, 0) : cy + half,
            max(cx - half, 0) : cx + half,
        ]
        center_path = OUT_DIR / f"{tile_id}_center.png"
        _save_rgb(crop_center, center_path, dpi=200)
        print(f"  ✅ {center_path.name}")

        # Detail crop (100×100) — offset from centre for variety
        dy, dx = H // 3, W // 3
        detail_half = 50
        crop_detail = basemap[
            max(dy - detail_half, 0) : dy + detail_half,
            max(dx - detail_half, 0) : dx + detail_half,
        ]
        detail_path = OUT_DIR / f"{tile_id}_detail.png"
        _save_rgb(crop_detail, detail_path, dpi=200)
        print(f"  ✅ {detail_path.name}")

        # Tile metadata
        tile_meta: dict[str, object] = {
            "tile_id": tile_id,
            "grid_size": [int(H), int(W)],
            "building_cells": int(np.sum(heightmap > 0)),
            "road_cells": int(np.sum(roads_mask)),
            "landuse_classes": sorted(set(int(v) for v in landuse_map.ravel())),
        }

        # Try loading raster_meta
        raster_meta_path = DATA_DIR / tile_id / "raster_meta.json"
        if raster_meta_path.exists():
            with open(raster_meta_path) as f:
                rm = json.load(f)
            tile_meta["crs"] = rm.get("target_crs", "EPSG:32634")
            tile_meta["resolution_m"] = rm.get("resolution_m")

        # Try loading fetch_meta
        fetch_meta_path = DATA_DIR / tile_id / "fetch_meta.json"
        if fetch_meta_path.exists():
            with open(fetch_meta_path) as f:
                fm = json.load(f)
            tile_meta["center_latlon"] = fm.get("center_latlon")

        meta[tile_id] = tile_meta

    # Write metadata
    meta_out = {
        "tiles": meta,
        "style": "default",
        "attribution": DEFAULT_STYLE.attribution_text,
        "hillshade_enabled": DEFAULT_STYLE.hillshade_enabled,
    }
    meta_path = OUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2, default=str)
    print(f"\n📋 Metadata → {meta_path.name}")

    # Summary
    pngs = sorted(OUT_DIR.glob("*.png"))
    print(f"\n{'='*50}")
    print(f"Gallery: {len(pngs)} PNGs + metadata.json")
    for p in pngs:
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
