"""GeoJSON + CSV export for mission products and event annotations.

Produces per-mission-type export products:

  M1 Civil Protection:
    - fire_perimeter.geojson  — validated perimeter points as GeoJSON FeatureCollection
    - corridor_status.csv     — checkpoint reachability over time
    - alert_timeline.csv      — detection → response timeline

  M2 Maritime Domain:
    - corridor_coverage.csv   — patrol segment coverage scores
    - event_response.csv      — distress event response latencies
    - vessel_tracks.geojson   — AIS-like vessel track LineStrings

  M3 Critical Infrastructure:
    - inspection_log.csv      — inspection site visit log
    - restriction_zones.geojson — dynamic restriction zone polygons

  All missions:
    - replan_log.csv          — full causal replan chain
    - trajectory.geojson      — UAV trajectory as LineString
    - event_bus.csv           — all UpdateBus events

Usage::

    from uavbench.visualization.export import export_geojson, export_csv, export_all

    export_all(result_v2, output_dir, tile_data=tile)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _to_geojson_coord(
    grid_pos: tuple[int, int],
    center_latlon: tuple[float, float] = (0.0, 0.0),
    resolution_m: float = 3.0,
    grid_size: int = 500,
) -> tuple[float, float]:
    """Convert grid (x, y) to approximate (lon, lat).

    Simple linear approximation from grid center.
    """
    lat_c, lon_c = center_latlon
    x, y = grid_pos
    # Meters from grid center
    dx_m = (x - grid_size / 2) * resolution_m
    dy_m = (grid_size / 2 - y) * resolution_m  # y inverted
    # Degrees (rough, at ~38°N: 1° lat ≈ 111km, 1° lon ≈ 87km)
    dlat = dy_m / 111_000
    dlon = dx_m / 87_000
    return (lon_c + dlon, lat_c + dlat)


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# GeoJSON exports
# ─────────────────────────────────────────────────────────────────────────────

def export_trajectory_geojson(
    trajectory: Sequence[tuple[int, int]],
    output_path: Path,
    *,
    center_latlon: tuple[float, float] = (0.0, 0.0),
    resolution_m: float = 3.0,
    grid_size: int = 500,
    properties: dict[str, Any] | None = None,
) -> Path:
    """Export UAV trajectory as GeoJSON LineString."""
    coords = [
        _to_geojson_coord(p, center_latlon, resolution_m, grid_size)
        for p in trajectory
    ]
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": _serialise(properties or {}),
    }
    geojson = {
        "type": "FeatureCollection",
        "features": [feature],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    return output_path


def export_points_geojson(
    points: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    center_latlon: tuple[float, float] = (0.0, 0.0),
    resolution_m: float = 3.0,
    grid_size: int = 500,
) -> Path:
    """Export a list of point features as GeoJSON FeatureCollection.

    Each point dict should have "x", "y" and any additional properties.
    """
    features = []
    for p in points:
        xy = (p.get("x", 0), p.get("y", 0))
        coord = _to_geojson_coord(xy, center_latlon, resolution_m, grid_size)
        props = {k: v for k, v in p.items() if k not in ("x", "y")}
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": list(coord),
            },
            "properties": _serialise(props),
        })
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CSV exports
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Export a list of dicts as CSV."""
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_serialise(row))
    return output_path


def export_replan_log_csv(
    replan_log: Sequence[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Export the full causal replan log as CSV."""
    return export_csv(replan_log, output_path)


def export_event_bus_csv(
    events: Sequence[Any],
    output_path: Path,
) -> Path:
    """Export UpdateBus event log as CSV (event_type, step, description, severity).

    Accepts either UpdateEvent objects or plain dicts.
    """
    rows = []
    for e in events:
        if isinstance(e, dict):
            rows.append({
                "event_id": e.get("event_id", ""),
                "event_type": e.get("event_type", ""),
                "step": e.get("step", 0),
                "description": e.get("description", ""),
                "severity": e.get("severity", 0.0),
                "position": str(e.get("position", "")),
            })
        else:
            rows.append({
                "event_id": getattr(e, "event_id", ""),
                "event_type": str(getattr(e, "event_type", "")),
                "step": getattr(e, "step", 0),
                "description": getattr(e, "description", ""),
                "severity": getattr(e, "severity", 0.0),
                "position": str(getattr(e, "position", "")),
            })
    return export_csv(rows, output_path)


# ─────────────────────────────────────────────────────────────────────────────
# All-in-one export
# ─────────────────────────────────────────────────────────────────────────────

def export_all(
    result: Any,  # MissionResultV2
    output_dir: Path,
    *,
    center_latlon: tuple[float, float] = (0.0, 0.0),
    resolution_m: float = 3.0,
    grid_size: int = 500,
) -> list[Path]:
    """Export all products from a MissionResultV2.

    Creates:
      - trajectory.geojson
      - replan_log.csv
      - Per-mission product CSVs and GeoJSONs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. Trajectory GeoJSON
    if hasattr(result, "trajectory") and result.trajectory:
        p = export_trajectory_geojson(
            result.trajectory,
            output_dir / "trajectory.geojson",
            center_latlon=center_latlon,
            resolution_m=resolution_m,
            grid_size=grid_size,
            properties={
                "mission_id": result.mission_id,
                "planner": result.planner_id,
                "difficulty": result.difficulty,
                "step_count": result.step_count,
            },
        )
        written.append(p)

    # 2. Replan log CSV
    if hasattr(result, "replan_log") and result.replan_log:
        written.append(
            export_replan_log_csv(result.replan_log, output_dir / "replan_log.csv")
        )

    # 3. Per-product-type CSV export
    for product_type, rows in result.products.items():
        if not rows:
            continue
        p = output_dir / product_type
        written.append(export_csv(rows, p))

    # 4. Fire perimeter GeoJSON (M1)
    if result.mission_id == "civil_protection":
        perimeter_points = result.products.get("fire_perimeter.geojson", [])
        if perimeter_points:
            written.append(export_points_geojson(
                perimeter_points,
                output_dir / "fire_perimeter.geojson",
                center_latlon=center_latlon,
                resolution_m=resolution_m,
                grid_size=grid_size,
            ))

    # 5. Metrics JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_serialise(result.metrics), f, indent=2)
    written.append(metrics_path)

    return written
