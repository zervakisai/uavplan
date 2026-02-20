"""Exportable demo pack generator for stakeholder presentations.

Produces a self-contained directory with:
  - episode.mp4          — 1080p deterministic replay video
  - episode.gif          — lightweight preview
  - keyframes/           — high-DPI PNG keyframes (event-driven)
  - metadata.json        — episode metadata, metrics, attribution
  - summary.txt          — human-readable one-page summary
  - thumbnail.png        — first-frame thumbnail

Usage::

    from uavbench.visualization.demo_pack import export_demo_pack

    export_demo_pack(
        renderer=renderer,
        output_dir=Path("demo_packs/civil_protection_medium"),
        metrics=final_metrics,
        episode_log=engine.export_episode_log(),
    )
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def export_demo_pack(
    renderer: Any,
    output_dir: Path,
    *,
    metrics: dict[str, Any] | None = None,
    episode_log: list[dict[str, Any]] | None = None,
    mp4_fps: int = 10,
    gif_fps: int = 8,
    gif_max_frames: int = 150,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a complete demo pack from a StakeholderRenderer.

    Parameters
    ----------
    renderer : StakeholderRenderer
        Renderer with accumulated frames.
    output_dir : Path
        Output directory (will be created).
    metrics : dict, optional
        Final mission metrics.
    episode_log : list[dict], optional
        Step-by-step episode log from MissionEngine.
    mp4_fps : int
        MP4 frame rate.
    gif_fps : int
        GIF frame rate.
    gif_max_frames : int
        Maximum frames for GIF.
    extra_metadata : dict, optional
        Additional metadata to include.

    Returns
    -------
    Path
        The output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Thumbnail (first frame) ──
    if renderer._frames:
        renderer.save_frame(output_dir / "thumbnail.png", renderer._frames[0])

    # ── 2. Keyframe PNGs ──
    keyframe_dir = output_dir / "keyframes"
    keyframe_paths = renderer.export_keyframes(keyframe_dir)

    # If no keyframes were marked, export first, middle, and last frames
    if not keyframe_paths and len(renderer._frames) >= 3:
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        indices = [0, len(renderer._frames) // 2, len(renderer._frames) - 1]
        for idx in indices:
            p = keyframe_dir / f"keyframe_{idx:04d}.png"
            renderer.save_frame(p, renderer._frames[idx])
            keyframe_paths.append(p)

    # ── 3. MP4 video ──
    try:
        renderer.export_mp4(output_dir / "episode.mp4", fps=mp4_fps)
    except Exception as e:
        # ffmpeg may not be available — write a note
        (output_dir / "mp4_error.txt").write_text(
            f"MP4 export failed: {e}\n"
            f"Install ffmpeg to enable video export.\n"
        )

    # ── 4. GIF preview ──
    try:
        renderer.export_gif(
            output_dir / "episode.gif",
            fps=gif_fps,
            max_frames=gif_max_frames,
        )
    except Exception as e:
        (output_dir / "gif_error.txt").write_text(f"GIF export failed: {e}\n")

    # ── 5. Metadata JSON ──
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "UAVBench StakeholderRenderer",
        "version": "1.0.0",
    }
    if metrics:
        meta["metrics"] = _serialise(metrics)
    if extra_metadata:
        meta.update(_serialise(extra_metadata))
    renderer.export_metadata(output_dir / "metadata.json", extra=meta)

    # ── 6. Episode log ──
    if episode_log:
        with open(output_dir / "episode_log.jsonl", "w") as f:
            for entry in episode_log:
                f.write(json.dumps(_serialise(entry), ensure_ascii=False) + "\n")

    # ── 7. Summary text ──
    _write_summary(output_dir / "summary.txt", renderer, metrics, keyframe_paths)

    return output_dir


def _serialise(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
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


def _write_summary(
    path: Path,
    renderer: Any,
    metrics: dict[str, Any] | None,
    keyframe_paths: list[Path],
) -> None:
    """Write a human-readable summary."""
    profile = renderer.profile
    m = metrics or {}

    lines = [
        "=" * 60,
        f"  UAVBench Demo Pack — {profile.name}",
        f"  {profile.name_el}",
        "=" * 60,
        "",
        f"  Agency:     {profile.agency}",
        f"              {profile.agency_el}",
        f"  Mission:    {profile.name}",
        f"  Tile:       {profile.tile_id} ({renderer.tile.center_latlon[0]:.4f}°N, "
        f"{renderer.tile.center_latlon[1]:.4f}°E)",
        f"  Resolution: {renderer.tile.resolution_m} m/pixel",
        f"  Grid:       {renderer.tile.grid_size}×{renderer.tile.grid_size}",
        f"  CRS:        {renderer.tile.crs}",
        f"  Planner:    {renderer.planner_name}",
        f"  Difficulty: {renderer.difficulty}",
        "",
        "─" * 60,
        "  RESULTS",
        "─" * 60,
        "",
        f"  Total Frames:     {len(renderer._frames)}",
        f"  Keyframes:        {len(keyframe_paths)}",
        f"  Events Logged:    {len(renderer._events)}",
    ]

    if m:
        lines += [
            "",
            f"  Tasks Completed:  {m.get('tasks_completed', 'N/A')}",
            f"  Mission Score:    {m.get('mission_score', 'N/A')}",
            f"  Risk Integral:    {m.get('risk_integral', 'N/A')}",
            f"  Replans:          {m.get('replans', 'N/A')}",
            f"  Path Length:      {m.get('path_length', 'N/A')}",
        ]

    lines += [
        "",
        "─" * 60,
        "  ATTRIBUTION",
        "─" * 60,
        "",
        "  Map data © OpenStreetMap contributors",
        "  https://www.openstreetmap.org/copyright",
        "",
        f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "  Generator: UAVBench (https://github.com/zervakisai/uavbench)",
        "",
        "=" * 60,
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
