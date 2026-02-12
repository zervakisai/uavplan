"""Publication-quality static figures for UAVBench.

All functions produce matplotlib figures at 300 DPI with tight_layout,
suitable for academic papers. Each function:
  - Takes plain numpy arrays (no model objects)
  - Saves to the specified path (PNG/PDF)
  - Closes the figure after saving to free memory
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

GridPos = Tuple[int, int]

_DPI = 300
_FONT_SIZE = 10

# Discrete landuse colormap: 0=empty, 1=forest, 2=urban, 3=industrial, 4=water
_LANDUSE_COLORS = ["#e0e0e0", "#228b22", "#b0b0b0", "#8b4513", "#4169e1"]
_LANDUSE_LABELS = ["Empty", "Forest", "Urban", "Industrial", "Water"]
_LANDUSE_CMAP = mcolors.ListedColormap(_LANDUSE_COLORS)


def _apply_paper_style() -> None:
    """Set matplotlib rcParams for publication-quality output."""
    plt.rcParams.update({
        "font.size": _FONT_SIZE,
        "axes.titlesize": _FONT_SIZE + 2,
        "axes.labelsize": _FONT_SIZE,
        "xtick.labelsize": _FONT_SIZE - 1,
        "ytick.labelsize": _FONT_SIZE - 1,
        "legend.fontsize": _FONT_SIZE - 1,
        "figure.dpi": _DPI,
        "savefig.dpi": _DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def plot_tile_overview(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    save_path: str | Path,
    *,
    roads_mask: np.ndarray | None = None,
    landuse_map: np.ndarray | None = None,
    risk_map: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Multi-panel overview of a tile's data layers."""
    _apply_paper_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, np.ndarray, object, list | None]] = [
        ("Heightmap (m)", heightmap, "hot", None),
        ("No-fly zones", no_fly.astype(float), "Reds", None),
    ]
    if roads_mask is not None:
        panels.append(("Roads", roads_mask.astype(float), "gray_r", None))
    if landuse_map is not None:
        panels.append(("Land use", landuse_map, _LANDUSE_CMAP, [0, 4]))
    if risk_map is not None:
        panels.append(("Risk (population)", risk_map, "YlOrRd", [0, 1]))

    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, (label, data, cmap, vlim) in enumerate(panels):
        ax = axes_flat[idx]
        kwargs: dict = {"cmap": cmap, "interpolation": "nearest"}
        if vlim is not None:
            kwargs["vmin"] = vlim[0]
            kwargs["vmax"] = vlim[1]
        im = ax.imshow(data, **kwargs)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=_FONT_SIZE + 4, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {save_path}")


def plot_tile_comparison(
    tiles: List[dict],
    save_path: str | Path,
    *,
    title: str = "Athens OSM tiles",
) -> None:
    """Side-by-side comparison of multiple tiles.

    Args:
        tiles: List of dicts with keys:
            label (str), heightmap (ndarray), no_fly (ndarray),
            optionally: roads_mask, path, start, goal.
    """
    _apply_paper_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(tiles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for idx, tile_data in enumerate(tiles):
        ax = axes[idx]
        hm = tile_data["heightmap"]
        nfz = tile_data["no_fly"]
        H, W = hm.shape

        ax.imshow(hm > 0, cmap="gray", alpha=0.8, interpolation="nearest")

        if tile_data.get("roads_mask") is not None:
            roads_rgba = np.zeros((*tile_data["roads_mask"].shape, 4), dtype=np.float32)
            roads_rgba[tile_data["roads_mask"], :3] = 0.2
            roads_rgba[tile_data["roads_mask"], 3] = 0.35
            ax.imshow(roads_rgba, interpolation="nearest")

        nfz_rgba = np.zeros((H, W, 4), dtype=np.float32)
        nfz_rgba[nfz, 0] = 1.0
        nfz_rgba[nfz, 3] = 0.4
        ax.imshow(nfz_rgba, interpolation="nearest")

        if tile_data.get("path"):
            px = [p[0] for p in tile_data["path"]]
            py = [p[1] for p in tile_data["path"]]
            ax.plot(px, py, linewidth=1.5, color="blue", zorder=4)

        if "start" in tile_data:
            ax.scatter(*tile_data["start"], c="green", s=60, zorder=5,
                       edgecolors="darkgreen", linewidth=1)
        if "goal" in tile_data:
            ax.scatter(*tile_data["goal"], c="gold", s=80, marker="*", zorder=5,
                       edgecolors="orange", linewidth=1)

        ax.set_title(tile_data.get("label", f"Tile {idx}"))
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=_FONT_SIZE + 2, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {save_path}")


def plot_trajectory_with_dynamics(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    path: List[GridPos],
    save_path: str | Path,
    *,
    fire_mask: np.ndarray | None = None,
    burned_mask: np.ndarray | None = None,
    vehicle_positions: np.ndarray | None = None,
    roads_mask: np.ndarray | None = None,
    risk_map: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Static snapshot: UAV path overlaid on map with fire/traffic hazards.

    fire_mask / burned_mask / vehicle_positions should be snapshots at a
    specific timestep (typically the last frame or a representative midpoint).
    """
    _apply_paper_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = heightmap.shape
    fig, ax = plt.subplots(figsize=(5, 5))

    # Background: buildings
    ax.imshow(heightmap > 0, cmap="gray", alpha=0.8, interpolation="nearest")

    # Roads
    if roads_mask is not None:
        roads_rgba = np.zeros((*roads_mask.shape, 4), dtype=np.float32)
        roads_rgba[roads_mask, :3] = 0.2
        roads_rgba[roads_mask, 3] = 0.35
        ax.imshow(roads_rgba, interpolation="nearest")

    # Risk heatmap
    if risk_map is not None:
        masked = np.ma.masked_where(risk_map < 0.01, risk_map)
        ax.imshow(masked, cmap="YlOrRd", alpha=0.25, vmin=0, vmax=1,
                  interpolation="nearest")

    # No-fly zones
    nfz_rgba = np.zeros((H, W, 4), dtype=np.float32)
    nfz_rgba[no_fly, 0] = 1.0
    nfz_rgba[no_fly, 3] = 0.4
    ax.imshow(nfz_rgba, interpolation="nearest")

    # Burned area
    if burned_mask is not None:
        burned_rgba = np.zeros((H, W, 4), dtype=np.float32)
        burned_rgba[burned_mask, :3] = 0.15
        burned_rgba[burned_mask, 3] = 0.5
        ax.imshow(burned_rgba, interpolation="nearest")

    # Active fire
    if fire_mask is not None:
        fire_rgba = np.zeros((H, W, 4), dtype=np.float32)
        fire_rgba[fire_mask, 0] = 1.0
        fire_rgba[fire_mask, 1] = 0.4
        fire_rgba[fire_mask, 3] = 0.7
        ax.imshow(fire_rgba, interpolation="nearest")

    # Traffic vehicles
    if vehicle_positions is not None and len(vehicle_positions) > 0:
        vx = vehicle_positions[:, 1]  # (y, x) -> x
        vy = vehicle_positions[:, 0]  # (y, x) -> y
        ax.scatter(vx, vy, c="dodgerblue", s=15, marker="s",
                   zorder=3, edgecolors="navy", linewidth=0.3)

    # Path
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, linewidth=1.5, color="blue", zorder=4, alpha=0.9)

    # Start / Goal
    ax.scatter(*start, c="green", s=80, zorder=5, edgecolors="darkgreen", linewidth=1.5)
    ax.scatter(*goal, c="gold", s=100, marker="*", zorder=5, edgecolors="orange", linewidth=1)

    # Legend
    legend_elements = [
        Patch(facecolor="gray", alpha=0.8, label="Buildings"),
    ]
    if roads_mask is not None:
        legend_elements.append(Patch(facecolor=(0.2, 0.2, 0.2), alpha=0.35, label="Roads"))
    if fire_mask is not None:
        legend_elements.append(Patch(facecolor=(1.0, 0.4, 0.0), alpha=0.7, label="Active fire"))
    if burned_mask is not None:
        legend_elements.append(Patch(facecolor=(0.15, 0.15, 0.15), alpha=0.5, label="Burned"))
    if vehicle_positions is not None and len(vehicle_positions) > 0:
        legend_elements.append(Patch(facecolor="dodgerblue", label="Vehicles"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.8)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    fig.savefig(str(save_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {save_path}")


def plot_fire_evolution(
    heightmap: np.ndarray,
    fire_states: List[np.ndarray],
    burned_states: List[np.ndarray],
    save_path: str | Path,
    *,
    timestep_indices: List[int] | None = None,
    ncols: int = 4,
    title: str = "Fire spread evolution",
) -> None:
    """Multi-panel figure showing fire progression at selected timesteps."""
    _apply_paper_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(fire_states)
    if timestep_indices is None:
        n_panels = min(ncols * 2, n_frames)
        timestep_indices = np.linspace(0, n_frames - 1, n_panels, dtype=int).tolist()

    n_panels = len(timestep_indices)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    H, W = heightmap.shape

    for panel_idx, t in enumerate(timestep_indices):
        ax = axes_flat[panel_idx]

        # Background
        ax.imshow(heightmap > 0, cmap="gray", alpha=0.6, interpolation="nearest")

        # Burned
        burned_rgba = np.zeros((H, W, 4), dtype=np.float32)
        burned_rgba[burned_states[t], :3] = 0.15
        burned_rgba[burned_states[t], 3] = 0.5
        ax.imshow(burned_rgba, interpolation="nearest")

        # Active fire
        fire_rgba = np.zeros((H, W, 4), dtype=np.float32)
        fire_rgba[fire_states[t], 0] = 1.0
        fire_rgba[fire_states[t], 1] = 0.4
        fire_rgba[fire_states[t], 3] = 0.8
        ax.imshow(fire_rgba, interpolation="nearest")

        fire_pct = 100.0 * fire_states[t].sum() / (H * W)
        burned_pct = 100.0 * burned_states[t].sum() / (H * W)
        ax.set_title(f"t={t}  fire={fire_pct:.1f}%  burned={burned_pct:.1f}%",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=_FONT_SIZE + 2, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {save_path}")


def plot_event_timeline(
    events: List[dict],
    save_path: str | Path,
    *,
    title: str = "Event distribution",
) -> None:
    """Stacked bar chart of event types over simulation steps."""
    _apply_paper_style()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not events:
        print("[FIGURE] No events to plot, skipping.")
        return

    event_types = sorted(set(e["type"] for e in events))
    color_map = {
        "fire_exposure": "#ff6600",
        "traffic_proximity": "#4169e1",
        "no_fly_violation_attempt": "#cc0000",
        "collision_building_attempt": "#666666",
    }

    fig, ax = plt.subplots(figsize=(6, 3))

    max_step = max(e["step"] for e in events)
    bin_size = max(1, max_step // 20)
    bins = np.arange(0, max_step + bin_size, bin_size)

    bottom = np.zeros(len(bins) - 1)
    for etype in event_types:
        steps = [e["step"] for e in events if e["type"] == etype]
        counts, _ = np.histogram(steps, bins=bins)
        color = color_map.get(etype, "#999999")
        ax.bar(bins[:-1] + bin_size / 2, counts, width=bin_size * 0.8,
               bottom=bottom, color=color, label=etype.replace("_", " "),
               edgecolor="white", linewidth=0.3)
        bottom += counts

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Event count")
    ax.legend(fontsize=7, loc="upper right")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {save_path}")
