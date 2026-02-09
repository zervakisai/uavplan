"""Animated path playback with optional fire/traffic/roads/risk overlays.

Uses matplotlib for rendering. Two entry points:
  - play_path_window()  — interactive window with frame-by-frame drawing
  - save_path_video()   — MP4/GIF export via FuncAnimation
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

GridPos = Tuple[int, int]  # (x, y)


# --------------- Shared rendering helpers ----------------


def _setup_base_figure(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    *,
    roads_mask: np.ndarray | None = None,
    risk_map: np.ndarray | None = None,
    title: str = "UAVBench Path",
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 100,
) -> tuple:
    """Create matplotlib figure with all static background layers.

    Layer stacking order (bottom to top):
      1. heightmap > 0 as gray buildings
      2. roads_mask as dark overlay
      3. risk_map as semi-transparent heatmap
      4. no_fly as red overlay
      5. start/goal markers

    Returns (fig, ax).
    """
    H, W = heightmap.shape
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Layer 1: Buildings (gray)
    ax.imshow(heightmap > 0, cmap="gray", alpha=0.8, interpolation="nearest")

    # Layer 2: Roads
    if roads_mask is not None:
        roads_rgba = np.zeros((*roads_mask.shape, 4), dtype=np.float32)
        roads_rgba[roads_mask, :3] = 0.2
        roads_rgba[roads_mask, 3] = 0.4
        ax.imshow(roads_rgba, interpolation="nearest")

    # Layer 3: Risk heatmap
    if risk_map is not None:
        masked_risk = np.ma.masked_where(risk_map < 0.01, risk_map)
        ax.imshow(masked_risk, cmap="YlOrRd", alpha=0.3, vmin=0, vmax=1,
                  interpolation="nearest")

    # Layer 4: No-fly zones
    ax.imshow(no_fly, cmap="Reds", alpha=0.5, interpolation="nearest")

    # Layer 5: Start/Goal markers
    ax.scatter(*start, c="green", s=120, label="Start", zorder=5,
               edgecolors="darkgreen", linewidth=2)
    ax.scatter(*goal, c="gold", s=150, marker="*", label="Goal", zorder=5,
               edgecolors="orange", linewidth=1)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(False)

    return fig, ax


def _init_dynamic_overlays(
    ax: plt.Axes,
    shape: tuple[int, int],
    has_fire: bool,
    has_burned: bool,
    has_traffic: bool,
) -> dict:
    """Create matplotlib artists for dynamic overlays (updated per frame).

    Returns dict of artist handles keyed by overlay name.
    """
    artists: dict = {}
    H, W = shape

    if has_burned:
        burned_rgba = np.zeros((H, W, 4), dtype=np.float32)
        artists["burned_img"] = ax.imshow(burned_rgba, interpolation="nearest", zorder=2)

    if has_fire:
        fire_rgba = np.zeros((H, W, 4), dtype=np.float32)
        artists["fire_img"] = ax.imshow(fire_rgba, interpolation="nearest", zorder=2)

    if has_traffic:
        artists["traffic_scatter"] = ax.scatter(
            [], [], c="dodgerblue", s=30, marker="s",
            zorder=3, edgecolors="navy", linewidth=0.5,
        )

    return artists


def _update_dynamic_overlays(
    artists: dict,
    frame_idx: int,
    fire_states: list[np.ndarray] | None,
    burned_states: list[np.ndarray] | None,
    traffic_states: list[np.ndarray] | None,
) -> list:
    """Update dynamic overlay artists for the given frame index.

    Returns list of updated artists (for FuncAnimation blit mode).
    """
    updated = []

    if burned_states is not None and "burned_img" in artists:
        burned_mask = burned_states[min(frame_idx, len(burned_states) - 1)]
        H, W = burned_mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[burned_mask, 0] = 0.15
        rgba[burned_mask, 1] = 0.15
        rgba[burned_mask, 2] = 0.15
        rgba[burned_mask, 3] = 0.6
        artists["burned_img"].set_data(rgba)
        updated.append(artists["burned_img"])

    if fire_states is not None and "fire_img" in artists:
        fire_mask = fire_states[min(frame_idx, len(fire_states) - 1)]
        H, W = fire_mask.shape
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        rgba[fire_mask, 0] = 1.0
        rgba[fire_mask, 1] = 0.4
        rgba[fire_mask, 2] = 0.0
        rgba[fire_mask, 3] = 0.7
        artists["fire_img"].set_data(rgba)
        updated.append(artists["fire_img"])

    if traffic_states is not None and "traffic_scatter" in artists:
        positions = traffic_states[min(frame_idx, len(traffic_states) - 1)]
        if len(positions) > 0:
            # positions are [N, 2] as (y, x) -- scatter needs (x, y)
            xs = positions[:, 1]
            ys = positions[:, 0]
            artists["traffic_scatter"].set_offsets(np.column_stack([xs, ys]))
        else:
            artists["traffic_scatter"].set_offsets(np.empty((0, 2)))
        updated.append(artists["traffic_scatter"])

    return updated


# --------------- Public API ----------------


def play_path_window(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    path: List[GridPos],
    *,
    title: str = "UAVBench Path Player",
    fps: int = 8,
    fire_states: list[np.ndarray] | None = None,
    burned_states: list[np.ndarray] | None = None,
    traffic_states: list[np.ndarray] | None = None,
    roads_mask: np.ndarray | None = None,
    risk_map: np.ndarray | None = None,
) -> None:
    """Open a window and play the path as a live animation with dynamic overlays."""
    if not path:
        raise ValueError("Empty path: nothing to play.")

    H, W = heightmap.shape

    # Ensure interactive backend
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

    print(f"\n[VIDEO] Opening animation window...")
    print(f"  Scenario size: {W}x{H}")
    print(f"  Path length: {len(path)} steps")
    print(f"  FPS: {fps}")
    overlays = []
    if fire_states is not None:
        overlays.append("fire")
    if traffic_states is not None:
        overlays.append("traffic")
    if roads_mask is not None:
        overlays.append("roads")
    if risk_map is not None:
        overlays.append("risk")
    if overlays:
        print(f"  Overlays: {', '.join(overlays)}")
    print(f"  Backend: {matplotlib.get_backend()}")

    fig, ax = _setup_base_figure(
        heightmap, no_fly, start, goal,
        roads_mask=roads_mask, risk_map=risk_map,
        title=title, figsize=(8, 8),
    )
    fig.canvas.manager.set_window_title("UAVBench Path Animation")

    # Animated path line
    (line,) = ax.plot([], [], linewidth=3, color="blue", zorder=4,
                      marker="o", markersize=4)

    # Dynamic overlays
    overlay_artists = _init_dynamic_overlays(
        ax, (H, W),
        has_fire=fire_states is not None,
        has_burned=burned_states is not None,
        has_traffic=traffic_states is not None,
    )

    print(f"  Close the window to continue...\n")

    interval_ms = int(1000 / max(1, fps))

    try:
        for i in range(len(path)):
            xs = [p[0] for p in path[: i + 1]]
            ys = [p[1] for p in path[: i + 1]]
            line.set_data(xs, ys)

            _update_dynamic_overlays(
                overlay_artists, i,
                fire_states, burned_states, traffic_states,
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            time.sleep(interval_ms / 1000.0)

            if not plt.fignum_exists(fig.number):
                break

        print("[VIDEO] Animation finished. Keeping window open...")
        plt.show(block=True)

    except KeyboardInterrupt:
        print("[VIDEO] Animation interrupted by user.")
    except Exception as e:
        print(f"[WARNING] Animation error: {e}")
    finally:
        plt.close(fig)


def save_path_video(
    heightmap: np.ndarray,
    no_fly: np.ndarray,
    start: GridPos,
    goal: GridPos,
    path: List[GridPos],
    output_path: str | Path,
    *,
    title: str = "UAVBench Path",
    fps: int = 8,
    dpi: int = 100,
    fire_states: list[np.ndarray] | None = None,
    burned_states: list[np.ndarray] | None = None,
    traffic_states: list[np.ndarray] | None = None,
    roads_mask: np.ndarray | None = None,
    risk_map: np.ndarray | None = None,
) -> None:
    """Save an animated path visualization as MP4 or GIF with dynamic overlays.

    Attempts MP4 first (requires ffmpeg), falls back to GIF (Pillow).
    """
    if not path:
        raise ValueError("Empty path: nothing to save.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = heightmap.shape

    fig, ax = _setup_base_figure(
        heightmap, no_fly, start, goal,
        roads_mask=roads_mask, risk_map=risk_map,
        title=title, figsize=(7, 7), dpi=dpi,
    )

    # Animated path line
    (line,) = ax.plot([], [], linewidth=2, color="blue", zorder=4)

    # Dynamic overlays
    overlay_artists = _init_dynamic_overlays(
        ax, (H, W),
        has_fire=fire_states is not None,
        has_burned=burned_states is not None,
        has_traffic=traffic_states is not None,
    )

    interval_ms = int(1000 / max(1, fps))

    def init():
        line.set_data([], [])
        return (line,)

    def update(i: int):
        xs = [p[0] for p in path[: i + 1]]
        ys = [p[1] for p in path[: i + 1]]
        line.set_data(xs, ys)

        updated = _update_dynamic_overlays(
            overlay_artists, i,
            fire_states, burned_states, traffic_states,
        )

        return (line, *updated)

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(path), interval=interval_ms,
        blit=True, repeat=False,
    )

    # Try MP4 first (ffmpeg)
    if str(output_path).endswith(".mp4"):
        try:
            anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=dpi)
            print(f"Video saved: {output_path}")
            plt.close(fig)
            return
        except Exception as e:
            print(f"MP4 save failed (ffmpeg not installed): {e}")
            gif_path = str(output_path).replace(".mp4", ".gif")
            print(f"  Falling back to GIF: {gif_path}")
            output_path = Path(gif_path)

    # Save as GIF (Pillow)
    try:
        anim.save(str(output_path), writer="pillow", fps=fps, dpi=dpi)
        print(f"Animation saved: {output_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("  Ensure Pillow is installed: pip install Pillow")
    finally:
        plt.close(fig)
