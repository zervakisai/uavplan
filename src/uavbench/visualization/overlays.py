"""Visualization overlays (z=3..z=10).

Pure-numpy drawing functions for path, trajectory, markers, fire, etc.
All drawing is deterministic (VZ-3).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Colors (RGB uint8) — Okabe-Ito colorblind-safe palette (Wong 2011)
# ---------------------------------------------------------------------------

COLOR_CYAN = np.array([86, 180, 233], dtype=np.uint8)      # #56B4E9 — planned path (sky blue)
COLOR_PATH_OUTLINE = np.array([0, 0, 0], dtype=np.uint8)   # black outline
COLOR_TRAJ_BLUE = np.array([0, 114, 178], dtype=np.uint8)  # #0072B2 — trajectory (blue)
COLOR_TRAJ_OUTLINE = np.array([255, 255, 255], dtype=np.uint8)  # white outline
COLOR_START = np.array([0, 158, 115], dtype=np.uint8)      # #009E73 — start (bluish-green)
COLOR_GOAL = np.array([240, 228, 66], dtype=np.uint8)      # #F0E442 — goal (yellow)
COLOR_AGENT = np.array([0, 114, 178], dtype=np.uint8)      # #0072B2 — UAV (blue)
COLOR_FIRE = np.array([230, 159, 0], dtype=np.uint8)       # #E69F00 — fire (orange)
COLOR_SMOKE = np.array([160, 160, 160], dtype=np.uint8)    # #A0A0A0 — smoke (grey)
COLOR_FORCED_BLOCK = np.array([213, 94, 0], dtype=np.uint8)  # #D55E00 — forced block (vermillion)
COLOR_NFZ = np.array([204, 121, 167], dtype=np.uint8)       # #CC79A7 — NFZ (reddish purple)
COLOR_TRAFFIC = np.array([230, 159, 0], dtype=np.uint8)      # #E69F00 — traffic closure (orange)
COLOR_FIRE_BUFFER = np.array([213, 94, 0], dtype=np.uint8)   # #D55E00 — fire buffer (vermillion)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cell_center(x: int, y: int, cell: int) -> tuple[int, int]:
    """Pixel center of cell (x, y)."""
    return x * cell + cell // 2, y * cell + cell // 2


def _draw_line(
    frame: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color: np.ndarray,
    width: int = 1,
) -> None:
    """Draw a line on frame using Bresenham, with width."""
    H, W = frame.shape[:2]
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    cx, cy = x0, y0
    half = width // 2

    while True:
        # Draw with width
        for wy in range(cy - half, cy + half + 1):
            for wx in range(cx - half, cx + half + 1):
                if 0 <= wy < H and 0 <= wx < W:
                    frame[wy, wx] = color
        if cx == x1 and cy == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            cx += sx
        if e2 < dx:
            err += dx
            cy += sy


def _draw_circle(
    frame: np.ndarray,
    cx: int, cy: int,
    radius: int,
    color: np.ndarray,
    filled: bool = True,
) -> None:
    """Draw a circle on frame."""
    H, W = frame.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            px, py = cx + dx, cy + dy
            if 0 <= py < H and 0 <= px < W:
                dist_sq = dx * dx + dy * dy
                if filled:
                    if dist_sq <= radius * radius:
                        frame[py, px] = color
                else:
                    if abs(dist_sq - radius * radius) <= radius:
                        frame[py, px] = color


def _draw_x(
    frame: np.ndarray,
    cx: int, cy: int,
    size: int,
    color: np.ndarray,
) -> None:
    """Draw an X marker at (cx, cy)."""
    for i in range(-size, size + 1):
        px1, py1 = cx + i, cy + i
        px2, py2 = cx + i, cy - i
        H, W = frame.shape[:2]
        if 0 <= py1 < H and 0 <= px1 < W:
            frame[py1, px1] = color
        if 0 <= py2 < H and 0 <= px2 < W:
            frame[py2, px2] = color


# ---------------------------------------------------------------------------
# Path overlay (z=9) — VC-1
# ---------------------------------------------------------------------------


def draw_path(
    frame: np.ndarray,
    path: list[tuple[int, int]],
    cell: int,
) -> None:
    """Draw planned path as dashed cyan line with black outline (VC-1).

    Uses dashed pattern: 6px on, 4px off.
    """
    if len(path) < 2:
        return

    # Draw outline first (wider, black), then cyan on top
    points = [_cell_center(x, y, cell) for x, y in path]

    for color, width in [(COLOR_PATH_OUTLINE, 3), (COLOR_CYAN, 2)]:
        for i in range(len(points) - 1):
            px0, py0 = points[i]
            px1, py1 = points[i + 1]
            # Dashed: draw every other segment
            if i % 2 == 0:
                _draw_line(frame, px0, py0, px1, py1, color, width)


# ---------------------------------------------------------------------------
# Trajectory (z=9.4-9.5)
# ---------------------------------------------------------------------------


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[tuple[int, int]],
    cell: int,
) -> None:
    """Draw agent trajectory (white outline + blue line)."""
    if len(trajectory) < 2:
        return

    points = [_cell_center(x, y, cell) for x, y in trajectory]

    for color, width in [(COLOR_TRAJ_OUTLINE, 3), (COLOR_TRAJ_BLUE, 1)]:
        for i in range(len(points) - 1):
            px0, py0 = points[i]
            px1, py1 = points[i + 1]
            _draw_line(frame, px0, py0, px1, py1, color, width)


# ---------------------------------------------------------------------------
# Markers (z=9.6)
# ---------------------------------------------------------------------------


def draw_start(
    frame: np.ndarray,
    start_xy: tuple[int, int],
    cell: int,
) -> None:
    """Draw start marker (green circle)."""
    cx, cy = _cell_center(start_xy[0], start_xy[1], cell)
    radius = max(3, cell // 2)
    _draw_circle(frame, cx, cy, radius, COLOR_START, filled=True)


def draw_goal(
    frame: np.ndarray,
    goal_xy: tuple[int, int],
    cell: int,
) -> None:
    """Draw goal marker (gold circle, slightly larger)."""
    cx, cy = _cell_center(goal_xy[0], goal_xy[1], cell)
    radius = max(4, cell // 2 + 1)
    _draw_circle(frame, cx, cy, radius, COLOR_GOAL, filled=True)


# ---------------------------------------------------------------------------
# Agent icon (z=10)
# ---------------------------------------------------------------------------


def draw_agent(
    frame: np.ndarray,
    agent_xy: tuple[int, int],
    cell: int,
) -> None:
    """Draw UAV agent icon (blue circle with white border)."""
    cx, cy = _cell_center(agent_xy[0], agent_xy[1], cell)
    radius = max(2, cell // 3)
    _draw_circle(frame, cx, cy, radius + 1, COLOR_TRAJ_OUTLINE, filled=True)
    _draw_circle(frame, cx, cy, radius, COLOR_AGENT, filled=True)


# ---------------------------------------------------------------------------
# Fire overlay (z=4)
# ---------------------------------------------------------------------------


def draw_fire(
    frame: np.ndarray,
    fire_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw fire cells as red-orange overlay (vectorized)."""
    if not fire_mask.any():
        return
    fire_px = np.repeat(np.repeat(fire_mask, cell, axis=0), cell, axis=1)
    mask_3d = fire_px[:, :, np.newaxis]
    fg = COLOR_FIRE.astype(np.uint16)
    blended = ((frame.astype(np.uint16) * 90 + fg * 166) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)


def draw_smoke(
    frame: np.ndarray,
    smoke_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw smoke cells as grey overlay (vectorized, z=3.5)."""
    smoke_bool = smoke_mask >= 0.3
    if not smoke_bool.any():
        return
    smoke_px = np.repeat(np.repeat(smoke_bool, cell, axis=0), cell, axis=1)
    mask_3d = smoke_px[:, :, np.newaxis]
    fg = COLOR_SMOKE.astype(np.uint16)
    blended = ((frame.astype(np.uint16) * 154 + fg * 102) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)


# ---------------------------------------------------------------------------
# Forced block markers (z=8) — VC-3
# ---------------------------------------------------------------------------


def draw_forced_blocks(
    frame: np.ndarray,
    forced_block_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw forced block X-markers (VC-3)."""
    ys, xs = np.where(forced_block_mask)
    size = max(2, cell // 3)
    for y, x in zip(ys, xs):
        cx, cy = _cell_center(x, y, cell)
        _draw_x(frame, cx, cy, size, COLOR_FORCED_BLOCK)


# ---------------------------------------------------------------------------
# NFZ overlay (z=5) — restriction zones
# ---------------------------------------------------------------------------


def draw_nfz(
    frame: np.ndarray,
    nfz_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw dynamic no-fly zones as purple overlay with hatching."""
    if not nfz_mask.any():
        return
    nfz_px = np.repeat(np.repeat(nfz_mask, cell, axis=0), cell, axis=1)
    mask_3d = nfz_px[:, :, np.newaxis]
    fg = COLOR_NFZ.astype(np.uint16)
    blended = ((frame.astype(np.uint16) * 128 + fg * 128) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)

    # Diagonal hatching for NFZ cells (every 4th pixel)
    H, W = frame.shape[:2]
    hatch_color = np.array([180, 80, 140], dtype=np.uint8)
    ys, xs = np.where(nfz_px)
    hatch = (ys + xs) % 4 == 0
    frame[ys[hatch], xs[hatch]] = hatch_color


# ---------------------------------------------------------------------------
# Traffic overlay (z=6) — emergency vehicles and closures
# ---------------------------------------------------------------------------


def draw_traffic(
    frame: np.ndarray,
    traffic_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw traffic closure/occupancy zones as orange overlay."""
    if not traffic_mask.any():
        return
    traf_px = np.repeat(np.repeat(traffic_mask, cell, axis=0), cell, axis=1)
    mask_3d = traf_px[:, :, np.newaxis]
    fg = COLOR_TRAFFIC.astype(np.uint16)
    blended = ((frame.astype(np.uint16) * 154 + fg * 102) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)


# ---------------------------------------------------------------------------
# Fire buffer overlay (z=3.8) — safety buffer around active fire
# ---------------------------------------------------------------------------


def draw_fire_buffer(
    frame: np.ndarray,
    fire_mask: np.ndarray,
    buffer_radius: int,
    cell: int,
) -> None:
    """Draw fire safety buffer zone as semi-transparent vermillion."""
    if not fire_mask.any() or buffer_radius <= 0:
        return
    from scipy.ndimage import binary_dilation, generate_binary_structure
    struct = generate_binary_structure(2, 1)
    buffer_zone = binary_dilation(fire_mask, structure=struct, iterations=buffer_radius)
    # Buffer is the ring around fire, not the fire cells themselves
    buffer_ring = buffer_zone & ~fire_mask
    if not buffer_ring.any():
        return
    buf_px = np.repeat(np.repeat(buffer_ring, cell, axis=0), cell, axis=1)
    mask_3d = buf_px[:, :, np.newaxis]
    fg = COLOR_FIRE_BUFFER.astype(np.uint16)
    # Lighter blend than fire itself (alpha ~0.25)
    blended = ((frame.astype(np.uint16) * 192 + fg * 64) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)
