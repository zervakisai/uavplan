"""Visualization overlays (z=3..z=10).

Pure-numpy drawing functions for path, trajectory, markers, fire, etc.
All drawing is deterministic (VZ-3).
"""

from __future__ import annotations

import numpy as np

from uavbench.blocking import SMOKE_BLOCKING_THRESHOLD


# ---------------------------------------------------------------------------
# Colors (RGB uint8) — Okabe-Ito colorblind-safe palette (Wong 2011)
# ---------------------------------------------------------------------------

COLOR_CYAN = np.array([86, 180, 233], dtype=np.uint8)      # #56B4E9 — planned path (sky blue)
COLOR_PATH_OUTLINE = np.array([0, 0, 0], dtype=np.uint8)   # black outline
COLOR_TRAJ_BLUE = np.array([0, 114, 178], dtype=np.uint8)  # #0072B2 — trajectory (blue)
COLOR_TRAJ_OUTLINE = np.array([255, 255, 255], dtype=np.uint8)  # white outline
COLOR_START = np.array([0, 158, 115], dtype=np.uint8)      # #009E73 — start (bluish-green)
COLOR_GOAL = np.array([240, 228, 66], dtype=np.uint8)      # #F0E442 — goal (yellow)
COLOR_AGENT = np.array([255, 140, 0], dtype=np.uint8)      # #FF8C00 — UAV (vivid orange)
COLOR_FIRE = np.array([255, 80, 20], dtype=np.uint8)        # vivid red-orange — fire
COLOR_SMOKE = np.array([160, 160, 160], dtype=np.uint8)     # #A0A0A0 — smoke (grey)
COLOR_NFZ = np.array([204, 121, 167], dtype=np.uint8)       # #CC79A7 — NFZ (reddish purple)
COLOR_TRAFFIC = np.array([230, 159, 0], dtype=np.uint8)     # #E69F00 — traffic closure (orange)
COLOR_FIRE_BUFFER = np.array([255, 180, 100], dtype=np.uint8)  # light orange — fire buffer
COLOR_DEBRIS = np.array([120, 100, 80], dtype=np.uint8)       # dark brown — structural debris
COLOR_VEHICLE = np.array([178, 34, 34], dtype=np.uint8)       # #B22222 — dark red vehicle marker

# POI icon colors (Okabe-Ito + mission-specific)
COLOR_POI_PHARMACY = np.array([0, 158, 115], dtype=np.uint8)   # #009E73 — green pharmacy cross
COLOR_POI_RESCUE = np.array([213, 94, 0], dtype=np.uint8)      # #D55E00 — vermillion rescue cross
COLOR_POI_SURVEY = np.array([0, 114, 178], dtype=np.uint8)     # #0072B2 — blue survey diamond
COLOR_POI_COMPLETED = np.array([160, 160, 160], dtype=np.uint8)  # grey for completed POIs


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


# ---------------------------------------------------------------------------
# Path overlay (z=9) — VC-1
# ---------------------------------------------------------------------------


def draw_path(
    frame: np.ndarray,
    path: list[tuple[int, int]],
    cell: int,
) -> None:
    """Draw planned path as dashed cyan line with black outline (VC-1).

    Line width scales with cell size; dashed every 3 segments.
    """
    if len(path) < 2:
        return

    points = [_cell_center(x, y, cell) for x, y in path]
    w_outer = max(5, cell + 1)
    w_inner = max(3, cell)

    for color, width in [(COLOR_PATH_OUTLINE, w_outer), (COLOR_CYAN, w_inner)]:
        for i in range(len(points) - 1):
            px0, py0 = points[i]
            px1, py1 = points[i + 1]
            # Dashed: draw 2 of every 3 segments
            if i % 3 != 2:
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
    w_outer = max(4, cell)
    w_inner = max(2, cell - 1)

    for color, width in [(COLOR_TRAJ_OUTLINE, w_outer), (COLOR_TRAJ_BLUE, w_inner)]:
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
    """Draw start marker (green circle with black outline)."""
    cx, cy = _cell_center(start_xy[0], start_xy[1], cell)
    radius = max(6, cell * 2)
    _draw_circle(frame, cx, cy, radius + 1, COLOR_PATH_OUTLINE, filled=True)
    _draw_circle(frame, cx, cy, radius, COLOR_START, filled=True)


def draw_goal(
    frame: np.ndarray,
    goal_xy: tuple[int, int],
    cell: int,
) -> None:
    """Draw goal marker (gold circle with black outline + concentric ring)."""
    cx, cy = _cell_center(goal_xy[0], goal_xy[1], cell)
    radius = max(7, int(cell * 2.5))
    # Concentric outer ring
    _draw_circle(frame, cx, cy, radius + 3, COLOR_PATH_OUTLINE, filled=False)
    _draw_circle(frame, cx, cy, radius + 1, COLOR_PATH_OUTLINE, filled=True)
    _draw_circle(frame, cx, cy, radius, COLOR_GOAL, filled=True)


# ---------------------------------------------------------------------------
# Agent icon (z=10)
# ---------------------------------------------------------------------------


def _draw_cross(
    frame: np.ndarray,
    cx: int, cy: int,
    size: int,
    color: np.ndarray,
    width: int = 1,
) -> None:
    """Draw a + cross marker at (cx, cy)."""
    H, W = frame.shape[:2]
    half = width // 2
    for i in range(-size, size + 1):
        for w in range(-half, half + 1):
            # Horizontal arm
            py, px = cy + w, cx + i
            if 0 <= py < H and 0 <= px < W:
                frame[py, px] = color
            # Vertical arm
            py, px = cy + i, cx + w
            if 0 <= py < H and 0 <= px < W:
                frame[py, px] = color


def draw_agent(
    frame: np.ndarray,
    agent_xy: tuple[int, int],
    cell: int,
) -> None:
    """Draw UAV agent icon — prominent orange disc with white cross.

    Must be visible at any zoom, distinct from trajectory (blue) and path (cyan).
    """
    cx, cy = _cell_center(agent_xy[0], agent_xy[1], cell)
    r_outer = max(12, cell * 4)
    r_inner = max(9, cell * 3)
    # White outer ring (high contrast against any background)
    _draw_circle(frame, cx, cy, r_outer + 2, np.array([255, 255, 255], dtype=np.uint8), filled=True)
    # Black border
    _draw_circle(frame, cx, cy, r_outer, np.array([0, 0, 0], dtype=np.uint8), filled=True)
    # Orange fill
    _draw_circle(frame, cx, cy, r_inner, COLOR_AGENT, filled=True)
    # White rotor cross (larger)
    cross_size = max(6, int(r_inner * 0.7))
    _draw_cross(frame, cx, cy, cross_size, np.array([255, 255, 255], dtype=np.uint8), width=max(2, cell // 2))


# ---------------------------------------------------------------------------
# POI icons (z=9.7) — mission-specific markers
# ---------------------------------------------------------------------------

# Category → icon color mapping
_POI_CATEGORY_COLORS: dict[str, np.ndarray] = {
    "pharmacy_pickup": COLOR_POI_PHARMACY,
    "delivery_point": COLOR_POI_PHARMACY,
    "rescue_site": COLOR_POI_RESCUE,
    "survey_point": COLOR_POI_SURVEY,
}


def _draw_pharmacy_icon(
    frame: np.ndarray, cx: int, cy: int, size: int, color: np.ndarray,
) -> None:
    """Draw a pharmacy cross (✚) — thick plus sign."""
    H, W = frame.shape[:2]
    arm_w = max(2, size // 3)  # arm width
    half_w = arm_w // 2
    for i in range(-size, size + 1):
        for w in range(-half_w, half_w + 1):
            # Horizontal arm
            py, px = cy + w, cx + i
            if 0 <= py < H and 0 <= px < W:
                frame[py, px] = color
            # Vertical arm
            py, px = cy + i, cx + w
            if 0 <= py < H and 0 <= px < W:
                frame[py, px] = color


def _draw_rescue_icon(
    frame: np.ndarray, cx: int, cy: int, size: int, color: np.ndarray,
) -> None:
    """Draw a rescue cross — thin cross with circle outline."""
    H, W = frame.shape[:2]
    arm_w = max(1, size // 4)
    half_w = arm_w // 2
    # Cross arms
    for i in range(-size, size + 1):
        for w in range(-half_w, half_w + 1):
            py, px = cy + w, cx + i
            if 0 <= py < H and 0 <= px < W:
                frame[py, px] = color
            py, px = cy + i, cx + w
            if 0 <= py < H and 0 <= px < W:
                frame[py, px] = color
    # Circle outline
    _draw_circle(frame, cx, cy, size + 2, color, filled=False)


def _draw_survey_icon(
    frame: np.ndarray, cx: int, cy: int, size: int, color: np.ndarray,
) -> None:
    """Draw a surveillance diamond (◆) with center dot."""
    H, W = frame.shape[:2]
    # Diamond: |dx| + |dy| <= size
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            if abs(dx) + abs(dy) <= size:
                py, px = cy + dy, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    # Fill interior, outline on border
                    if abs(dx) + abs(dy) >= size - 1:
                        frame[py, px] = color  # border
                    else:
                        # Lighter fill
                        frame[py, px] = (
                            (frame[py, px].astype(np.uint16) + color.astype(np.uint16)) >> 1
                        ).astype(np.uint8)
    # Center dot (solid)
    dot_r = max(2, size // 3)
    _draw_circle(frame, cx, cy, dot_r, color, filled=True)


def draw_task_pois(
    frame: np.ndarray,
    task_info_list: list[dict],
    cell: int,
) -> None:
    """Draw mission POI icons at z=9.7.

    Each task dict has: xy, category, status, task_id.
    Active tasks get full-color icons; completed tasks get grey.
    """
    if not task_info_list:
        return

    for task in task_info_list:
        xy = task["xy"]
        category = task.get("category", "")
        status = task.get("status", "active")
        completed = status == "completed"

        cx, cy = _cell_center(xy[0], xy[1], cell)
        size = max(9, int(cell * 3))

        # Color: grey if completed, mission-specific if active
        color = COLOR_POI_COMPLETED if completed else _POI_CATEGORY_COLORS.get(
            category, COLOR_GOAL,
        )

        # White background circle for visibility
        bg_radius = size + 3
        _draw_circle(frame, cx, cy, bg_radius + 1,
                      np.array([0, 0, 0], dtype=np.uint8), filled=True)
        _draw_circle(frame, cx, cy, bg_radius,
                      np.array([255, 255, 255], dtype=np.uint8), filled=True)

        # Mission-specific icon shape
        if category in ("pharmacy_pickup", "delivery_point"):
            _draw_pharmacy_icon(frame, cx, cy, size, color)
        elif category == "rescue_site":
            _draw_rescue_icon(frame, cx, cy, size, color)
        elif category == "survey_point":
            _draw_survey_icon(frame, cx, cy, size, color)
        else:
            # Fallback: filled circle
            _draw_circle(frame, cx, cy, size, color, filled=True)

        # Completed checkmark overlay (small ✓)
        if completed:
            check_s = max(3, size // 2)
            _draw_line(
                frame,
                cx - check_s, cy,
                cx - check_s // 3, cy + check_s,
                np.array([60, 180, 60], dtype=np.uint8), width=2,
            )
            _draw_line(
                frame,
                cx - check_s // 3, cy + check_s,
                cx + check_s, cy - check_s,
                np.array([60, 180, 60], dtype=np.uint8), width=2,
            )


# ---------------------------------------------------------------------------
# Fire overlay (z=4)
# ---------------------------------------------------------------------------


def draw_fire(
    frame: np.ndarray,
    fire_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw fire cells as vivid red-orange overlay at 80% opacity."""
    if not fire_mask.any():
        return
    fire_px = np.repeat(np.repeat(fire_mask, cell, axis=0), cell, axis=1)
    mask_3d = fire_px[:, :, np.newaxis]
    fg = COLOR_FIRE.astype(np.uint16)
    # 80% opacity: (256 - 204) = 52 for bg, 204 for fg
    blended = ((frame.astype(np.uint16) * 52 + fg * 204) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)


def draw_smoke(
    frame: np.ndarray,
    smoke_mask: np.ndarray,
    cell: int,
    alpha_256: int = 102,
) -> None:
    """Draw smoke cells as grey overlay (vectorized, z=3.5).

    alpha_256: opacity in [0, 256] scale. 102 ≈ 40% (ops), 51 ≈ 20% (paper).
    """
    smoke_bool = smoke_mask >= SMOKE_BLOCKING_THRESHOLD
    if not smoke_bool.any():
        return
    smoke_px = np.repeat(np.repeat(smoke_bool, cell, axis=0), cell, axis=1)
    mask_3d = smoke_px[:, :, np.newaxis]
    fg = COLOR_SMOKE.astype(np.uint16)
    bg_weight = 256 - alpha_256
    blended = ((frame.astype(np.uint16) * bg_weight + fg * alpha_256) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)


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
# Vehicle icons (z=6.5) — individual vehicle markers on top of traffic zones
# ---------------------------------------------------------------------------


def draw_vehicle_icons(
    frame: np.ndarray,
    vehicle_positions: np.ndarray,
    cell: int,
) -> None:
    """Draw emergency vehicle icons (top-down truck) at each vehicle position.

    vehicle_positions: int[N, 2] as (y, x) from TrafficModel.
    Each vehicle is drawn as a rectangular body with emergency light bar,
    wheel dots, and white cross marking — recognizable as a fire truck
    or emergency vehicle from above.
    """
    if vehicle_positions is None or len(vehicle_positions) == 0:
        return

    H, W = frame.shape[:2]
    # Vehicle dimensions in pixels — prominent enough to see
    half_h = max(8, cell * 3)      # half-height of truck body
    half_w = max(5, cell * 2)      # half-width of truck body
    _white = np.array([255, 255, 255], dtype=np.uint8)
    _black = np.array([0, 0, 0], dtype=np.uint8)
    _light_bar = np.array([30, 80, 220], dtype=np.uint8)   # blue emergency light
    _cabin = np.array([140, 25, 25], dtype=np.uint8)        # darker cabin
    _stripe = np.array([255, 255, 200], dtype=np.uint8)     # reflective stripe

    for vy, vx in vehicle_positions:
        cx, cy = _cell_center(int(vx), int(vy), cell)

        # 1. White outline (shadow/border) — 2px larger all around
        for dy in range(-(half_h + 2), half_h + 3):
            for dx in range(-(half_w + 2), half_w + 3):
                py, px = cy + dy, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    frame[py, px] = _white

        # 2. Black border — 1px larger
        for dy in range(-(half_h + 1), half_h + 2):
            for dx in range(-(half_w + 1), half_w + 2):
                py, px = cy + dy, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    frame[py, px] = _black

        # 3. Main body fill (dark red)
        for dy in range(-half_h, half_h + 1):
            for dx in range(-half_w, half_w + 1):
                py, px = cy + dy, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    frame[py, px] = COLOR_VEHICLE

        # 4. Cabin (front section — top 30% is darker)
        cabin_h = max(3, half_h // 3)
        for dy in range(-half_h, -half_h + cabin_h):
            for dx in range(-half_w + 1, half_w):
                py, px = cy + dy, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    frame[py, px] = _cabin

        # 5. Reflective white stripe across the middle
        stripe_y = cy
        stripe_w = max(1, half_w // 4)
        for dx in range(-half_w + 1, half_w):
            for sw in range(-stripe_w, stripe_w + 1):
                py, px = stripe_y + sw, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    frame[py, px] = _stripe

        # 6. Emergency light bar (blue rectangle on top of cabin)
        light_w = max(2, half_w // 2)
        light_h = max(1, cabin_h // 2)
        ly = cy - half_h + cabin_h // 2
        for dy in range(-light_h, light_h + 1):
            for dx in range(-light_w, light_w + 1):
                py, px = ly + dy, cx + dx
                if 0 <= py < H and 0 <= px < W:
                    frame[py, px] = _light_bar

        # 7. Wheel dots (4 corners of the body)
        wheel_r = max(2, cell)
        wheel_positions = [
            (cy - half_h + wheel_r + 1, cx - half_w - 1),   # front-left
            (cy - half_h + wheel_r + 1, cx + half_w + 1),   # front-right
            (cy + half_h - wheel_r - 1, cx - half_w - 1),   # rear-left
            (cy + half_h - wheel_r - 1, cx + half_w + 1),   # rear-right
        ]
        for wy, wx in wheel_positions:
            for ddy in range(-wheel_r, wheel_r + 1):
                for ddx in range(-wheel_r, wheel_r + 1):
                    if ddy * ddy + ddx * ddx <= wheel_r * wheel_r:
                        py, px = wy + ddy, wx + ddx
                        if 0 <= py < H and 0 <= px < W:
                            frame[py, px] = _black


# ---------------------------------------------------------------------------
# Fire buffer overlay (z=3.8) — safety buffer around active fire
# ---------------------------------------------------------------------------


def draw_fire_buffer(
    frame: np.ndarray,
    fire_mask: np.ndarray,
    buffer_radius: int,
    cell: int,
) -> None:
    """Draw fire safety buffer zone as semi-transparent light orange with dots."""
    if not fire_mask.any() or buffer_radius <= 0:
        return
    from scipy.ndimage import binary_dilation
    from uavbench.blocking import _CROSS_STRUCT
    buffer_zone = binary_dilation(fire_mask, structure=_CROSS_STRUCT, iterations=buffer_radius)
    # Buffer is the ring around fire, not the fire cells themselves
    buffer_ring = buffer_zone & ~fire_mask
    if not buffer_ring.any():
        return
    buf_px = np.repeat(np.repeat(buffer_ring, cell, axis=0), cell, axis=1)
    mask_3d = buf_px[:, :, np.newaxis]
    fg = COLOR_FIRE_BUFFER.astype(np.uint16)
    # 30% opacity: (256 - 77) = 179 for bg, 77 for fg
    blended = ((frame.astype(np.uint16) * 179 + fg * 77) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)

    # Dotted pattern for buffer zone (every 3rd pixel)
    ys, xs = np.where(buf_px)
    dots = (ys + xs) % 3 == 0
    dot_color = np.array([220, 140, 60], dtype=np.uint8)
    frame[ys[dots], xs[dots]] = dot_color


# ---------------------------------------------------------------------------
# Risk heatmap overlay (z=3.2) — continuous cost map visualization
# ---------------------------------------------------------------------------

# Risk colormap: green(0) → yellow(0.5) → red(1.0)
_RISK_GREEN = np.array([0, 180, 0], dtype=np.uint8)
_RISK_YELLOW = np.array([255, 220, 0], dtype=np.uint8)
_RISK_RED = np.array([220, 30, 0], dtype=np.uint8)


def draw_debris(
    frame: np.ndarray,
    debris_mask: np.ndarray,
    cell: int,
) -> None:
    """Draw debris cells as dark brown overlay with cross-hatching (z=4.5)."""
    if not debris_mask.any():
        return
    debris_px = np.repeat(np.repeat(debris_mask, cell, axis=0), cell, axis=1)
    mask_3d = debris_px[:, :, np.newaxis]
    fg = COLOR_DEBRIS.astype(np.uint16)
    # 85% opacity
    blended = ((frame.astype(np.uint16) * 38 + fg * 218) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)

    # Cross-hatching pattern (every 3rd pixel on both diagonals)
    ys, xs = np.where(debris_px)
    hatch = ((ys + xs) % 3 == 0) | ((ys - xs) % 3 == 0)
    hatch_color = np.array([80, 65, 50], dtype=np.uint8)
    frame[ys[hatch], xs[hatch]] = hatch_color


# ---------------------------------------------------------------------------
# Risk heatmap overlay (z=3.2) — continuous cost map visualization
# ---------------------------------------------------------------------------


def draw_risk_heatmap(
    frame: np.ndarray,
    cost_map: np.ndarray,
    cell: int,
    alpha: float = 0.4,
) -> None:
    """Draw risk cost map as green→yellow→red translucent overlay (z=3.2).

    Only draws cells with cost > 0.01 (skip zero-risk areas for clarity).
    """
    # Skip trivial maps
    if cost_map.max() < 0.01:
        return

    H, W = cost_map.shape
    # Build RGB color map: interpolate green→yellow→red
    risk_rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Green to yellow (cost 0 to 0.5)
    low = cost_map <= 0.5
    t_low = cost_map * 2.0  # normalize to [0,1]
    for c in range(3):
        risk_rgb[:, :, c] = np.where(
            low,
            (_RISK_GREEN[c] * (1.0 - t_low) + _RISK_YELLOW[c] * t_low).astype(np.uint8),
            risk_rgb[:, :, c],
        )

    # Yellow to red (cost 0.5 to 1.0)
    high = cost_map > 0.5
    t_high = (cost_map - 0.5) * 2.0  # normalize to [0,1]
    for c in range(3):
        risk_rgb[:, :, c] = np.where(
            high,
            (_RISK_YELLOW[c] * (1.0 - t_high) + _RISK_RED[c] * t_high).astype(np.uint8),
            risk_rgb[:, :, c],
        )

    # Mask: only draw where cost > 0.01
    active = cost_map > 0.01
    if not active.any():
        return

    # Upscale to pixel resolution
    risk_px = np.repeat(np.repeat(risk_rgb, cell, axis=0), cell, axis=1)
    active_px = np.repeat(np.repeat(active, cell, axis=0), cell, axis=1)

    # Alpha blend
    mask_3d = active_px[:, :, np.newaxis]
    alpha_int = int(alpha * 256)
    bg_weight = 256 - alpha_int
    blended = ((frame.astype(np.uint16) * bg_weight + risk_px.astype(np.uint16) * alpha_int) >> 8).astype(np.uint8)
    frame[:] = np.where(mask_3d, blended, frame)
