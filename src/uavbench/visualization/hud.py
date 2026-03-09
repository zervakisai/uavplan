"""HUD badge computation and text rendering (VC-2).

Computes plan status badges (NO PLAN, STALE, PLAN: Nwp).
Renders HUD text onto frame using PIL/Pillow for crisp text,
with bitmap font fallback.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# PIL font setup (module-level cache for determinism + performance)
# ---------------------------------------------------------------------------

_pil_available: bool = False
_font_cache: dict[int, Any] = {}  # size → PIL font
_font_path: str | None = None     # resolved path (found once, reused)

try:
    from PIL import Image, ImageDraw, ImageFont

    _pil_available = True

    # System monospace fonts in preference order
    _FONT_CANDIDATES = [
        "/System/Library/Fonts/Menlo.ttc",                        # macOS
        "Menlo.ttc",                                               # macOS (short)
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",  # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",       # Linux
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",  # Linux alt
        "DejaVuSansMono.ttf",
        "Consolas.ttf",                                            # Windows
    ]

    def _load_pil_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load monospace PIL font at given size, cached per size."""
        global _font_path
        if size in _font_cache:
            return _font_cache[size]

        # If we already know which font file works, use it directly
        if _font_path is not None:
            try:
                _font_cache[size] = ImageFont.truetype(_font_path, size)
                return _font_cache[size]
            except (OSError, IOError):
                _font_path = None  # invalidate

        # Probe candidates
        for name in _FONT_CANDIDATES:
            try:
                font = ImageFont.truetype(name, size)
                _font_path = name
                _font_cache[size] = font
                return font
            except (OSError, IOError):
                continue

        # Fall back to PIL default bitmap font
        _font_cache[size] = ImageFont.load_default()
        return _font_cache[size]

except ImportError:
    _pil_available = False


# ---------------------------------------------------------------------------
# Planner display names (human-readable for paper_min HUD)
# ---------------------------------------------------------------------------

from uavbench.visualization.labels import PLANNER_LABELS as PLANNER_DISPLAY


# ---------------------------------------------------------------------------
# Badge computation (VC-2)
# ---------------------------------------------------------------------------


def compute_badges(state: dict[str, Any]) -> dict[str, str]:
    """Compute HUD badges from frame state.

    Returns dict with:
        plan_badge: str  — "NO PLAN", "STALE PLAN (reason)", or "PLAN: Nwp"
        block_badge: str — always "" (forced blocks removed)
    """
    # VC-2: Plan status badge
    plan_len = state.get("plan_len", 0)
    plan_age = state.get("plan_age_steps", 0)
    plan_reason = state.get("plan_reason", "")
    replan_every = state.get("replan_every_steps", 6)

    if plan_len <= 1:
        plan_badge = "NO PLAN"
    elif plan_age > 2 * replan_every:
        reason_str = f" ({plan_reason})" if plan_reason else ""
        plan_badge = f"STALE PLAN{reason_str}"
    else:
        plan_badge = f"PLAN: {plan_len}wp"

    return {
        "plan_badge": plan_badge,
        "block_badge": "",
    }


# ---------------------------------------------------------------------------
# Pixel font (4x6 bitmap for HUD text, deterministic) — FALLBACK
# ---------------------------------------------------------------------------

# Minimal 4x6 bitmap font for ASCII 32-126.
# Each char is 4px wide, 6px tall. Stored as bit patterns.
_CHAR_W = 4
_CHAR_H = 6
_CHAR_SPACING = 1


def _render_char(ch: str) -> np.ndarray:
    """Return 6x4 bool array for character (monospace bitmap)."""
    # Simple fixed-width font using numpy
    bitmap = np.zeros((_CHAR_H, _CHAR_W), dtype=bool)
    o = ord(ch)

    # Only render printable ASCII
    if o < 32 or o > 126:
        return bitmap

    # Hash-based pseudo-bitmap: generate deterministic pixel pattern
    # For letters/digits, fill more pixels; for space, leave empty
    if ch == ' ':
        return bitmap

    # Use a simple rule: for common chars, define patterns
    patterns = _get_font_patterns()
    if ch in patterns:
        rows = patterns[ch]
        for r, row_val in enumerate(rows):
            if r >= _CHAR_H:
                break
            for c in range(_CHAR_W):
                if row_val & (1 << (_CHAR_W - 1 - c)):
                    bitmap[r, c] = True
    else:
        # Fallback: block character
        bitmap[1:5, 0:3] = True

    return bitmap


def _get_font_patterns() -> dict[str, list[int]]:
    """Return 4px-wide bitmap patterns for common chars."""
    # Each int is a 4-bit row (MSB=left). 6 rows per char.
    return {
        'A': [0b0110, 0b1001, 0b1111, 0b1001, 0b1001, 0b0000],
        'B': [0b1110, 0b1001, 0b1110, 0b1001, 0b1110, 0b0000],
        'C': [0b0111, 0b1000, 0b1000, 0b1000, 0b0111, 0b0000],
        'D': [0b1110, 0b1001, 0b1001, 0b1001, 0b1110, 0b0000],
        'E': [0b1111, 0b1000, 0b1110, 0b1000, 0b1111, 0b0000],
        'F': [0b1111, 0b1000, 0b1110, 0b1000, 0b1000, 0b0000],
        'G': [0b0111, 0b1000, 0b1011, 0b1001, 0b0111, 0b0000],
        'H': [0b1001, 0b1001, 0b1111, 0b1001, 0b1001, 0b0000],
        'I': [0b1110, 0b0100, 0b0100, 0b0100, 0b1110, 0b0000],
        'K': [0b1001, 0b1010, 0b1100, 0b1010, 0b1001, 0b0000],
        'L': [0b1000, 0b1000, 0b1000, 0b1000, 0b1111, 0b0000],
        'M': [0b1001, 0b1111, 0b1111, 0b1001, 0b1001, 0b0000],
        'N': [0b1001, 0b1101, 0b1011, 0b1001, 0b1001, 0b0000],
        'O': [0b0110, 0b1001, 0b1001, 0b1001, 0b0110, 0b0000],
        'P': [0b1110, 0b1001, 0b1110, 0b1000, 0b1000, 0b0000],
        'R': [0b1110, 0b1001, 0b1110, 0b1010, 0b1001, 0b0000],
        'S': [0b0111, 0b1000, 0b0110, 0b0001, 0b1110, 0b0000],
        'T': [0b1111, 0b0100, 0b0100, 0b0100, 0b0100, 0b0000],
        'U': [0b1001, 0b1001, 0b1001, 0b1001, 0b0110, 0b0000],
        'V': [0b1001, 0b1001, 0b1001, 0b0110, 0b0110, 0b0000],
        'W': [0b1001, 0b1001, 0b1111, 0b1111, 0b1001, 0b0000],
        'X': [0b1001, 0b1001, 0b0110, 0b1001, 0b1001, 0b0000],
        'Y': [0b1001, 0b1001, 0b0110, 0b0100, 0b0100, 0b0000],
        'Z': [0b1111, 0b0001, 0b0110, 0b1000, 0b1111, 0b0000],
        'J': [0b0011, 0b0001, 0b0001, 0b1001, 0b0110, 0b0000],
        'Q': [0b0110, 0b1001, 0b1001, 0b1011, 0b0111, 0b0000],
        '0': [0b0110, 0b1001, 0b1001, 0b1001, 0b0110, 0b0000],
        '1': [0b0100, 0b1100, 0b0100, 0b0100, 0b1110, 0b0000],
        '2': [0b0110, 0b1001, 0b0010, 0b0100, 0b1111, 0b0000],
        '3': [0b1110, 0b0001, 0b0110, 0b0001, 0b1110, 0b0000],
        '4': [0b1001, 0b1001, 0b1111, 0b0001, 0b0001, 0b0000],
        '5': [0b1111, 0b1000, 0b1110, 0b0001, 0b1110, 0b0000],
        '6': [0b0110, 0b1000, 0b1110, 0b1001, 0b0110, 0b0000],
        '7': [0b1111, 0b0001, 0b0010, 0b0100, 0b0100, 0b0000],
        '8': [0b0110, 0b1001, 0b0110, 0b1001, 0b0110, 0b0000],
        '9': [0b0110, 0b1001, 0b0111, 0b0001, 0b0110, 0b0000],
        ':': [0b0000, 0b0100, 0b0000, 0b0100, 0b0000, 0b0000],
        '|': [0b0100, 0b0100, 0b0100, 0b0100, 0b0100, 0b0000],
        '/': [0b0001, 0b0010, 0b0100, 0b1000, 0b0000, 0b0000],
        '.': [0b0000, 0b0000, 0b0000, 0b0000, 0b0100, 0b0000],
        ',': [0b0000, 0b0000, 0b0000, 0b0100, 0b1000, 0b0000],
        '-': [0b0000, 0b0000, 0b1111, 0b0000, 0b0000, 0b0000],
        '_': [0b0000, 0b0000, 0b0000, 0b0000, 0b1111, 0b0000],
        '(': [0b0010, 0b0100, 0b0100, 0b0100, 0b0010, 0b0000],
        ')': [0b0100, 0b0010, 0b0010, 0b0010, 0b0100, 0b0000],
        '%': [0b1001, 0b0010, 0b0100, 0b1001, 0b0000, 0b0000],
        '=': [0b0000, 0b1111, 0b0000, 0b1111, 0b0000, 0b0000],
        '+': [0b0000, 0b0100, 0b1110, 0b0100, 0b0000, 0b0000],
        '!': [0b0100, 0b0100, 0b0100, 0b0000, 0b0100, 0b0000],
        '?': [0b0110, 0b1001, 0b0010, 0b0000, 0b0010, 0b0000],
        '#': [0b1010, 0b1111, 0b1010, 0b1111, 0b1010, 0b0000],
        '*': [0b0100, 0b1110, 0b0100, 0b1010, 0b0000, 0b0000],
        '<': [0b0010, 0b0100, 0b1000, 0b0100, 0b0010, 0b0000],
        '>': [0b1000, 0b0100, 0b0010, 0b0100, 0b1000, 0b0000],
        '[': [0b0110, 0b0100, 0b0100, 0b0100, 0b0110, 0b0000],
        ']': [0b0110, 0b0010, 0b0010, 0b0010, 0b0110, 0b0000],
    }


def _render_text_bitmap(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
) -> None:
    """Render text onto frame at pixel (x, y) using bitmap font (fallback)."""
    H, W = frame.shape[:2]
    cx = x
    color_arr = np.array(color, dtype=np.uint8)

    for ch in text.upper():
        bitmap = _render_char(ch)
        for r in range(_CHAR_H):
            for c in range(_CHAR_W):
                if bitmap[r, c]:
                    for sy in range(scale):
                        for sx in range(scale):
                            py = y + r * scale + sy
                            px = cx + c * scale + sx
                            if 0 <= py < H and 0 <= px < W:
                                frame[py, px] = color_arr
        cx += (_CHAR_W + _CHAR_SPACING) * scale


def _render_text(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
) -> None:
    """Render text onto frame using PIL/Pillow for crisp text.

    Falls back to bitmap font if PIL is unavailable.
    Deterministic for same font+size+text (VZ-3).
    """
    if not _pil_available:
        _render_text_bitmap(frame, text, x, y, color, scale)
        return

    # Compute font size proportional to bitmap scale
    font_size = max(10, scale * 7)
    font = _load_pil_font(font_size)

    # Render text into a temporary RGBA image, then composite onto frame
    H, W = frame.shape[:2]

    # Measure text dimensions
    # Create a scratch image to measure
    scratch = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(scratch)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + 2
    th = bbox[3] - bbox[1] + 2

    if tw <= 0 or th <= 0:
        return

    # Render text onto a small image
    txt_img = Image.new("RGB", (tw, th), (0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_img)
    txt_draw.text((-bbox[0], -bbox[1]), text, fill=color, font=font)
    txt_arr = np.array(txt_img, dtype=np.uint8)

    # Composite onto frame (overwrite non-black pixels)
    # Clip to frame bounds
    src_y0, src_x0 = 0, 0
    dst_y0, dst_x0 = y, x
    dst_y1 = min(y + th, H)
    dst_x1 = min(x + tw, W)

    if dst_y0 < 0:
        src_y0 = -dst_y0
        dst_y0 = 0
    if dst_x0 < 0:
        src_x0 = -dst_x0
        dst_x0 = 0

    copy_h = dst_y1 - dst_y0
    copy_w = dst_x1 - dst_x0
    if copy_h <= 0 or copy_w <= 0:
        return

    src_region = txt_arr[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    # Mask: any channel > 0 means text pixel
    mask = np.any(src_region > 0, axis=-1)
    dst_region = frame[dst_y0:dst_y1, dst_x0:dst_x1]
    dst_region[mask] = src_region[mask]


def _text_width(text: str, scale: int = 1) -> int:
    """Calculate pixel width of rendered text."""
    if _pil_available:
        font_size = max(10, scale * 7)
        font = _load_pil_font(font_size)
        try:
            return int(font.getlength(text))
        except AttributeError:
            # Older Pillow versions without getlength
            pass
    return len(text) * (_CHAR_W + _CHAR_SPACING) * scale


# ---------------------------------------------------------------------------
# HUD rendering (z=12)
# ---------------------------------------------------------------------------


def render_hud_text(
    frame: np.ndarray,
    state: dict[str, Any],
    badges: dict[str, str],
    minimal: bool = False,
) -> None:
    """Render HUD text box onto frame (z=12).

    Draws semi-transparent background with badge text.
    In minimal mode (paper_min): single row with planner, step, replans.
    In full mode (ops_full): 4-row HUD with mission details.
    """
    H, W = frame.shape[:2]
    scale = max(2, W // 200)

    if minimal:
        # Paper-min mode: single compact row
        planner = state.get("planner_name", "")
        display_name = PLANNER_DISPLAY.get(planner, planner)
        step = state.get("step_idx", 0)
        replans = state.get("replans", 0)
        row = f"{display_name}  |  Step {step}  |  {replans} replans"

        # Compute font size for height calculation
        if _pil_available:
            font_size = max(10, scale * 7)
        else:
            font_size = _CHAR_H * scale
        hud_h = font_size + 8  # font_size + 8px padding

        hud_y = 2
        hud_x = 2
        hud_w = min(W - 4, W)

        # Draw semi-transparent background
        y_end = min(hud_y + hud_h, H)
        x_end = min(hud_x + hud_w, W)
        bg_region = frame[hud_y:y_end, hud_x:x_end].astype(np.uint16)
        bg_color = np.array([10, 15, 26], dtype=np.uint16)
        frame[hud_y:y_end, hud_x:x_end] = (
            (bg_region * 46 + bg_color * 210) >> 8
        ).astype(np.uint8)

        tx = hud_x + 4
        ty = hud_y + 4
        _render_text(frame, row, tx, ty, (232, 232, 232), scale)
        return

    # Full mode (ops_full): 4-row HUD
    hud_h = (_CHAR_H * scale + 2) * 4 + 4
    hud_y = 2
    hud_x = 2
    hud_w = min(W - 4, W)

    # Draw semi-transparent background (#0A0F1A, alpha=210/256 ~= 0.82)
    # Integer-only blend for cross-platform determinism
    bg_region = frame[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w].astype(np.uint16)
    bg_color = np.array([10, 15, 26], dtype=np.uint16)
    frame[hud_y:hud_y + hud_h, hud_x:hud_x + hud_w] = (
        (bg_region * 46 + bg_color * 210) >> 8
    ).astype(np.uint8)

    line_h = _CHAR_H * scale + 2
    tx = hud_x + 4
    ty = hud_y + 2

    # Row 1: Mission type + priority (short for screen fit)
    mission_domain = state.get("mission_domain", "")
    priority = state.get("priority", "")
    _prio_abbr = {"critical": "CRIT", "high": "HIGH", "normal": "NORM", "low": "LOW"}
    priority_tag = f" [{_prio_abbr.get(priority, priority.upper()[:4])}]" if priority else ""
    row1 = f"{mission_domain.upper().replace('_', ' ')}{priority_tag}"
    _render_text(frame, row1, tx, ty, (255, 220, 100), scale)
    ty += line_h

    # Row 2: Origin → Destination + Planner
    origin = state.get("origin_name", "")
    destination = state.get("destination_name", "")
    planner = state.get("planner_name", "")
    if origin and destination:
        row2 = f"{origin} > {destination}  |  PLN: {planner}"
    else:
        scenario_id = state.get("scenario_id", "")
        row2 = f"SCN: {scenario_id}  |  PLN: {planner}"
    _render_text(frame, row2, tx, ty, (200, 200, 200), scale)
    ty += line_h

    # Row 3: Live metrics + badges
    step = state.get("step_idx", 0)
    replans = state.get("replans", 0)
    plan_badge = badges.get("plan_badge", "")
    block_badge = badges.get("block_badge", "")

    row3 = f"T: {step}  |  REP: {replans}  |  {plan_badge}"
    if block_badge:
        row3 += f"  |  {block_badge}"
    _render_text(frame, row3, tx, ty, (232, 232, 232), scale)
    ty += line_h

    # Row 4: Distance + progress + deliverable
    dist = state.get("distance_to_task", 0)
    progress = state.get("task_progress", "")
    deliverable = state.get("deliverable_name", "")
    row4 = f"DIST: {int(dist)}  |  TASKS: {progress}  |  {deliverable}"
    _render_text(frame, row4, tx, ty, (180, 200, 220), scale)
