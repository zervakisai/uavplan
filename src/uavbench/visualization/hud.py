"""HUD badge computation and text rendering (VC-2, VC-3).

Computes plan status badges (NO PLAN, STALE, PLAN: Nwp) and
forced block lifecycle badges (ACTIVE, CLEARED).
Renders HUD text onto frame using pixel font.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Badge computation (VC-2, VC-3)
# ---------------------------------------------------------------------------


def compute_badges(state: dict[str, Any]) -> dict[str, str]:
    """Compute HUD badges from frame state.

    Returns dict with:
        plan_badge: str  — "NO PLAN", "STALE PLAN (reason)", or "PLAN: Nwp"
        block_badge: str — "FORCED BLOCK: ACTIVE", "FORCED BLOCK: CLEARED", or ""
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

    # VC-3: Forced block lifecycle badge
    lifecycle = state.get("forced_block_lifecycle", "none")
    if lifecycle == "active":
        block_badge = "FORCED BLOCK: ACTIVE"
    elif lifecycle == "cleared":
        block_badge = "FORCED BLOCK: CLEARED"
    else:
        block_badge = ""

    return {
        "plan_badge": plan_badge,
        "block_badge": block_badge,
    }


# ---------------------------------------------------------------------------
# Pixel font (4x6 bitmap for HUD text, deterministic)
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


def _render_text(
    frame: np.ndarray,
    text: str,
    x: int, y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
) -> None:
    """Render text onto frame at pixel (x, y) using bitmap font."""
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


def _text_width(text: str, scale: int = 1) -> int:
    """Calculate pixel width of rendered text."""
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
    """
    H, W = frame.shape[:2]
    scale = max(1, min(W // 200, 3))

    # HUD background area
    hud_h = (_CHAR_H * scale + 2) * (2 if minimal else 4) + 4
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

    # Row 1: Identity
    scenario_id = state.get("scenario_id", "")
    mission_domain = state.get("mission_domain", "")
    row1 = f"SCN: {scenario_id}  |  MSN: {mission_domain}"
    _render_text(frame, row1, tx, ty, (232, 232, 232), scale)
    ty += line_h

    # Row 2: Planner
    planner = state.get("planner_name", "")
    row2 = f"PLN: {planner}  |  MOD: {state.get('mode', 'run')}"
    _render_text(frame, row2, tx, ty, (200, 200, 200), scale)
    ty += line_h

    if minimal:
        return

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

    # Row 4: Mission
    obj_label = state.get("objective_label", "")
    dist = state.get("distance_to_task", 0)
    progress = state.get("task_progress", "")
    deliverable = state.get("deliverable_name", "")
    row4 = f"OBJ: {obj_label}  D: {int(dist)}  P: {progress}"
    _render_text(frame, row4, tx, ty, (180, 200, 220), scale)
