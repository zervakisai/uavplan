"""Mission HUD overlay for UAVBench visualization.

Renders a 4-line mission status overlay on matplotlib axes:

    MISSION: [objective] → [destination_name]
    STATUS:  EN_ROUTE | SERVICING | COMPLETED | FAILED  |  Dist: N cells
    PLAN:    ACTIVE | NO_PLAN | STALE
    STEP:    N / max_time_steps

Integrates with both OperationalRenderer and StakeholderRenderer.
Degrades gracefully: shows "NO MISSION" when no briefing is available.

Contract: VC-1 (plan visible in frame), VC-2 (NO_PLAN / STALE badge).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import matplotlib.patheffects as path_effects

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False


# ── Status constants ─────────────────────────────────────────────────────

MISSION_STATUS_EN_ROUTE = "EN_ROUTE"
MISSION_STATUS_SERVICING = "SERVICING"
MISSION_STATUS_COMPLETED = "COMPLETED"
MISSION_STATUS_FAILED = "FAILED"

PLAN_STATUS_ACTIVE = "ACTIVE"
PLAN_STATUS_NO_PLAN = "NO_PLAN"
PLAN_STATUS_STALE = "STALE"


# ── HUD state ────────────────────────────────────────────────────────────

@dataclass
class MissionHUDState:
    """Per-frame HUD state passed from the episode loop.

    All fields have safe defaults so callers can omit any they lack.
    """

    objective: str = ""
    destination_name: str = ""
    mission_status: str = MISSION_STATUS_EN_ROUTE
    distance_to_goal: int = 0
    plan_status: str = PLAN_STATUS_ACTIVE
    step: int = 0
    max_steps: int = 0


# ── HUD renderer ─────────────────────────────────────────────────────────

class MissionHUD:
    """Renders the 4-line mission HUD overlay on a matplotlib axes.

    Parameters
    ----------
    has_briefing : bool
        True when a MissionBriefing was generated for this episode.
        False → the HUD shows a single "NO MISSION" badge.
    bg_color : str
        Background colour of the HUD box.
    text_color : str
        Primary text colour.
    accent_color : str
        Colour for status badges (EN_ROUTE, ACTIVE, etc.).
    warn_color : str
        Colour for warning badges (STALE, FAILED).
    """

    def __init__(
        self,
        has_briefing: bool = True,
        *,
        bg_color: str = "#0A0F1A",
        text_color: str = "#E8E8E8",
        accent_color: str = "#00DDFF",
        warn_color: str = "#FF6B35",
    ) -> None:
        self.has_briefing = has_briefing
        self.bg_color = bg_color
        self.text_color = text_color
        self.accent_color = accent_color
        self.warn_color = warn_color

    # ── Public API ────────────────────────────────────────────────────

    def format_lines(self, state: MissionHUDState) -> str:
        """Build the 4-line HUD text from episode state.

        Returns
        -------
        str
            Multi-line string ready for matplotlib ``ax.text()``.
        """
        if not self.has_briefing:
            return "NO MISSION"

        # Line 1: MISSION
        obj = state.objective or "—"
        dest = state.destination_name or "—"
        line1 = f"MISSION: {obj} → {dest}"

        # Line 2: STATUS + distance
        line2 = f"STATUS: {state.mission_status}  |  Dist: {state.distance_to_goal} cells"

        # Line 3: PLAN
        line3 = f"PLAN: {state.plan_status}"

        # Line 4: STEP
        line4 = f"STEP: {state.step} / {state.max_steps}"

        return "\n".join([line1, line2, line3, line4])

    def draw(
        self,
        ax: Any,
        state: MissionHUDState,
        *,
        x: float = 0.99,
        y: float = 0.995,
        fontsize: float = 6.5,
        alpha: float = 0.82,
        ha: str = "right",
    ) -> None:
        """Render the mission HUD on *ax* as a text box.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes (map pane).
        state : MissionHUDState
            Current frame state.
        x, y : float
            Axes-normalised position (default: top-right corner).
        fontsize : float
            Font size for the HUD text.
        alpha : float
            Background box opacity.
        ha : str
            Horizontal alignment ("right" for top-right placement).
        """
        if not HAS_MATPLOTLIB:
            return  # pragma: no cover

        hud_text = self.format_lines(state)

        # Choose border colour: warn if FAILED or STALE, accent otherwise
        if state.mission_status == MISSION_STATUS_FAILED or state.plan_status == PLAN_STATUS_STALE:
            border_color = self.warn_color
        else:
            border_color = self.accent_color

        txt = ax.text(
            x,
            y,
            hud_text,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontfamily="monospace",
            fontweight="bold",
            verticalalignment="top",
            horizontalalignment=ha,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=self.bg_color,
                alpha=alpha,
                edgecolor=border_color,
                linewidth=1.5,
            ),
            color=self.text_color,
            zorder=25,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=0.4, foreground="black")]
        )


# ── Helper: derive HUD state from info dict ─────────────────────────────

def derive_plan_status(
    *,
    plan_len: int = 0,
    plan_stale: bool = False,
    plan_reason: str = "none",
) -> str:
    """Derive PLAN badge from planner info dict fields.

    Enforces VC-2: if no plan, shows NO_PLAN or STALE — never blank.
    """
    if plan_stale:
        return PLAN_STATUS_STALE
    if plan_len > 1:
        return PLAN_STATUS_ACTIVE
    # plan_len <= 1 and not stale
    if plan_reason not in ("none", "initial", ""):
        return PLAN_STATUS_NO_PLAN
    # Initial state or single-cell plan
    return PLAN_STATUS_NO_PLAN if plan_len == 0 else PLAN_STATUS_ACTIVE


def derive_mission_status(
    *,
    terminated: bool = False,
    reached_goal: bool = False,
    at_goal: bool = False,
) -> str:
    """Derive STATUS badge from episode state."""
    if terminated and reached_goal:
        return MISSION_STATUS_COMPLETED
    if terminated and not reached_goal:
        return MISSION_STATUS_FAILED
    if at_goal:
        return MISSION_STATUS_SERVICING
    return MISSION_STATUS_EN_ROUTE
