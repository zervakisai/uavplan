"""Paper-figure–style frame renderer (single source of truth for GIFs).

Every GIF-producing script in `scripts/` renders frames through this module
so they match the static paper figures (in particular
`gen_trajectory_comparison_5panel.py`) pixel-for-pixel in style:

    * basemap from `Renderer(paper_min, cell_px=1)` — same as the 5-panel figure
    * overlay order: smoke (α=40) → fire → traffic closures → vehicle icons
      → NFZ → debris
    * matplotlib panels (IEEE serif, small titles) with Okabe-Ito colors
    * markers: cyan "X" with white halo for BOTH start and live drone,
      yellow "P" for goal, green circle for success end, red "X" for failure
    * trajectory drawn with a subtle black stroke (path_effects)

The renderer is deterministic (VZ-3): same inputs → byte-identical frames.
Do NOT introduce additional module-level randomness (DC-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg

from flare.scenarios.schema import ScenarioConfig
from flare.visualization.labels import PLANNER_COLORS, PLANNER_SHORT
from flare.visualization.overlays import (
    draw_debris, draw_fire, draw_nfz, draw_smoke, draw_traffic,
)
from flare.visualization.renderer import Renderer


# IEEE-style matplotlib formatting (matches the static figures).
_RC_PAPER = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 100,
}


def apply_paper_rc() -> None:
    """Apply IEEE-style rcParams (idempotent)."""
    plt.rcParams.update(_RC_PAPER)


# ---------------------------------------------------------------------------
# Per-planner panel descriptor
# ---------------------------------------------------------------------------


@dataclass
class PanelSpec:
    """Description of one panel in a paper-figure frame.

    Attributes:
        planner_id: registry key (e.g. 'astar'). Used for title + trajectory color.
        trajectory: list of (x, y) cells visited so far. First element is the
            start marker; last element is the live drone position.
        goal_xy: (x, y) mission goal cell. Drawn as yellow plus.
        success: None = still running, True = reached goal, False = failed.
        termination_reason: short string shown in subtitle when not successful.
        coeff_label: text appended to planner short name (e.g. "α=5.0").
        title_override: if set, replaces the default "<short> (<coeff>)" title.
        subtitle_override: if set, replaces the default step/termination line.
    """
    planner_id: str
    trajectory: Sequence[tuple[int, int]]
    goal_xy: tuple[int, int]
    success: bool | None = None
    termination_reason: str | None = None
    coeff_label: str | None = None
    title_override: str | None = None
    subtitle_override: str | None = None
    zoom_center: tuple[int, int] | None = None  # cell coords for zoomed view
    zoom_radius: int | None = None               # half-width in cells


@dataclass
class Annotation:
    """Text annotation rendered on top of the figure (not on axes)."""
    text: str
    xy: tuple[float, float]          # figure coords (0..1)
    fontsize: float = 8.0
    color: str = "#111111"
    box: bool = True                 # draw a translucent box behind the text
    ha: str = "left"
    va: str = "top"


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------


class PaperFrameRenderer:
    """Render frames in the paper-figure style.

    The basemap (cream/grey/green/blue + building footprints) is rendered
    once per episode via `Renderer._render_basemap`, then cached. On every
    frame, dynamic overlays are drawn on a fresh copy of that cached basemap
    before handing it to matplotlib.

    Usage:
        r = PaperFrameRenderer(config)
        rgb = r.render_frame(
            heightmap, state, dyn_state,
            panels=[PanelSpec("aggressive_replan", traj, goal, success=None)],
            suptitle="Seed 42 · step 120",
        )
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self._inner = Renderer(config, mode="paper_min")
        self._inner._cell_px = 1
        self._cell = 1
        self._basemap: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        apply_paper_rc()

    # -- basemap ----------------------------------------------------------

    def _ensure_basemap(self, heightmap: np.ndarray, state: dict[str, Any]) -> None:
        if self._basemap is not None:
            return
        H, W = heightmap.shape
        self._basemap = self._inner._render_basemap(
            heightmap, H, W, self._cell,
            state.get("landuse_map"),
            state.get("roads_mask"),
        )
        self._shape = (H, W)

    def reset_basemap(self) -> None:
        """Drop the cached basemap (useful across different scenarios)."""
        self._basemap = None
        self._shape = None

    # -- dynamic overlays -------------------------------------------------

    @staticmethod
    def _draw_dyn_overlays(
        bg: np.ndarray,
        dyn_state: dict[str, Any] | None,
        cell: int,
    ) -> None:
        """Exact overlay sequence used by gen_trajectory_comparison_5panel.py."""
        if dyn_state is None:
            return
        smoke = dyn_state.get("smoke_mask")
        if smoke is not None:
            draw_smoke(bg, smoke, cell, alpha_256=40)
        fire = dyn_state.get("fire_mask")
        if fire is not None:
            draw_fire(bg, fire, cell)
        closures = dyn_state.get("traffic_closure_mask")
        if closures is not None:
            draw_traffic(bg, closures, cell)
        positions = dyn_state.get("traffic_positions")
        if positions is not None and len(positions) > 0:
            H_px, W_px = bg.shape[:2]
            r = max(5, cell * 3)
            r_out = r + 2
            for vy, vx in positions:
                cy, cx = int(vy * cell + cell // 2), int(vx * cell + cell // 2)
                for dy in range(-r_out, r_out + 1):
                    for dx in range(-r_out, r_out + 1):
                        d2 = dy * dy + dx * dx
                        py, px = cy + dy, cx + dx
                        if 0 <= py < H_px and 0 <= px < W_px:
                            if d2 <= r * r:
                                bg[py, px] = [0, 0, 0]
                            elif d2 <= r_out * r_out:
                                bg[py, px] = [200, 30, 30]
        nfz = dyn_state.get("nfz_mask") or dyn_state.get("dynamic_nfz_mask")
        if nfz is not None:
            draw_nfz(bg, nfz, cell)
        debris = dyn_state.get("debris_mask")
        if debris is not None:
            draw_debris(bg, debris, cell)

    # -- single-panel drawing --------------------------------------------

    @staticmethod
    def _draw_panel(
        ax,
        bg: np.ndarray,
        H: int,
        W: int,
        spec: PanelSpec,
    ) -> None:
        ax.imshow(
            bg, origin="upper", interpolation="nearest",
            aspect="equal", extent=[0, W, H, 0],
        )
        if spec.zoom_center is not None and spec.zoom_radius is not None:
            cx, cy = spec.zoom_center
            r = spec.zoom_radius
            ax.set_xlim(max(0, cx - r), min(W, cx + r))
            ax.set_ylim(min(H, cy + r), max(0, cy - r))
        else:
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)

        traj = spec.trajectory
        running = spec.success is None
        success = bool(spec.success) if not running else False

        # Start marker — cyan X with white halo (same as the static figure)
        if traj:
            sx, sy = traj[0]
            ax.plot(sx, sy, "x", color="white", markersize=6,
                    markeredgewidth=1.5, zorder=2)
            ax.plot(sx, sy, "x", color="#4090D0", markersize=5,
                    markeredgewidth=1.0, zorder=3)

        # Trajectory line in the planner's Okabe-Ito color
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ls = "-" if (running or success) else ":"
            color = PLANNER_COLORS.get(spec.planner_id, "#0072B2")
            ax.plot(xs, ys, ls, color=color, linewidth=1.2,
                    path_effects=[
                        pe.Stroke(linewidth=2.0, foreground="black", alpha=0.25),
                        pe.Normal(),
                    ], zorder=5)

        # Goal — yellow plus
        if spec.goal_xy is not None:
            gx, gy = spec.goal_xy
            ax.plot(gx, gy, "P", color="#E6C619", markersize=7,
                    markeredgecolor="k", markeredgewidth=0.4, zorder=10)

        # Head marker:
        #   running         → cyan X + white halo (same as start marker)
        #   finished + ok   → green circle
        #   finished + fail → red X
        if traj:
            ex, ey = traj[-1]
            if running:
                ax.plot(ex, ey, "x", color="white", markersize=7,
                        markeredgewidth=2.0, zorder=11)
                ax.plot(ex, ey, "x", color="#4090D0", markersize=6,
                        markeredgewidth=1.3, zorder=12)
            elif success:
                ax.plot(ex, ey, "o", color="#009E73", markersize=4,
                        markeredgecolor="white", markeredgewidth=0.5, zorder=11)
            else:
                ax.plot(ex, ey, "X", color="#CC3311", markersize=6,
                        markeredgecolor="white", markeredgewidth=0.4, zorder=11)

        # Titles (fixed 2-line layout — matches static figure)
        if spec.title_override is not None:
            title_line = spec.title_override
        else:
            short = PLANNER_SHORT.get(spec.planner_id, spec.planner_id)
            coeff = spec.coeff_label or ""
            title_line = f"{short} ({coeff})" if coeff else short

        if spec.subtitle_override is not None:
            sub_line = spec.subtitle_override
        else:
            steps_now = max(0, len(traj) - 1)
            if running:
                sub_line = f"t={steps_now}"
            elif success:
                sub_line = f"{steps_now} steps"
            else:
                term = spec.termination_reason or "failed"
                sub_line = f"{term}, t={steps_now}"
        ax.set_title(f"{title_line}\n{sub_line}",
                     fontsize=5.5, fontweight="normal", color="black")
        ax.axis("off")

    # -- public frame API -------------------------------------------------

    def render_frame(
        self,
        heightmap: np.ndarray,
        state: dict[str, Any],
        dyn_state: dict[str, Any] | None,
        panels: Sequence[PanelSpec],
        *,
        suptitle: str | None = None,
        panel_width_in: float = 2.4,
        panel_height_in: float = 2.2,
        annotations: Sequence[Annotation] | None = None,
    ) -> np.ndarray:
        """Render one RGB frame with N panels laid out horizontally."""
        self._ensure_basemap(heightmap, state)
        assert self._basemap is not None and self._shape is not None
        H, W = self._shape

        bg = self._basemap.copy()
        self._draw_dyn_overlays(bg, dyn_state, self._cell)

        n = max(1, len(panels))
        fig_w = panel_width_in * n + 0.1
        fig_h = panel_height_in + 0.4  # room for suptitle
        fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
        if n == 1:
            axes = [axes]
        fig.subplots_adjust(
            wspace=0.05, top=0.85, bottom=0.02, left=0.01, right=0.99,
        )

        for ax, spec in zip(axes, panels):
            self._draw_panel(ax, bg, H, W, spec)

        if suptitle:
            fig.suptitle(suptitle, fontsize=8, fontweight="normal",
                         color="black", y=0.98)

        if annotations:
            for a in annotations:
                bbox = dict(facecolor="white", edgecolor="none",
                            alpha=0.75, pad=2.0) if a.box else None
                fig.text(a.xy[0], a.xy[1], a.text,
                         fontsize=a.fontsize, color=a.color,
                         ha=a.ha, va=a.va, bbox=bbox,
                         transform=fig.transFigure, zorder=20)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        img = buf[:, :, :3].copy()
        plt.close(fig)
        return img


# ---------------------------------------------------------------------------
# Trajectory growth helper
# ---------------------------------------------------------------------------


def growing_trajectory(
    full_trajectory: Sequence[tuple[int, int]],
    step: int,
) -> list[tuple[int, int]]:
    """Return the slice of a trajectory up to step `step` (clamped)."""
    if not full_trajectory:
        return []
    end = max(1, min(step + 1, len(full_trajectory)))
    return list(full_trajectory[:end])


# ---------------------------------------------------------------------------
# Per-step episode capture
# ---------------------------------------------------------------------------


@dataclass
class EpisodeCapture:
    """Data captured from a `run_episode` via frame_callback.

    Fields:
        heightmap: episode heightmap (copied once)
        state0: snapshot of invariant state fields (start/goal/landuse/roads)
        trajectory: per-step agent positions
        dyn_snapshots: per-step copies of the dynamic masks (optional)
    """
    heightmap: np.ndarray | None = None
    state0: dict[str, Any] = field(default_factory=dict)
    trajectory: list[tuple[int, int]] = field(default_factory=list)
    dyn_snapshots: list[dict[str, Any]] = field(default_factory=list)


def _copy_arr(a):
    if a is None:
        return None
    return np.asarray(a).copy()


def _copy_pos(p):
    if p is None:
        return None
    if hasattr(p, "copy"):
        return p.copy()
    return list(p)


def make_capture_callback(
    capture: EpisodeCapture,
    *,
    record_dyn: bool = True,
):
    """Build a frame_callback that fills an EpisodeCapture.

    Fire CA is agent-independent (FD-4), so downstream scripts may capture
    dyn_state from ONE planner run and reuse it across other planners for
    the same (scenario, seed). Set `record_dyn=False` for the re-used runs.
    """
    def _cb(heightmap, state, dyn_state, cfg):
        ax, ay = state.get("agent_xy", (0, 0))
        capture.trajectory.append((int(ax), int(ay)))
        if capture.heightmap is None:
            capture.heightmap = heightmap.copy()
            capture.state0 = {
                "landuse_map": state.get("landuse_map"),
                "roads_mask": state.get("roads_mask"),
                "start_xy": tuple(state.get("start_xy", (ax, ay))),
                "goal_xy": tuple(state.get("goal_xy", (0, 0))),
            }
        if record_dyn:
            capture.dyn_snapshots.append({
                "step_idx": int(state.get("step_idx", len(capture.trajectory) - 1)),
                "fire_mask": _copy_arr(dyn_state.get("fire_mask")),
                "smoke_mask": _copy_arr(dyn_state.get("smoke_mask")),
                "debris_mask": _copy_arr(dyn_state.get("debris_mask")),
                "traffic_closure_mask": _copy_arr(dyn_state.get("traffic_closure_mask")),
                "traffic_positions": _copy_pos(dyn_state.get("traffic_positions")),
                "nfz_mask": _copy_arr(dyn_state.get("dynamic_nfz_mask")
                                      or dyn_state.get("nfz_mask")),
            })
    return _cb


def pick_dyn_snapshot(
    snapshots: Sequence[dict[str, Any]],
    step: int,
) -> dict[str, Any]:
    """Return the dyn snapshot at step `step`, clamped to the last available."""
    if not snapshots:
        return {}
    idx = min(max(0, step), len(snapshots) - 1)
    return snapshots[idx]
