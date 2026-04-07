#!/usr/bin/env python3
"""Wind-driven fire figure (Task 7).

Side-by-side comparison: isotropic fire (no wind) vs wind-driven fire
showing how wind turns symmetric danger into a deadly corridor.

Output: outputs/paper_figures/wind_driven_fire.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flare.benchmark.runner import run_episode
from flare.scenarios.loader import load_scenario

ROOT = Path(__file__).resolve().parent.parent

# IEEE formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
})

SCENARIO = "osm_penteli_pharma_delivery_medium"
SEED = 42
CAPTURE_STEP = 150


def _save(fig, name, out_dir="outputs/paper_figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir}/{name}.{{png,pdf}}")


def _make_config(base, **overrides):
    """Create a modified config from a frozen dataclass."""
    for key, val in overrides.items():
        object.__setattr__(base, key, val)
    return base


def _run_and_capture(scenario_id, seed, config_override, capture_step):
    """Run episode and capture fire/heightmap at a specific step."""
    captured = {}

    def cb(heightmap, state, dyn_state, cfg_arg):
        step = state.get("step_idx", 0)
        if step == capture_step and "done" not in captured:
            captured["heightmap"] = heightmap.copy()
            captured["fire"] = dyn_state.get(
                "fire_mask", np.zeros_like(heightmap, dtype=bool)
            ).copy()
            captured["smoke"] = dyn_state.get(
                "smoke_mask", np.zeros_like(heightmap, dtype=np.float32)
            ).copy()
            captured["start_xy"] = state.get("start_xy")
            captured["goal_xy"] = state.get("goal_xy")
            tp = dyn_state.get("traffic_positions")
            captured["traffic_positions"] = tp.copy() if tp is not None else None
            captured["agent_xy"] = state.get("agent_xy")
            captured["done"] = True

    run_episode(
        scenario_id, "aggressive_replan", seed,
        frame_callback=cb, config_override=config_override,
    )
    return captured


def _render_panel(ax, captured, title, wind_arrow=False):
    """Render a single map panel."""
    hm = captured["heightmap"]

    # Base image
    base = np.full((*hm.shape, 3), 240, dtype=np.uint8)
    base[hm > 0] = [80, 80, 80]

    # Fire
    fire = captured["fire"]
    if fire.any():
        base[fire] = [255, 60, 20]

    # Smoke
    smoke = captured["smoke"]
    smoke_mask = smoke >= 0.3
    if smoke_mask.any():
        base[smoke_mask] = (
            base[smoke_mask].astype(np.float64) * 0.6
            + np.array([160, 160, 160], dtype=np.float64) * 0.4
        ).astype(np.uint8)

    # Dynamic obstacles: black circles with red outline
    tp = captured.get("traffic_positions")
    if tp is not None and len(tp) > 0:
        H, W = base.shape[:2]
        r, r_out = 5, 7
        for vy, vx in tp:
            cy, cx = int(vy), int(vx)
            for dy in range(-r_out, r_out + 1):
                for dx in range(-r_out, r_out + 1):
                    d2 = dy * dy + dx * dx
                    py, px = cy + dy, cx + dx
                    if 0 <= py < H and 0 <= px < W:
                        if d2 <= r * r:
                            base[py, px] = [0, 0, 0]
                        elif d2 <= r_out * r_out:
                            base[py, px] = [200, 30, 30]

    ax.imshow(base, origin="upper")

    # Draw drone (blue X with white outline)
    agent = captured.get("agent_xy")
    if agent:
        # White outline
        ax.plot(agent[0], agent[1], "x", color="white",
                markersize=6, markeredgewidth=1.5, zorder=14)
        # Blue cross (top)
        ax.plot(agent[0], agent[1], "x", color="#4090D0",
                markersize=5, markeredgewidth=1.0, zorder=15)

    # Goal (yellow cross)
    start = captured.get("start_xy")
    goal = captured.get("goal_xy")
    if goal:
        ax.plot(goal[0], goal[1], "P", color="#E6C619", markersize=7,
                markeredgecolor="k", markeredgewidth=0.4, zorder=10)

    # Draw A* corridor as dashed line
    if start and goal:
        ax.plot([start[0], goal[0]], [start[1], goal[1]],
                ls="--", color="cyan", lw=0.8, alpha=0.5, zorder=5)

    # Wind arrow
    if wind_arrow and fire.any():
        fy, fx = np.where(fire)
        cx, cy = int(fx.mean()), int(fy.mean())
        # 270 deg = wind from west → blowing east
        ax.annotate(
            "", xy=(cx + 60, cy), xytext=(cx - 20, cy),
            arrowprops=dict(arrowstyle="->", color="white", lw=2.5),
            zorder=15,
        )
        ax.text(cx + 20, cy - 15, r"Wind $\rightarrow$", fontsize=7,
                color="black", fontweight="bold", zorder=15,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5))

    ax.set_title(title, fontsize=7, fontweight="normal", color="black")
    ax.axis("off")

    # Fire stats
    fire_count = int(fire.sum())
    ax.text(
        0.02, 0.02, f"Fire: {fire_count} cells",
        transform=ax.transAxes, fontsize=6, color="black", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5),
        va="bottom",
    )


def main() -> None:
    base_cfg = load_scenario(SCENARIO)

    # Config 1: Isotropic (wind_speed=0)
    cfg_iso = load_scenario(SCENARIO)
    _make_config(cfg_iso, wind_speed=0.0)

    # Config 2: Wind-driven (wind_speed=3.0, direction=270 deg = from west)
    cfg_wind = load_scenario(SCENARIO)
    _make_config(cfg_wind, wind_speed=3.0, wind_direction_deg=270.0)

    print(f"Running isotropic fire (wind=0)...")
    cap_iso = _run_and_capture(SCENARIO, SEED, cfg_iso, CAPTURE_STEP)

    print(f"Running wind-driven fire (wind=3.0, dir=270)...")
    cap_wind = _run_and_capture(SCENARIO, SEED, cfg_wind, CAPTURE_STEP)

    if "done" not in cap_iso or "done" not in cap_wind:
        print("ERROR: Failed to capture frames.")
        sys.exit(1)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    _render_panel(ax1, cap_iso, f"(a) Isotropic fire (no wind), t={CAPTURE_STEP}")
    _render_panel(ax2, cap_wind, f"(b) Wind-driven fire (3 m/s W\u2192E), t={CAPTURE_STEP}",
                  wind_arrow=True)

    # Label safe/dangerous routes on wind panel
    if cap_wind["fire"].any():
        fy, fx = np.where(cap_wind["fire"])
        cy = int(fy.mean())
        # Upwind = west side (safe), downwind = east side (dangerous)
        ax2.text(30, cy - 50, "Safe\n(upwind)", fontsize=6, color="black",
                 fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5))
        ax2.text(cap_wind["heightmap"].shape[1] - 30, cy + 50, "Dangerous\n(downwind)",
                 fontsize=6, color="black", fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5))

    fig.suptitle(
        "Wind turns symmetric danger into a deadly corridor",
        fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="black", alpha=0.8, edgecolor="white", linewidth=1.0)
    )
    fig.tight_layout()

    _save(fig, "wind_driven_fire")

    # Stats
    iso_count = int(cap_iso["fire"].sum())
    wind_count = int(cap_wind["fire"].sum())
    print(f"\nFire cells at t={CAPTURE_STEP}:")
    print(f"  Isotropic:   {iso_count}")
    print(f"  Wind-driven: {wind_count}")
    print(f"  Ratio:       {wind_count / max(iso_count, 1):.2f}x")


if __name__ == "__main__":
    main()
