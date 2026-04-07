"""Generate improved 3-panel scenario overview figure for the FLARE paper.

Shows Penteli, Piraeus, Downtown side-by-side with:
- Buildings (dark grey), roads (light grey)
- Start (green) and goal (gold) markers, size 100+
- Building density label in each panel corner
- A* reference path overlay (thin dashed orange)
- Okabe-Ito colorblind-safe palette
- 300 DPI, figure* width (~7 inches) for IEEE two-column
"""

import sys
sys.path.insert(0, "/Users/konstantinos/Dev/planning/uavbench/src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from uavbench.scenarios.loader import load_scenario
from uavbench.envs.urban import UrbanEnvV2
from uavbench.planners.astar import AStarPlanner

# --- Okabe-Ito colorblind-safe palette ---
OI_GREEN  = "#009E73"   # start marker
OI_GOLD   = "#E69F00"   # goal marker
OI_ORANGE = "#D55E00"   # A* path
OI_BLUE   = "#0072B2"   # unused but available

# Building / road / free colors
COLOR_BUILDING = "#4A4A4A"  # dark grey
COLOR_ROAD     = "#D0D0D0"  # light grey
COLOR_FREE     = "#F0F0F0"  # very light grey for open cells

# Scenario definitions: (scenario_id, display_name, density, mission_label)
SCENARIOS = [
    ("osm_penteli_pharma_delivery_medium",  "Penteli",   0.18, "Pharma Delivery"),
    ("osm_piraeus_urban_rescue_medium",     "Piraeus",   0.29, "Urban Rescue"),
    ("osm_downtown_fire_surveillance_medium","Downtown",  0.50, "Fire Surveillance"),
]

SEED = 42
OUT_PATH = "/Users/konstantinos/Dev/planning/uavbench/paper/figures/scenario_overview_3panel.pdf"


def render_scenario_panel(ax, scenario_id, display_name, density, mission_label):
    """Render one scenario panel on the given axes."""
    # Load scenario and create environment
    cfg = load_scenario(scenario_id)
    env = UrbanEnvV2(cfg)
    obs, info = env.reset(seed=SEED)
    heightmap, no_fly, start_xy, goal_xy = env.export_planner_inputs()

    map_size = heightmap.shape[0]

    # Build an RGB image for the map
    # [y, x] indexing for the arrays
    rgb = np.full((map_size, map_size, 3), 0.94, dtype=np.float32)  # off-white default

    # Roads: light grey — check if roads mask available
    roads = env._roads
    rgb[roads] = np.array([0.816, 0.816, 0.816])  # #D0D0D0

    # Buildings: dark grey — wherever heightmap > 0
    buildings = heightmap > 0
    rgb[buildings] = np.array([0.29, 0.29, 0.29])  # #4A4A4A

    # No-fly zones: slightly tinted (optional, subtle)
    if no_fly.any():
        nfz_only = no_fly & ~buildings
        rgb[nfz_only] = np.array([0.5, 0.5, 0.6])

    # Display the map image
    # imshow uses [y, x] naturally — origin='upper' means row 0 at top
    ax.imshow(rgb, origin="upper", interpolation="nearest", aspect="equal")

    # Compute A* reference path
    planner = AStarPlanner(heightmap, no_fly)
    result = planner.plan(start_xy, goal_xy)

    if result.success and len(result.path) > 1:
        # Path is list of (x, y) tuples — plot as scatter/line
        path_xs = [p[0] for p in result.path]
        path_ys = [p[1] for p in result.path]
        # White halo for contrast against dark buildings
        ax.plot(path_xs, path_ys, color="white", linewidth=2.2,
                linestyle="-", alpha=0.6, zorder=3)
        ax.plot(path_xs, path_ys, color=OI_ORANGE, linewidth=1.2,
                linestyle="--", dashes=(5, 3), alpha=0.9, zorder=4,
                label="A* path")

    # Start marker (green circle)
    sx, sy = start_xy
    ax.scatter([sx], [sy], c=OI_GREEN, s=130, marker="o",
               edgecolors="black", linewidths=0.6, zorder=6, label="Start")

    # Goal marker (gold star)
    gx, gy = goal_xy
    ax.scatter([gx], [gy], c=OI_GOLD, s=180, marker="*",
               edgecolors="black", linewidths=0.4, zorder=6, label="Goal")

    # Building density label in upper-left corner
    ax.text(0.04, 0.96, f"$\\rho = {density:.2f}$",
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.85),
            zorder=10)

    # Panel title: "ScenarioName -- MissionType" (en-dash for IEEE style)
    ax.set_title(f"{display_name} \u2013 {mission_label}", fontsize=10, fontweight="bold",
                 pad=6)

    # Clean up axes
    ax.set_xlim(0, map_size)
    ax.set_ylim(map_size, 0)  # y-axis: 0 at top
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#888888")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), dpi=300)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06,
                        wspace=0.08)

    for ax, (sid, name, density, mission) in zip(axes, SCENARIOS):
        render_scenario_panel(ax, sid, name, density, mission)

    # Shared legend below the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=8, frameon=True, fancybox=False,
               edgecolor="#888888", borderpad=0.4,
               bbox_to_anchor=(0.5, -0.02))

    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
