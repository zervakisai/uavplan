#!/usr/bin/env python3
"""Generate all publication-quality figures for the UAVBench paper.

Produces:
  outputs/figure_1_tile_comparison.png  — 3 Athens tiles side-by-side
  outputs/figure_2_fire_spread.png      — Fire evolution on Penteli
  outputs/figure_3_trajectory_downtown.png — A* path through downtown + traffic
  outputs/figure_4_trajectory_penteli.png  — A* path through Penteli + fire
  outputs/figure_5_event_distribution.png  — Event types from fire+traffic episode

Usage:
  python tools/generate_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uavbench.envs.urban import UrbanEnv
from uavbench.planners import PLANNERS
from uavbench.scenarios.loader import load_scenario
from uavbench.viz.dynamics_sim import simulate_dynamics_along_path
from uavbench.viz.figures import (
    plot_fire_evolution,
    plot_event_timeline,
    plot_tile_comparison,
    plot_trajectory_with_dynamics,
)

OUTPUT_DIR = Path("outputs")
CONFIGS_DIR = Path("src/uavbench/scenarios/configs")
TILES_DIR = Path("data/maps")


def _run_astar(scenario_yaml: str, seed: int = 42):
    """Load scenario, reset env, plan A* path, return (env, heightmap, nfz, start, goal, path)."""
    cfg = load_scenario(CONFIGS_DIR / scenario_yaml)
    env = UrbanEnv(cfg)
    env.reset(seed=seed)
    hm, nfz, start, goal = env.export_planner_inputs()
    planner = PLANNERS["astar"](hm, nfz)
    path = planner.plan(start, goal)
    return env, hm, nfz, start, goal, path


def figure_1_tile_comparison():
    """3 Athens tiles side-by-side with buildings + roads."""
    tiles = []
    for tile_id, label in [("downtown", "Downtown"), ("penteli", "Penteli"), ("piraeus", "Piraeus")]:
        npz = TILES_DIR / f"{tile_id}.npz"
        if not npz.exists():
            print(f"  Skipping {tile_id}: {npz} not found")
            continue
        d = np.load(str(npz))
        tiles.append({
            "label": label,
            "heightmap": d["heightmap"],
            "no_fly": d["nfz_mask"],
            "roads_mask": d["roads_mask"],
        })

    if not tiles:
        print("[SKIP] No tiles found for figure 1")
        return

    plot_tile_comparison(tiles, OUTPUT_DIR / "figure_1_tile_comparison.png",
                         title="Athens OSM tiles (500x500, 3m/pixel)")


def figure_2_fire_spread():
    """Fire evolution on Penteli over 200 timesteps."""
    npz = TILES_DIR / "penteli.npz"
    if not npz.exists():
        print("[SKIP] penteli.npz not found for figure 2")
        return

    d = np.load(str(npz))
    dummy_path = [(0, 0)] * 200
    sim = simulate_dynamics_along_path(
        npz, dummy_path,
        enable_fire=True,
        fire_ignition_points=3,
        wind_direction=0.0,
        wind_speed=0.6,
        seed=42,
    )

    plot_fire_evolution(
        d["heightmap"],
        sim["fire_states"],
        sim["burned_states"],
        OUTPUT_DIR / "figure_2_fire_spread.png",
        timestep_indices=[0, 25, 50, 75, 100, 150, 175, 199],
        ncols=4,
        title="Wildfire spread - Penteli foothills",
    )


def figure_3_trajectory_downtown():
    """A* path through downtown Athens with traffic overlay."""
    env, hm, nfz, start, goal, path = _run_astar("osm_athens_easy.yaml", seed=42)
    if not path:
        print("[SKIP] No path found for figure 3")
        return

    npz = TILES_DIR / "downtown.npz"
    sim = simulate_dynamics_along_path(
        npz, path,
        enable_traffic=True,
        num_vehicles=8,
        seed=42,
    )

    plot_trajectory_with_dynamics(
        hm, nfz, start, goal, path,
        OUTPUT_DIR / "figure_3_trajectory_downtown.png",
        vehicle_positions=sim["traffic_states"][-1],
        roads_mask=sim["roads_mask"],
        risk_map=sim["risk_map"],
        title=f"A* path - Downtown Athens ({len(path)} steps, 8 vehicles)",
    )


def figure_4_trajectory_penteli():
    """A* path through Penteli with fire overlay."""
    # Use the base osm_athens_easy but pointed at penteli
    npz = TILES_DIR / "penteli.npz"
    if not npz.exists():
        print("[SKIP] penteli.npz not found for figure 4")
        return

    # Load penteli tile directly via a custom scenario
    cfg = load_scenario(CONFIGS_DIR / "osm_athens_fire_easy.yaml")
    env = UrbanEnv(cfg)
    env.reset(seed=42)
    hm, nfz, start, goal = env.export_planner_inputs()
    planner = PLANNERS["astar"](hm, nfz)
    path = planner.plan(start, goal)

    if not path:
        print("[SKIP] No path found for figure 4")
        return

    sim = simulate_dynamics_along_path(
        npz, path,
        enable_fire=True,
        fire_ignition_points=3,
        wind_direction=0.0,
        wind_speed=0.6,
        seed=42,
    )

    plot_trajectory_with_dynamics(
        hm, nfz, start, goal, path,
        OUTPUT_DIR / "figure_4_trajectory_penteli.png",
        fire_mask=sim["fire_states"][-1],
        burned_mask=sim["burned_states"][-1],
        roads_mask=sim["roads_mask"],
        title=f"A* path - Penteli with wildfire ({len(path)} steps)",
    )


def figure_5_event_distribution():
    """Event distribution from a fire+traffic episode."""
    cfg = load_scenario(CONFIGS_DIR / "osm_athens_fire_easy.yaml")
    env = UrbanEnv(cfg)
    obs, info = env.reset(seed=42)

    # Run episode with random actions to generate events
    for _ in range(500):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break

    events = env.events
    if not events:
        print("[SKIP] No events generated for figure 5")
        return

    plot_event_timeline(
        events,
        OUTPUT_DIR / "figure_5_event_distribution.png",
        title="Event distribution - Penteli fire episode (500 steps)",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating paper figures in {OUTPUT_DIR}/\n")

    print("--- Figure 1: Tile comparison ---")
    figure_1_tile_comparison()

    print("\n--- Figure 2: Fire spread ---")
    figure_2_fire_spread()

    print("\n--- Figure 3: Trajectory downtown ---")
    figure_3_trajectory_downtown()

    print("\n--- Figure 4: Trajectory Penteli ---")
    figure_4_trajectory_penteli()

    print("\n--- Figure 5: Event distribution ---")
    figure_5_event_distribution()

    print(f"\nDone. All figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
