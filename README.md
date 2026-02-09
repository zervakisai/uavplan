# UAVBench

A reproducible benchmark framework for UAV path planning in realistic urban environments. Uses real OpenStreetMap data rasterized to 2.5D grids with fire spread, traffic dynamics, and no-fly zone constraints.

## Features

- **Real-world maps** — 3 Athens tiles (Downtown, Penteli, Piraeus) from OpenStreetMap at 3m/pixel resolution
- **20 operational scenarios** — Wildfire, emergency response, port security, SAR, infrastructure patrol, border surveillance, crisis, comms-denied
- **Dynamic environments** — Cellular automaton fire spread with wind, pixel-space emergency vehicle traffic
- **Gymnasium-based** — Standard `reset()`/`step()` API, Discrete(6) action space, 7D observation
- **Operational metrics** — Safety (NFZ violations, risk exposure), efficiency (optimality, planning time), feasibility (smoothness, success rate)
- **Publication-ready visualization** — Trajectory plots, fire evolution panels, dynamics overlays, MP4 video export

## Quick Start

```bash
# Install
pip install -e ".[viz]"

# Run a benchmark
uavbench --scenarios osm_athens_wildfire_easy --planners astar --trials 5

# Visualize with dynamics
uavbench --scenarios osm_athens_emergency_easy --planners astar --trials 1 \
    --save-figures outputs/ --with-dynamics

# Full scenario pack (20 scenarios)
python tools/benchmark_scenario_pack.py --planners astar --trials 3 --output outputs/
```

## Scenario Pack

| Category | Scenarios | Tile | Dynamics |
|----------|-----------|------|----------|
| Wildfire WUI | easy / medium / hard | Penteli | Fire spread |
| Emergency Response | easy / medium / hard | Downtown | Traffic |
| Port Security | easy / medium / hard | Piraeus | Static |
| Combined Crisis | hard | Downtown | Fire + Traffic |
| Search & Rescue | easy / medium / hard | Penteli | Static |
| Infrastructure Patrol | easy / medium / hard | Downtown | Static |
| Border Surveillance | easy / medium / hard | Penteli | Static |
| Comms-Denied | hard | Downtown | Static |

Difficulty progression: altitude ceiling 10/8/6 levels, minimum L1 distance 150/200/250+.

## Project Structure

```
src/uavbench/
    envs/           Urban 2.5D grid environment (base + urban)
    scenarios/      Pydantic config schema, YAML loader, 20+ configs
    planners/       A* baseline (registry pattern for extensions)
    dynamics/       Fire spread model, traffic model
    metrics/        Safety, efficiency, feasibility metrics
    viz/            Matplotlib player, publication figures, dynamics sim
    cli/            Benchmark CLI entry point

tools/
    osm_pipeline/   Offline OSM fetch + rasterize scripts
    benchmark_scenario_pack.py
    generate_paper_figures.py

data/maps/          Pre-rasterized .npz tiles (gitignored)
```

## Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Safety | `nfz_violation_count` | Path cells on no-fly zones |
| Safety | `risk_exposure_sum` | Cumulative risk along path |
| Efficiency | `path_optimality` | Manhattan distance / path length |
| Efficiency | `planning_time_ms` | Wall-clock planning time |
| Feasibility | `path_smoothness` | 1 - direction changes / steps |
| Feasibility | `success` | Path found with no violations |

## Documentation

- [INSTALLATION.md](INSTALLATION.md) — Installation and tile generation
- [USAGE.md](USAGE.md) — CLI reference, custom scenarios, custom planners
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — Classes, methods, data formats
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md) — Benchmark results and performance data
- [scenarios README](src/uavbench/scenarios/configs/README_SCENARIOS.md) — Detailed scenario descriptions

## Requirements

- Python 3.10+
- Core: gymnasium, numpy, pydantic, pyyaml
- Visualization: matplotlib (`pip install -e ".[viz]"`)
- OSM pipeline: osmnx, geopandas, rasterio (`pip install -e ".[pipeline]"`)

## Citation

```bibtex
@software{uavbench2026,
  title     = {UAVBench: A Reproducible Benchmark for UAV Path Planning in Urban Environments},
  year      = {2026},
  url       = {https://github.com/uavbench/uavbench},
}
```

## License

MIT License. See [LICENSE](LICENSE).
