# src/uavbench/dynamics/

Runtime dynamic models that evolve during UAVBench simulation episodes.

## Design Principles

1. **numpy-only**: No heavy geospatial dependencies at runtime
2. **Deterministic**: All models accept an `np.random.Generator` for reproducibility
3. **Grid-aligned**: Operate on the same [H, W] grid as UAVBench environments
4. **Composable**: Each model is independent; environments compose them as needed

## Planned Modules

### fire_spread.py
Cellular automaton fire propagation model. Fire starts at ignition points
and spreads based on fuel load, wind, and building adjacency.

### traffic.py
Time-varying traffic density model. Traffic patterns change over the
episode, affecting traversal costs and collision risk.

### population.py
Population risk/density heatmaps for search-and-rescue prioritization.
Higher population density areas should be searched first.

## Integration

Dynamic models are injected into environments via ScenarioConfig's
`extra` dict or future dedicated config fields. The environment's
`_step_impl()` advances each active model per timestep.
