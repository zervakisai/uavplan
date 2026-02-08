"""Runtime dynamic models for UAVBench realistic scenarios.

This package provides numpy-only implementations of environmental
dynamics that evolve during simulation episodes:

- fire_spread: Cellular automaton fire propagation model
- traffic: Time-varying traffic density on road networks
- population: Population risk/density heatmaps for SAR prioritization

All models must:
- Use only numpy (no heavy geospatial deps at runtime)
- Accept an np.random.Generator for deterministic seeding
- Operate on the same grid coordinate system as UAVBench environments
"""
