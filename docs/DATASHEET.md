# DATASHEET.md — UAVBench Dataset & Benchmark Datasheet

Following the template from Gebru et al. (2021), "Datasheets for Datasets."

---

## Motivation

**Purpose:** UAVBench evaluates path-planning algorithms for autonomous UAVs operating in dynamic urban environments with real-world operational constraints (no-fly zones, fire, traffic, comms degradation). It targets government/defense applications in Greece.

**Creators:** Konstantinos Zervakis and collaborators.

**Funding:** Academic research.

---

## Composition

**What does the dataset consist of?**

UAVBench is a *benchmark* (code + scenarios), not a static dataset. Each "instance" is a (scenario, planner, seed) triple that generates an episode trajectory.

| Component | Count | Description |
|---|---|---|
| Scenarios | 9 | 3 missions × 3 difficulties, grounded in real Greek incidents |
| Planners | 7 (+ 4 aliases) | A*, Theta*, periodic/aggressive replan, greedy local, MPPI, D* Lite |
| Seeds | User-defined (≥30 recommended) | Deterministic episode generation |
| Grid size | 500×500 | OSM-derived urban heightmaps |
| Action space | Discrete(6) | 4 cardinal + hover + diagonal |

**Incidents referenced:**

| Mission | Incident | Year |
|---|---|---|
| Civil Protection | Attica Wildfire (Penteli/Mati) | 2018 |
| Maritime Domain | Agia Zoni II Oil Spill (Saronic Gulf) | 2017 |
| Critical Infrastructure | Athens Metro Bomb Threat (Syntagma) | 2021 |

**Confidentiality:** All scenario data is derived from public sources (OSM tiles, news reports, NOTAMs). No classified or personal data.

---

## Collection Process

**How was data collected?**

- **Map tiles:** OpenStreetMap (OSM) extracts for Penteli, Piraeus, and Downtown Athens, rasterized to 500×500 grids with building footprints as obstacles.
- **Incident parameters:** Derived from publicly available reports (HCAA NOTAMs, REMPEC pollution reports, ELAS press releases).
- **Dynamics:** Procedurally generated (fire spread, traffic, NFZ expansion) using seeded RNGs.

**Who collected it?** The benchmark authors.

**Time period:** OSM tiles extracted 2024. Incident provenance: 2017–2021.

---

## Preprocessing

- OSM building footprints → binary obstacle mask (heightmap)
- No-fly zones → binary mask overlaid on obstacle grid
- Start/goal positions → fixed per scenario YAML

---

## Uses

**Intended uses:**
- Evaluating UAV path planners under dynamic constraints
- Comparing replanning strategies (periodic, reactive, incremental)
- Studying planner behavior under communications degradation
- Ablation studies on benchmark components (guardrail, interactions, risk fields)

**Not intended for:**
- Real-time flight control (this is a simulation benchmark)
- Training reinforcement learning agents (no reward shaping designed for RL)
- Operational mission planning (scenarios are simplified)

---

## Distribution

**License:** See `LICENSE` file in repository root.

**Repository:** https://github.com/zervakisai/uavbench

---

## Maintenance

**Who maintains?** Konstantinos Zervakis.

**Update policy:** Scenarios and planner registry may expand. Breaking changes will bump the major version.

**Contact:** Via GitHub issues.
