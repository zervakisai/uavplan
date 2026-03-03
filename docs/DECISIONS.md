# UAVBench v2 — Design Decisions & Assumptions

This document records all design decisions made during Phase 0 (baseline extraction).
Each decision references the question it answers and the rationale.

---

## Q1: Horizon Policy

**Decision**: Env-enforced at `4 × map_size` steps.

**Rationale**: v1 uses `max_steps = 4 * map_size` (e.g., 2000 for 500x500 grids).
This is generous enough for any planner to reach the goal under moderate dynamics,
yet short enough to prevent infinite episodes. The runner does NOT override this
via CLI (`--max-episode-steps` was removed from paper protocol after causing confusion).

**Status**: DECIDED

---

## Q2: OSM Distribution

**Decision**: Deterministic fetch+rasterize pipeline. Tiles are pre-baked `.npz` files
under `data/maps/`. The pipeline is offline-only (`tools/osm_pipeline/`), never called
at runtime. ODbL compliance via attribution in paper acknowledgments.

**Tile inventory** (3 tiles):
| Tile ID    | Region           | Mission Type             | Building Density |
|------------|------------------|--------------------------|------------------|
| `penteli`  | Penteli, Attica  | civil_protection         | 0.18             |
| `piraeus`  | Piraeus port     | maritime_domain          | 0.29             |
| `downtown` | Athens center    | critical_infrastructure  | 0.50             |

**v2 change**: v2 will support BOTH `osm` and `synthetic` map sources, matching v1.
The synthetic path uses deterministic heightmap generation from seed.

**Status**: DECIDED

---

## Q3: Action Model

**Decision**: 2D grid, 4-connected movement (up/down/left/right).
Altitude (z) is tracked but NOT part of the action space for paper planners.
v1 uses `Discrete(6)` (4 cardinal + ascend/descend), but altitude actions are
effectively unused by all paper planners.

**v2 change**: v2 will use `Discrete(5)` — 4 cardinal directions + STAY.
The STAY action replaces the 2 altitude actions, which simplifies the action model
and supports service-time completion at POIs (agent stays at task location).

**Rationale**:
- All 6 paper planners produce 2D (x,y) paths; altitude is cosmetic
- STAY action is needed for mission service_time semantics (MC-2)
- Simpler action space → fewer untested code paths
- 2.5D extension (altitude as optional dimension) deferred to post-paper

**Status**: DECIDED — ASSUMPTION: altitude is cosmetic for paper. If paper
requires altitude metrics, revisit.

---

## Q4: Planner Suite

**Decision**: Exactly 6 planners, matching v1's `PAPER_PLANNERS` tuple.

| Registry Key           | Algorithm                | Replanning Strategy  |
|------------------------|--------------------------|----------------------|
| `astar`                | A* (4-connected)         | None (one-shot)      |
| `theta_star`           | Theta* (any-angle)       | None (one-shot)      |
| `periodic_replan`      | A* + periodic replan     | Every N steps        |
| `aggressive_replan`    | A* + aggressive replan   | Every N/2 steps      |
| `dstar_lite`           | D* Lite (Koenig 2002)    | Incremental          |
| `mppi_grid`            | MPPI (sampling-based)    | Every step           |

**v2 changes vs v1**:
1. `periodic_replan` and `aggressive_replan` use the PlannerBase interface
   (NOT the legacy AdaptiveAStarPlanner wrapper)
2. `dstar_lite` is a TRUE incremental D* Lite implementation
   (v1's "dstar_lite" was actually periodic A* replan — a misname)
3. `mppi_grid` uses env-seeded RNG (v1 used hardcoded `seed=42`)
4. All planners implement unified `PlannerBase` with `plan()`, `update()`,
   `should_replan()` methods
5. Deprecated aliases (`ad_star`, `dwa`, `mppi`) are NOT carried forward

**Status**: DECIDED

---

## Q5: v2 Independence

**Decision**: v2 is fully independent. No v1 schema compatibility required.
No import of v1 modules. No shared configuration files.

**Implications**:
- v2 has its own `ScenarioConfig` dataclass (may differ from v1)
- v2 has its own YAML configs under `src/uavbench/scenarios/configs/`
- v2 has its own CLI entry point (`python -m uavbench`)
- v2 test suite lives in `tests/v2/` (separate from `tests/`)
- v1 remains frozen and untouched in `src/uavbench/`

**Status**: DECIDED

---

## Q6: Coordinate Convention (derived from extraction)

**Decision**: All public APIs use `(x, y)` tuples. Internal array indexing uses
`array[y, x]`. This matches v1's convention but v2 will document it explicitly.

**Contract**:
- `agent_xy`, `goal_xy` → `(x, y)`
- `heightmap[y, x]`, `no_fly[y, x]`, `fire_mask[y, x]` → row-major
- `TrafficModel.positions` → `(y, x)` internally, converted at boundary
- Path entries: `list[tuple[int, int]]` in `(x, y)` order

**Status**: DECIDED

---

## Q7: Unified Blocking Mask (derived from failure mode analysis)

**Decision**: ONE `compute_blocking_mask(state) → np.ndarray[H,W] bool` function
used by BOTH step legality AND guardrail BFS. This is contract MP-1.

**Rationale**: v1 has a critical mask mismatch bug where step legality and guardrail
compute blocking differently (e.g., fire blocking is config-gated in step but
always-on in guardrail). v2 eliminates this by construction.

**Layers merged** (in order):
1. Static: `heightmap > 0` (buildings)
2. Static: `no_fly_mask` (static NFZ)
3. Dynamic: `forced_block_mask` (interdictions)
4. Dynamic: `traffic_closure_mask` (interaction engine)
5. Dynamic: `fire_mask` (if `config.fire_blocks_movement`)
6. Dynamic: traffic occupancy (if `config.traffic_blocks_movement`)
7. Dynamic: `moving_target_buffer` (if enabled)
8. Dynamic: `intruder_buffer` (if enabled)
9. Dynamic: `dynamic_nfz_mask` (if enabled)

**Status**: DECIDED

---

## Q8: Collision vs Rejection Semantics (derived from failure mode analysis)

**Decision**: v2 separates "hard collision" (terminates episode) from "soft rejection"
(move rejected, agent stays in place).

| Category | Layers | Effect |
|----------|--------|--------|
| Hard collision | buildings, static NFZ | Terminates episode (if `terminate_on_collision=True`) |
| Soft rejection | fire, traffic, forced blocks, intruders, target buffer, dynamic NFZ, traffic closures | Move rejected, agent stays, logged with `RejectReason` enum |

**RejectReason enum** (v2):
```
BUILDING, NO_FLY, FORCED_BLOCK, TRAFFIC_CLOSURE,
FIRE, TRAFFIC_BUFFER, MOVING_TARGET, INTRUDER, DYNAMIC_NFZ
```

Each rejection logs: `reject_reason`, `reject_layer`, `reject_cell`, `step_idx`.
This is contract EC-1.

**Status**: DECIDED

---

## Q9: Step Counter Ownership (derived from failure mode analysis)

**Decision**: The runner owns the authoritative `step_idx`. It passes it into
`env.step()`, dynamics, logger, and renderer. No component maintains its own counter.

**Rationale**: v1 has multiple step counter sources (`env._step_count`,
`info["step"]`, `engine.step_count`) that can diverge. v2 eliminates this by
making the runner the single source of truth.

**Contract**: EV-1 (every event contains authoritative `step_idx`).

**Status**: DECIDED

---

## Q10: RNG Threading (derived from failure mode analysis)

**Decision**: ONE `np.random.Generator` instance, seeded from
`np.random.default_rng(seed)` during `reset()`. All stochastic components
(fire, traffic, intruders, target, restriction zones, planner sampling)
receive child generators via `rng.spawn(N)` to ensure independence AND
reproducibility.

**Contract**: DC-1 (one RNG source, deterministic initialization).

**Status**: DECIDED

---

## Q11: Forced Interdiction Lifecycle (derived from failure mode analysis)

**Decision**: Each forced interdiction is tracked by ID with explicit lifecycle:
`PENDING → TRIGGERED → ACTIVE → CLEARED`. The `forced_block_cleared_by_guardrail`
flag is per-block, not global.

**Rationale**: v1 tracks a single global flag that never resets, causing stale
"cleared" reporting when new blocks are injected.

**Status**: DECIDED

---

## Q12: Replan Storm Prevention (derived from failure mode analysis)

**Decision**: Path-progress tracking prevents naive replanning. Contract RS-1
requires ≤20% of replans to be "naive" (same start/goal, no new obstacles).

**Mechanism**:
- Track `last_replan_pos`, `last_replan_mask_hash`
- If current position AND blocking mask are identical to last replan → naive
- Cooldown: minimum 3 steps between replans (unless forced)
- Budget: `max_replans_per_episode` from scenario config

**Status**: DECIDED

---

## Q13: Smoke Threshold (derived from failure mode analysis)

**Decision**: Smoke threshold is `0.3` with `>=` comparison (not `>`).
Applied consistently across ALL modules: step legality, guardrail, planner,
visualization.

**Rationale**: v1 has inconsistent `>` vs `>=` comparisons for smoke threshold,
causing planners to disagree on obstacle cells.

**Status**: DECIDED

---

## Q14: Plan Staleness (derived from failure mode analysis)

**Decision**: v2 tracks `plan_age_steps` (integer) instead of boolean `plan_stale`.
Stale threshold: if `plan_age_steps > 2 * replan_every_steps`, HUD shows `STALE`.
Cleared to 0 on every successful replan.

**Status**: DECIDED

---

## Q15: Mission Task Model

**Decision**: Multi-task mission model with:
- `TaskSpec` dataclass: `task_id`, `xy`, `weight`, `time_decay`, `time_window`,
  `service_time`, `category`, `injected_at`
- Task lifecycle: `PENDING → ACTIVE → COMPLETED | EXPIRED | SKIPPED`
- Mission policies: `greedy` (nearest-decay-aware) and `lookahead` (bounded orienteering)
- Mission utility: `U = Σ w_i · exp(-λ · elapsed_i)` for completed tasks

**v2 simplification**: Mission mode is an optional extension. The core benchmark
contract (DC/FC/EC/GC/EV/VC/MC/PC) applies to single-goal navigation episodes.
Multi-task episodes are a separate mode.

**Status**: DECIDED

---

## Q16: Scenario Configuration Carry-Forward

**Decision**: v2 carries forward all 9 government mission scenarios with identical
parameters. The ScenarioConfig dataclass is redesigned from scratch but maps 1:1
to v1 config values.

**Scenarios** (9 total):
- 3 easy (static track): `gov_{civil_protection,maritime_domain,critical_infrastructure}_easy`
- 6 stress-test (dynamic track): `gov_{civil_protection,maritime_domain,critical_infrastructure}_{medium,hard}`

**Status**: DECIDED

---

## Q17: Renderer Modes

**Decision**: Two renderer modes matching v1:
- `paper_min`: Minimal rendering for paper figures (clean, high-DPI)
- `ops_full`: Full operational rendering with all overlays and HUD

**Status**: DECIDED
