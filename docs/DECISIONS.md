# UAVBench — Design Decisions & Assumptions

This document records all design decisions made during design.
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
| `penteli`  | Penteli, Attica  | fire_delivery / fire_surveillance | 0.18             |
| `piraeus`  | Piraeus port     | flood_rescue                      | 0.29             |
| `downtown` | Athens center    | (synthetic maps used instead)     | 0.50             |

**Design**: UAVBench supports BOTH `osm` and `synthetic` map sources, matching v1.
The synthetic path uses deterministic heightmap generation from seed.

**Status**: DECIDED

---

## Q3: Action Model

**Decision**: 2D grid, 4-connected movement (up/down/left/right).
Altitude (z) is tracked but NOT part of the action space for paper planners.
v1 uses `Discrete(6)` (4 cardinal + ascend/descend), but altitude actions are
effectively unused by all paper planners.

**v2 change**: UAVBench uses `Discrete(5)` — 4 cardinal directions + STAY.
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

**Decision**: Exactly 4 planners for the paper benchmark.

| Registry Key           | Algorithm                | Replanning Strategy  |
|------------------------|--------------------------|----------------------|
| `astar`                | A* (4-connected)         | None (one-shot)      |
| `periodic_replan`      | A* + periodic replan     | Every N steps        |
| `aggressive_replan`    | A* + mask-change replan  | On obstacle change   |
| `dstar_lite`           | D* Lite (simplified)     | On path blocked      |

**Removed from v1 planner set (with rationale):**
- `theta_star`: Removed due to BUG-1 (Theta* paradox — any-angle paths bypass
  single-cell interdictions). A* serves as the static baseline instead.
- `mppi_grid`: Removed due to BUG-3 (dead code — registered but never executed).

**Design notes:**
1. All planners implement unified `PlannerBase` with `plan()`, `update()`,
   `should_replan()` methods
2. `dstar_lite` uses A* internally with obstacle-change detection (simplified,
   not true incremental D* Lite — see PC-4)
3. Deprecated aliases (`ad_star`, `dwa`, `mppi`, `theta_star`) are NOT carried forward

**Status**: DECIDED

---

## Q5: Clean-Room Design

**Decision**: Clean-room implementation. No legacy schema compatibility required.

**Implications**:
- Own `ScenarioConfig` dataclass (clean-room design)
- Own YAML configs under `src/uavbench/scenarios/configs/`
- Own CLI entry point (`python -m uavbench`)
- Test suite in `tests/`

**Status**: DECIDED

---

## Q6: Coordinate Convention (derived from extraction)

**Decision**: All public APIs use `(x, y)` tuples. Internal array indexing uses
`array[y, x]`. This matches row-major convention, documented explicitly.

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
always-on in guardrail). UAVBench eliminates this by construction.

**Layers merged** (in order):
1. Static: `heightmap > 0` (buildings)
2. Static: `no_fly_mask` (static NFZ)
3. Dynamic: `forced_block_mask` (interdictions)
4. Dynamic: `traffic_closure_mask` (interaction engine)
5. Dynamic: `fire_mask` (if `config.fire_blocks_movement`)
6. Dynamic: `smoke_mask >= 0.5` (if `config.fire_blocks_movement`)
7. Dynamic: traffic occupancy (if `config.traffic_blocks_movement`)
8. Dynamic: `dynamic_nfz_mask` (if enabled)

**Status**: DECIDED

---

## Q8: Collision vs Rejection Semantics (derived from failure mode analysis)

**Decision**: UAVBench separates "hard collision" (terminates episode) from "soft rejection"
(move rejected, agent stays in place).

| Category | Layers | Effect |
|----------|--------|--------|
| Hard collision | buildings, static NFZ | Terminates episode (if `terminate_on_collision=True`) |
| Soft rejection | fire, traffic, forced blocks, dynamic NFZ, traffic closures | Move rejected, agent stays, logged with `RejectReason` enum |

**RejectReason enum**:
```
BUILDING, NO_FLY, FORCED_BLOCK, TRAFFIC_CLOSURE,
FIRE, TRAFFIC_BUFFER, DYNAMIC_NFZ, OUT_OF_BOUNDS
```

Each rejection logs: `reject_reason`, `reject_layer`, `reject_cell`, `step_idx`.
This is contract EC-1.

**Status**: DECIDED

---

## Q9: Step Counter Ownership (derived from failure mode analysis)

**Decision**: The runner owns the authoritative `step_idx`. It passes it into
`env.step()`, dynamics, logger, and renderer. No component maintains its own counter.

**Rationale**: v1 has multiple step counter sources (`env._step_count`,
`info["step"]`, `engine.step_count`) that can diverge. UAVBench eliminates this by
making the runner the single source of truth.

**Contract**: EV-1 (every event contains authoritative `step_idx`).

**Status**: DECIDED

---

## Q10: RNG Threading (derived from failure mode analysis)

**Decision**: ONE `np.random.Generator` instance, seeded from
`np.random.default_rng(seed)` during `reset()`. All stochastic components
(fire, traffic, restriction zones, forced blocks)
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

**Decision**: Smoke threshold is `0.5` with `>=` comparison (not `>`).
Applied consistently across ALL modules via `compute_blocking_mask()` (MP-1).

**Rationale**: v1 had inconsistent `>` vs `>=` comparisons for smoke threshold,
causing planners to disagree on obstacle cells. Fixed by using ONE blocking
mask function everywhere.

**Status**: DECIDED

---

## Q14: Plan Staleness (derived from failure mode analysis)

**Decision**: UAVBench tracks `plan_age_steps` (integer) instead of boolean `plan_stale`.
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

**Simplification**: Mission mode is an optional extension. The core benchmark
contract (DC/FC/EC/GC/EV/VC/MC/PC) applies to single-goal navigation episodes.
Multi-task episodes are a separate mode.

**Status**: DECIDED

---

## Q16: Scenario Configuration Carry-Forward

**Decision**: UAVBench includes all 9 government mission scenarios with identical
parameters. The ScenarioConfig dataclass is redesigned from scratch but maps 1:1
to v1 config values.

**Scenarios** (9 total):
- 3 easy (static track): `gov_{fire_delivery,fire_surveillance,flood_rescue}_easy`
- 6 stress-test (dynamic track): `gov_{fire_delivery,fire_surveillance,flood_rescue}_{medium,hard}`

**Status**: DECIDED

---

## Q17: Renderer Modes

**Decision**: Two renderer modes matching v1:
- `paper_min`: Minimal rendering for paper figures (clean, high-DPI)
- `ops_full`: Full operational rendering with all overlays and HUD

**Status**: DECIDED
