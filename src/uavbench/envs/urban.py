"""UrbanEnvV2 — urban navigation environment with dynamics.

Enforces: DC-1 (one RNG), EN-1..EN-9, MC-1..MC-4, MP-1.
Action space: Discrete(5) — UP(0), DOWN(1), LEFT(2), RIGHT(3), STAY(4).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scipy.ndimage import binary_dilation

from uavbench.blocking import SMOKE_BLOCKING_THRESHOLD, _CROSS_STRUCT, compute_blocking_mask
from uavbench.dynamics.collapse import CollapseModel
from uavbench.dynamics.fire_ca import FireSpreadModel
from uavbench.planners.astar import AStarPlanner
from uavbench.dynamics.interaction_engine import InteractionEngine
from uavbench.dynamics.restriction_zones import RestrictionZoneModel
from uavbench.dynamics.traffic import TrafficModel
from uavbench.dynamics.pathfinding import bfs_shortest_path
from uavbench.envs.base import RejectReason, TerminationReason
from uavbench.missions.engine import MissionEngine
from uavbench.scenarios.schema import ScenarioConfig

# Action constants
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4

# Movement deltas: (dx, dy) for each action
_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
    ACTION_STAY: (0, 0),
}


class UrbanEnvV2(gym.Env):
    """2D grid navigation environment (EN-1).

    Deterministic from seed via a single np.random.default_rng() call
    in reset() (DC-1).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: ScenarioConfig) -> None:
        super().__init__()
        self.config = config
        self._map_size = config.map_size
        self._max_steps = config.effective_max_steps

        # Action space: Discrete(5) (EN-2)
        self.action_space = spaces.Discrete(5)

        # Observation space: [ax, ay, gx, gy, terrain_h] (EN-3)
        high = np.array(
            [self._map_size, self._map_size, self._map_size, self._map_size, 10.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=high,
            dtype=np.float32,
        )

        # State (set during reset)
        self._heightmap: np.ndarray = np.zeros((0, 0))
        self._no_fly: np.ndarray = np.zeros((0, 0), dtype=bool)
        self._roads: np.ndarray = np.zeros((0, 0), dtype=bool)
        self._landuse_map: np.ndarray | None = None
        self._agent_xy: tuple[int, int] = (0, 0)
        self._goal_xy: tuple[int, int] = (0, 0)
        self._step_idx: int = 0
        self._events: list[dict] = []
        self._mission: MissionEngine | None = None
        self._terminated: bool = False
        self._truncated: bool = False
        self._termination_reason = TerminationReason.IN_PROGRESS
        self._objective_completed: bool = False

        # Dynamics (set during reset, None when disabled)
        self._fire: FireSpreadModel | None = None
        self._traffic: TrafficModel | None = None
        self._nfz: RestrictionZoneModel | None = None
        self._collapse: CollapseModel | None = None
        self._interaction: InteractionEngine | None = None
        self._bfs_corridor: list[tuple[int, int]] = []
        self._use_dynamics: bool = (
            config.enable_fire
            or config.enable_traffic
            or config.enable_dynamic_nfz
            or config.enable_collapse
        )

    # -- Public properties (EN-5) --

    @property
    def agent_xy(self) -> tuple[int, int]:
        return self._agent_xy

    @property
    def goal_xy(self) -> tuple[int, int]:
        return self._goal_xy

    @property
    def events(self) -> list[dict]:
        return self._events

    @property
    def bfs_corridor(self) -> list[tuple[int, int]]:
        """BFS reference corridor on static grid (FC-1)."""
        return list(self._bfs_corridor)

    # -- Gymnasium API --

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment deterministically from seed (DC-1).

        ALL randomness flows from ONE np.random.default_rng(seed) root.
        """
        super().reset(seed=seed)

        # DC-1: ONE RNG source — all children via spawn()
        root_rng = np.random.default_rng(seed)
        children = root_rng.spawn(6)
        env_rng, fire_rng, traffic_rng, nfz_rng, collapse_rng, _reserved_rng = children

        # Generate or load map
        if self.config.map_source == "osm" and self.config.osm_tile_id:
            self._heightmap, self._roads, landuse_map, nfz_mask = (
                self._load_osm_tile(self.config.osm_tile_id)
            )
            self._no_fly = nfz_mask
        else:
            self._heightmap = self._generate_heightmap(env_rng)
            self._no_fly = np.zeros(
                (self._map_size, self._map_size), dtype=bool
            )
            self._roads = self._generate_roads(env_rng)
            landuse_map = None
        self._landuse_map = landuse_map

        # Place agent and goal with min_start_goal_l1 enforcement
        if self.config.fixed_start_xy is not None:
            self._agent_xy = self.config.fixed_start_xy
        else:
            self._agent_xy = self._random_free_cell(env_rng)

        if self.config.fixed_goal_xy is not None:
            self._goal_xy = self.config.fixed_goal_xy
        else:
            self._goal_xy = self._random_free_cell(
                env_rng,
                exclude=self._agent_xy,
                min_l1_from=self._agent_xy,
                min_l1_dist=self.config.min_start_goal_l1,
            )

        # Ensure start and goal are free (EN-7)
        sx, sy = self._agent_xy
        gx, gy = self._goal_xy
        self._heightmap[sy, sx] = 0.0
        self._heightmap[gy, gx] = 0.0

        # Reset state
        self._step_idx = 0
        self._events = []
        self._terminated = False
        self._truncated = False
        self._termination_reason = TerminationReason.IN_PROGRESS
        self._objective_completed = False

        # Mission engine (MC-1, MC-2)
        self._mission = MissionEngine(
            mission_type=self.config.mission_type,
            start_xy=self._agent_xy,
            goal_xy=self._goal_xy,
            config=self.config,
        )

        # Ensure task POI is also a free cell
        poi = self._mission.objective_poi
        if poi != self._goal_xy and poi != self._agent_xy:
            px, py = poi
            if 0 <= px < self._map_size and 0 <= py < self._map_size:
                self._heightmap[py, px] = 0.0

        # Reference corridor — computed BEFORE dynamics so fire can
        # use it for corridor-aware ignition placement (Bug 4 fix).
        # Uses A* (same as planners) so corridor matches actual paths.
        # BFS fallback if A* fails (e.g., no path).
        _ref_planner = AStarPlanner(self._heightmap, self._no_fly)
        _ref_result = _ref_planner.plan(self._agent_xy, self._goal_xy)
        if _ref_result.success:
            self._bfs_corridor = _ref_result.path
        else:
            self._bfs_corridor = bfs_shortest_path(
                self._heightmap, self._agent_xy, self._goal_xy
            )

        # Scatter mission POIs to realistic off-corridor positions.
        # POIs are placed near corridor fractions but offset perpendicular
        # by ≥10 cells, forcing the agent to deviate from the corridor.
        # Dynamic obstacles (fire, traffic) on the corridor still affect
        # the journey, and each planner must independently route to POIs.
        if (self._mission is not None
                and self._mission.service_time_s > 0
                and len(self._bfs_corridor) > 2):
            self._mission.scatter_pois(
                self._bfs_corridor, self._heightmap, env_rng,
            )
            # Ensure all scattered task cells are free (walkable)
            for pos in self._mission.all_task_positions:
                px, py = pos
                if 0 <= px < self._map_size and 0 <= py < self._map_size:
                    self._heightmap[py, px] = 0.0

        # Initialize dynamics (Phase 4)
        map_shape = (self._map_size, self._map_size)
        self._fire = None
        self._traffic = None
        self._nfz = None
        self._collapse = None
        self._interaction = None

        if self.config.enable_fire and self.config.fire_ignition_points > 0:
            # Fire corridor guarantee: pick corridor interior targets
            guarantee_targets = _pick_corridor_targets(
                self._bfs_corridor,
                self.config.num_fire_corridor_closures,
            )
            self._fire = FireSpreadModel(
                map_shape=map_shape,
                rng=fire_rng,
                n_ignition=self.config.fire_ignition_points,
                landuse_map=landuse_map,
                roads_mask=self._roads,
                corridor_cells=self._bfs_corridor,
                guarantee_targets=guarantee_targets,
                guarantee_step=self.config.event_t1 or 30,
                wind_speed=self.config.wind_speed,
                wind_direction=math.radians(self.config.wind_direction_deg),
            )

        if self.config.enable_traffic and self.config.num_emergency_vehicles > 0:
            # Roadblock vehicles: find road cells closest to corridor
            roadblock_cells = _pick_roadblock_cells(
                self._bfs_corridor,
                self._roads,
                self.config.num_roadblock_vehicles,
            )
            self._traffic = TrafficModel(
                roads_mask=self._roads,
                num_vehicles=self.config.num_emergency_vehicles,
                rng=traffic_rng,
                corridor_cells=self._bfs_corridor,
                num_corridor_vehicles=self.config.num_corridor_vehicles,
                roadblock_cells=roadblock_cells,
                roadblock_step=self.config.event_t1 or 30,
            )

        if self.config.enable_dynamic_nfz and self.config.num_nfz_zones > 0:
            self._nfz = RestrictionZoneModel(
                map_shape=map_shape,
                rng=nfz_rng,
                num_zones=self.config.num_nfz_zones,
                event_t1=self.config.event_t1 or 30,
                event_t2=self.config.event_t2 or 80,
                corridor=self._bfs_corridor,
            )

        if self.config.enable_collapse and self._fire is not None:
            self._collapse = CollapseModel(
                heightmap=self._heightmap,
                rng=collapse_rng,
                collapse_delay=self.config.collapse_delay,
                debris_prob=self.config.debris_prob,
            )

        if self._fire is not None or self._traffic is not None:
            self._interaction = InteractionEngine(
                map_shape=map_shape,
                roads_mask=self._roads,
            )

        self._events.append({
            "type": "reset",
            "step_idx": 0,
            "agent_xy": self._agent_xy,
            "goal_xy": self._goal_xy,
        })

        # Mission briefing event (MC-3)
        if self._mission is not None:
            briefing = self._mission.generate_briefing()
            self._events.append(briefing.to_event())

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step (EN-1).

        Returns (obs, reward, terminated, truncated, info).
        """
        self._step_idx += 1

        # FD-3: Compute blocking mask on CURRENT (pre-advance) fire state.
        # Planner saw this same state; move validated against it.
        dyn_state = self.get_dynamic_state()
        blocked = compute_blocking_mask(
            self._heightmap, self._no_fly, self.config, dyn_state
        )

        # Propose move
        dx, dy = _DELTAS.get(action, (0, 0))
        nx, ny = self._agent_xy[0] + dx, self._agent_xy[1] + dy

        reward = -1.0  # step cost (EN-8)
        accepted = False
        reject_reason: RejectReason | None = None
        reject_cell: tuple[int, int] | None = None

        if action == ACTION_STAY:
            accepted = True
        elif not (0 <= nx < self._map_size and 0 <= ny < self._map_size):
            # Out of bounds
            reject_reason = RejectReason.OUT_OF_BOUNDS
            reject_cell = (nx, ny)
        elif blocked[ny, nx]:
            # Classify reject reason from layers (EC-1)
            reject_reason = self._classify_block(ny, nx, dyn_state)
            reject_cell = (nx, ny)

            # Terminal collision only for static obstacles (building/NFZ)
            if reject_reason in (RejectReason.BUILDING, RejectReason.NO_FLY):
                if self.config.terminate_on_collision:
                    self._terminated = True
                    if reject_reason == RejectReason.BUILDING:
                        self._termination_reason = TerminationReason.COLLISION_BUILDING
                    else:
                        self._termination_reason = TerminationReason.COLLISION_NFZ
                    reward += -25.0  # terminal penalty
        else:
            accepted = True
            self._agent_xy = (nx, ny)

        # FD-3: Advance dynamics AFTER move validation and execution.
        # Fire state was frozen during planner computation and move validation.
        self._step_dynamics()

        # Progress shaping (EN-8)
        if accepted and action != ACTION_STAY:
            dist_before = abs(
                (self._agent_xy[0] - dx) - self._goal_xy[0]
            ) + abs((self._agent_xy[1] - dy) - self._goal_xy[1])
            dist_after = abs(
                self._agent_xy[0] - self._goal_xy[0]
            ) + abs(self._agent_xy[1] - self._goal_xy[1])
            reward += 0.2 * (dist_before - dist_after)

        # Mission engine step (MC-2)
        if self._mission is not None:
            events_before = len(self._mission.events)
            self._mission.step(self._agent_xy, action, self._step_idx)
            # Append any new mission events to env events
            for evt in self._mission.events[events_before:]:
                self._events.append(evt)

        # Check goal reached
        if not self._terminated and self._agent_xy == self._goal_xy:
            self._terminated = True
            self._termination_reason = TerminationReason.SUCCESS
            self._objective_completed = (
                self._mission is not None and self._mission.all_tasks_completed
            )
            reward += 50.0  # goal bonus (EN-8)
            self._events.append({
                "type": "goal_reached",
                "step_idx": self._step_idx,
            })

        # Check timeout (EN-4)
        if not self._terminated and self._step_idx >= self._max_steps:
            self._truncated = True
            self._termination_reason = TerminationReason.TIMEOUT

        # Log move events (EC-1, EC-2)
        if reject_reason is not None:
            self._events.append({
                "type": "move_rejected",
                "step_idx": self._step_idx,
                "reject_reason": reject_reason,
                "reject_layer": reject_reason.value,
                "reject_cell": reject_cell,
            })

        # Build info
        info = self._get_info()
        info["accepted_move"] = accepted
        info["dynamics_step"] = self._step_idx
        if reject_reason is not None:
            info["reject_reason"] = reject_reason
            info["reject_layer"] = reject_reason.value
            info["reject_cell"] = reject_cell
            info["step_idx"] = self._step_idx

        obs = self._get_obs()
        return obs, reward, self._terminated, self._truncated, info

    # -- Export APIs --

    def export_planner_inputs(
        self,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
        """Return (heightmap, no_fly, start_xy, goal_xy) (EN-6)."""
        return (
            self._heightmap.copy(),
            self._no_fly.copy(),
            self._agent_xy,
            self._goal_xy,
        )

    def get_dynamic_state(self) -> dict[str, Any]:
        """Return dynamic layer state (EN-9).

        Returns actual dynamic masks when dynamics are enabled.
        """
        shape = (self._map_size, self._map_size)
        return {
            "fire_mask": (
                self._fire.fire_mask if self._fire is not None else None
            ),
            "smoke_mask": (
                self._fire.smoke_mask if self._fire is not None else None
            ),
            "traffic_positions": (
                self._traffic.vehicle_positions
                if self._traffic is not None
                else None
            ),
            "risk_cost_map": None,  # Phase 4+
            "traffic_closure_mask": (
                self._interaction.traffic_closure_mask
                if self._interaction is not None
                else None
            ),
            "traffic_occupancy_mask": (
                self._traffic.get_occupancy_mask(shape)
                if self._traffic is not None
                else None
            ),
            "dynamic_nfz_mask": (
                self._nfz.get_nfz_mask() if self._nfz is not None else None
            ),
            "debris_mask": (
                self._collapse.debris_mask
                if self._collapse is not None else None
            ),
        }

    # -- Internal --

    def _step_dynamics(self) -> None:
        """Advance all dynamic layers by one timestep."""
        # Fire advances first
        if self._fire is not None:
            self._fire.step()

        # All other dynamics see the NEW fire state (consistent)
        fire_mask = self._fire.fire_mask if self._fire is not None else None

        if self._traffic is not None:
            self._traffic.step(fire_mask=fire_mask, step_idx=self._step_idx)

        if self._nfz is not None:
            self._nfz.step(fire_mask=fire_mask)

        if self._collapse is not None:
            self._collapse.step(
                fire_mask=fire_mask, step_idx=self._step_idx,
            )

        if self._interaction is not None:
            self._interaction.update(
                fire_mask=self._fire.fire_mask if self._fire else None,
                traffic_positions=(
                    self._traffic.vehicle_positions
                    if self._traffic
                    else None
                ),
                nfz_mask=(
                    self._nfz.get_nfz_mask() if self._nfz else None
                ),
            )


    def _classify_block(
        self,
        ny: int,
        nx: int,
        dyn_state: dict[str, Any],
    ) -> RejectReason:
        """Classify which layer blocked cell (ny, nx) (EC-1).

        Priority: building > no_fly > fire > smoke > traffic_closure >
        traffic_buffer > dynamic_nfz.
        """
        if self._heightmap[ny, nx] > 0:
            return RejectReason.BUILDING
        if self._no_fly[ny, nx]:
            return RejectReason.NO_FLY
        if dyn_state.get("fire_mask") is not None and dyn_state["fire_mask"][ny, nx]:
            return RejectReason.FIRE
        # Fire buffer: cell not burning but within fire_buffer_radius (FD-2)
        # Reuses _CROSS_STRUCT from blocking module for consistency (MP-1)
        if self.config.fire_buffer_radius > 0 and dyn_state.get("fire_mask") is not None:
            fire = dyn_state["fire_mask"]
            if fire.any():
                buf = binary_dilation(fire, structure=_CROSS_STRUCT, iterations=self.config.fire_buffer_radius)
                if buf[ny, nx]:
                    return RejectReason.FIRE_BUFFER
        if dyn_state.get("smoke_mask") is not None and dyn_state["smoke_mask"][ny, nx] >= SMOKE_BLOCKING_THRESHOLD:
            return RejectReason.SMOKE
        if dyn_state.get("traffic_closure_mask") is not None and dyn_state["traffic_closure_mask"][ny, nx]:
            return RejectReason.TRAFFIC_CLOSURE
        if dyn_state.get("traffic_occupancy_mask") is not None and dyn_state["traffic_occupancy_mask"][ny, nx]:
            return RejectReason.TRAFFIC_BUFFER
        if dyn_state.get("dynamic_nfz_mask") is not None and dyn_state["dynamic_nfz_mask"][ny, nx]:
            return RejectReason.DYNAMIC_NFZ
        if dyn_state.get("debris_mask") is not None and dyn_state["debris_mask"][ny, nx]:
            return RejectReason.DEBRIS
        # Fallback (should not happen if blocking mask is consistent)
        return RejectReason.BUILDING

    def _generate_roads(self, rng: np.random.Generator) -> np.ndarray:
        """Generate synthetic road network as grid pattern."""
        roads = np.zeros((self._map_size, self._map_size), dtype=bool)
        # Place roads on a grid with spacing ~10% of map size
        spacing = max(3, self._map_size // 8)
        for i in range(0, self._map_size, spacing):
            roads[i, :] = True  # horizontal
            roads[:, i] = True  # vertical
        # Roads must be on free cells (not buildings)
        roads = roads & (self._heightmap == 0.0)
        return roads

    def _generate_heightmap(self, rng: np.random.Generator) -> np.ndarray:
        """Generate synthetic heightmap from RNG (DC-1 compliant)."""
        h = np.zeros((self._map_size, self._map_size), dtype=np.float32)
        if self.config.building_density > 0:
            mask = rng.random((self._map_size, self._map_size)) < self.config.building_density
            h[mask] = rng.uniform(1.0, 5.0, size=mask.sum()).astype(np.float32)
        return h

    def _load_osm_tile(
        self, tile_id: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load pre-rasterized OSM tile from data/maps/{tile_id}.npz.

        Returns (heightmap, roads_mask, landuse_map, nfz_mask).
        Heightmap values are real meters; blocking logic uses >0 so this
        works without conversion.  Static on disk — no RNG involved (DC-1).
        """
        tile_path = (
            Path(__file__).resolve().parents[3]  # src/uavbench/envs → repo root
            / "data" / "maps" / f"{tile_id}.npz"
        )
        if not tile_path.exists():
            raise FileNotFoundError(
                f"OSM tile not found: {tile_path}. "
                f"Available tiles: penteli, piraeus, downtown"
            )
        data = np.load(tile_path)
        heightmap = data["heightmap"].astype(np.float32)
        roads_mask = data["roads_mask"].astype(bool)
        landuse_map = data["landuse_map"]
        nfz_mask = data["nfz_mask"].astype(bool)
        return heightmap, roads_mask, landuse_map, nfz_mask

    def _random_free_cell(
        self,
        rng: np.random.Generator,
        exclude: tuple[int, int] | None = None,
        min_l1_from: tuple[int, int] | None = None,
        min_l1_dist: int = 0,
    ) -> tuple[int, int]:
        """Pick a random free cell on the heightmap.

        If min_l1_from and min_l1_dist are set, only cells with
        Manhattan distance >= min_l1_dist from min_l1_from are eligible.
        """
        free_mask = self._heightmap == 0.0
        if exclude is not None:
            free_mask[exclude[1], exclude[0]] = False
        if min_l1_from is not None and min_l1_dist > 0:
            fx, fy = min_l1_from
            ys, xs = np.mgrid[0:self._map_size, 0:self._map_size]
            l1 = np.abs(xs - fx) + np.abs(ys - fy)
            free_mask = free_mask & (l1 >= min_l1_dist)
        free_yx = np.argwhere(free_mask)
        if len(free_yx) == 0:
            # Fallback: relax distance constraint
            free_mask = self._heightmap == 0.0
            if exclude is not None:
                free_mask[exclude[1], exclude[0]] = False
            free_yx = np.argwhere(free_mask)
        idx = rng.integers(len(free_yx))
        y, x = free_yx[idx]
        return (int(x), int(y))

    def _get_obs(self) -> np.ndarray:
        """Build observation vector (EN-3)."""
        ax, ay = self._agent_xy
        gx, gy = self._goal_xy
        terrain_h = float(self._heightmap[ay, ax])
        return np.array([ax, ay, gx, gy, terrain_h], dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Build info dict with all required fields."""
        ax, ay = self._agent_xy

        info: dict[str, Any] = {
            # Position
            "agent_pos": self._agent_xy,
            "agent_xy": self._agent_xy,
            "goal_pos": self._goal_xy,
            "goal_xy": self._goal_xy,
            # Step
            "step_idx": self._step_idx,
            # Termination (MC-4)
            "termination_reason": self._termination_reason,
            "objective_completed": self._objective_completed,
        }

        # Mission fields (MC-1, MC-3)
        if self._mission is not None:
            info["objective_poi"] = self._mission.objective_poi
            info["objective_reason"] = self._mission.objective_reason
            info["mission_domain"] = self.config.mission_type.value
            info["objective_label"] = self._mission.objective_label
            info["distance_to_task"] = self._mission.distance_to_task(
                self._agent_xy
            )
            info["task_progress"] = self._mission.task_progress
            info["deliverable_name"] = self._mission.deliverable_name
            info["service_time_s"] = self._mission.service_time_s
            info["origin_name"] = self._mission.origin_name
            info["destination_name"] = self._mission.destination_name
            info["priority"] = self._mission.priority
            info["task_info_list"] = self._mission.task_info_list

        return info


# ---------------------------------------------------------------------------
# Helpers for physical interdiction placement
# ---------------------------------------------------------------------------


def _pick_corridor_targets(
    corridor: list[tuple[int, int]],
    n: int,
) -> list[tuple[int, int]]:
    """Pick n evenly-spaced interior cells along the corridor.

    Returns list of (x, y) corridor cells where fire must arrive.
    Skips first and last (start/goal).
    """
    if n <= 0 or len(corridor) <= 2:
        return []
    interior = corridor[1:-1]
    if n == 1:
        mid = len(interior) // 2
        return [interior[mid]]
    step = max(1, len(interior) // (n + 1))
    return [interior[min((i + 1) * step, len(interior) - 1)] for i in range(n)]


def _pick_roadblock_cells(
    corridor: list[tuple[int, int]],
    roads_mask: np.ndarray,
    n: int,
) -> list[tuple[int, int]]:
    """Find n road cells closest to corridor midpoints.

    Returns list of (y, x) road cells suitable for stationary vehicles.
    """
    if n <= 0 or len(corridor) <= 2:
        return []
    road_ys, road_xs = np.where(roads_mask)
    if len(road_ys) == 0:
        return []

    interior = corridor[1:-1]
    step = max(1, len(interior) // (n + 1))
    anchors = [interior[min((i + 1) * step, len(interior) - 1)] for i in range(n)]

    result = []
    for ax, ay in anchors:  # corridor cells are (x, y)
        dists = np.abs(road_ys - ay) + np.abs(road_xs - ax)
        closest_idx = int(np.argmin(dists))
        result.append((int(road_ys[closest_idx]), int(road_xs[closest_idx])))
    return result
