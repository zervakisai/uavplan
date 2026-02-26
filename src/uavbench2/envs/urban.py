"""UrbanEnvV2 — urban navigation environment with dynamics.

Enforces: DC-1 (one RNG), EN-1..EN-9, MC-1..MC-4, MP-1.
Action space: Discrete(5) — UP(0), DOWN(1), LEFT(2), RIGHT(3), STAY(4).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uavbench2.blocking import compute_blocking_mask
from uavbench2.dynamics.fire_ca import FireSpreadModel
from uavbench2.dynamics.interaction_engine import InteractionEngine
from uavbench2.dynamics.restriction_zones import RestrictionZoneModel
from uavbench2.dynamics.traffic import TrafficModel
from uavbench2.envs.base import RejectReason, TerminationReason
from uavbench2.missions.engine import MissionEngine
from uavbench2.scenarios.schema import ScenarioConfig

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
        self._interaction: InteractionEngine | None = None
        self._use_dynamics: bool = (
            config.enable_fire
            or config.enable_traffic
            or config.enable_dynamic_nfz
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
        children = root_rng.spawn(5)
        env_rng, fire_rng, traffic_rng, nfz_rng, _reserved_rng = children

        # Generate heightmap (synthetic)
        self._heightmap = self._generate_heightmap(env_rng)
        self._no_fly = np.zeros(
            (self._map_size, self._map_size), dtype=bool
        )
        self._roads = self._generate_roads(env_rng)

        # Place agent and goal
        if self.config.fixed_start_xy is not None:
            self._agent_xy = self.config.fixed_start_xy
        else:
            self._agent_xy = self._random_free_cell(env_rng)

        if self.config.fixed_goal_xy is not None:
            self._goal_xy = self.config.fixed_goal_xy
        else:
            self._goal_xy = self._random_free_cell(env_rng, exclude=self._agent_xy)

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
        )

        # Ensure task POI is also a free cell
        poi = self._mission.objective_poi
        if poi != self._goal_xy and poi != self._agent_xy:
            px, py = poi
            if 0 <= px < self._map_size and 0 <= py < self._map_size:
                self._heightmap[py, px] = 0.0

        # Initialize dynamics (Phase 4)
        map_shape = (self._map_size, self._map_size)
        self._fire = None
        self._traffic = None
        self._nfz = None
        self._interaction = None

        if self.config.enable_fire and self.config.fire_ignition_points > 0:
            self._fire = FireSpreadModel(
                map_shape=map_shape,
                rng=fire_rng,
                n_ignition=self.config.fire_ignition_points,
                wind_speed=self.config.wind_speed,
            )

        if self.config.enable_traffic and self.config.num_emergency_vehicles > 0:
            self._traffic = TrafficModel(
                roads_mask=self._roads,
                num_vehicles=self.config.num_emergency_vehicles,
                rng=traffic_rng,
            )

        if self.config.enable_dynamic_nfz and self.config.num_nfz_zones > 0:
            self._nfz = RestrictionZoneModel(
                map_shape=map_shape,
                rng=nfz_rng,
                num_zones=self.config.num_nfz_zones,
                event_t1=self.config.event_t1 or 30,
                event_t2=self.config.event_t2 or 80,
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

        # Step dynamics BEFORE computing blocking mask
        self._step_dynamics()

        # Compute blocking mask with dynamic state (MP-1)
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
            "forced_block_mask": None,  # Phase 5
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
            "moving_target_buffer": None,  # Phase 4+
            "intruder_buffer": None,  # Phase 4+
            "dynamic_nfz_mask": (
                self._nfz.get_nfz_mask() if self._nfz is not None else None
            ),
        }

    # -- Internal --

    def _step_dynamics(self) -> None:
        """Advance all dynamic layers by one timestep."""
        fire_mask = self._fire.fire_mask if self._fire is not None else None

        if self._fire is not None:
            self._fire.step()

        if self._traffic is not None:
            self._traffic.step(fire_mask=fire_mask)

        if self._nfz is not None:
            self._nfz.step(fire_mask=fire_mask)

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
        traffic_buffer > dynamic_nfz > forced_block.
        """
        if self._heightmap[ny, nx] > 0:
            return RejectReason.BUILDING
        if self._no_fly[ny, nx]:
            return RejectReason.NO_FLY
        if dyn_state.get("fire_mask") is not None and dyn_state["fire_mask"][ny, nx]:
            return RejectReason.FIRE
        if dyn_state.get("smoke_mask") is not None and dyn_state["smoke_mask"][ny, nx] >= 0.3:
            return RejectReason.FIRE
        if dyn_state.get("traffic_closure_mask") is not None and dyn_state["traffic_closure_mask"][ny, nx]:
            return RejectReason.TRAFFIC_CLOSURE
        if dyn_state.get("traffic_occupancy_mask") is not None and dyn_state["traffic_occupancy_mask"][ny, nx]:
            return RejectReason.TRAFFIC_BUFFER
        if dyn_state.get("dynamic_nfz_mask") is not None and dyn_state["dynamic_nfz_mask"][ny, nx]:
            return RejectReason.DYNAMIC_NFZ
        if dyn_state.get("forced_block_mask") is not None and dyn_state["forced_block_mask"][ny, nx]:
            return RejectReason.FORCED_BLOCK
        if dyn_state.get("moving_target_buffer") is not None and dyn_state["moving_target_buffer"][ny, nx]:
            return RejectReason.MOVING_TARGET
        if dyn_state.get("intruder_buffer") is not None and dyn_state["intruder_buffer"][ny, nx]:
            return RejectReason.INTRUDER
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

    def _random_free_cell(
        self,
        rng: np.random.Generator,
        exclude: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Pick a random free cell on the heightmap."""
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

        return info
