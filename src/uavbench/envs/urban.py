from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uavbench.dynamics.adversarial_uav import AdversarialUAVModel
from uavbench.envs.base import UAVBenchEnv
from uavbench.dynamics.fire_spread import FireSpreadModel
from uavbench.dynamics.interaction_engine import InteractionEngine
from uavbench.dynamics.population_risk import PopulationRiskModel
from uavbench.dynamics.traffic import TrafficModel
from uavbench.planners.astar import AStarPlanner
from uavbench.planners.theta_star import ThetaStarPlanner
from uavbench.scenarios.schema import ScenarioConfig, Domain, Difficulty

GridPos = tuple[int, int]


class UrbanEnv(UAVBenchEnv):
    """Urban 2.5D SAR environment for UAVBench.

    - 2D grid (x, y) με διακριτό ύψος z
    - Buildings ως heightmap h(y, x)
    - Collision αν z <= h(y, x) ή αν είμαστε σε no-fly κελί
    - Easy/Medium/Hard μοιράζονται την ίδια πόλη, αλλάζουν density/blocks/wind
    """

    def __init__(self, config: ScenarioConfig):
        assert config.domain == Domain.URBAN
        super().__init__(config)

        self.map_size = config.map_size
        self.max_altitude = config.max_altitude
        self.max_building_height = 50.0

        # State: [x, y, z, gx, gy, gz, h(x,y)]
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, 0, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.map_size - 1,
                    self.map_size - 1,
                    self.max_altitude,
                    self.map_size - 1,
                    self.map_size - 1,
                    self.max_altitude,
                    self.max_building_height,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Actions: 0 up, 1 down, 2 left, 3 right, 4/5 change altitude
        self.action_space = spaces.Discrete(6)

        # Internal state
        self._agent_pos: np.ndarray  # [x, y, z]
        self._goal_pos: np.ndarray   # [x, y, z]
        self._heightmap: np.ndarray  # [H, W]
        self._no_fly_mask: np.ndarray  # [H, W], bool

    # --------------- Domain-specific reset/step ----------------

    def _reset_impl(self, options: dict[str, Any] | None = None):
        """Build or load a map + sample start/goal for a new episode."""
        options = options or {}
        cfg = self.config

        # ---------- Build / load the map ----------
        if cfg.map_source == "osm":
            if cfg.osm_tile_id is None:
                raise ValueError("map_source='osm' requires osm_tile_id")
            self._load_osm_tile(cfg.osm_tile_id)
        else:
            self._generate_synthetic_map(options)

        H, W = self._heightmap.shape

        # Runtime paper-protocol layers
        self._traffic_closure_mask = np.zeros((H, W), dtype=bool)
        self._forced_block_mask = np.zeros((H, W), dtype=bool)
        self._emergency_corridor_mask = np.zeros((H, W), dtype=bool)
        self._risk_cost_map = np.zeros((H, W), dtype=np.float32)
        self._forced_replans_triggered = 0
        self._forced_interdictions: list[dict[str, Any]] = []
        self._interdiction_hits = 0
        self._interdiction_total = 0
        self._reference_corridor_mask = np.zeros((H, W), dtype=bool)
        self._reference_path: list[GridPos] = []
        self._interaction_metrics_last: dict[str, float] = {}
        self._interaction_metrics_history: list[dict[str, float]] = []
        self._last_guardrail_status: dict[str, Any] = {
            "reachability_failed_before_relax": False,
            "relaxation_applied": {},
            "corridor_fallback_used": False,
            "feasible_after_guardrail": True,
        }
        self._guardrail_unreachable_streak = 0
        self._emergency_corridor_active = False

        # ---------- Initialize dynamic layers ----------
        self._fire_model: FireSpreadModel | None = None
        self._traffic_model: TrafficModel | None = None
        self._moving_target = None
        self._intruder_model = None
        self._dynamic_nfz = None

        if cfg.enable_fire:
            landuse = getattr(self, "_landuse_map", np.zeros((H, W), dtype=np.int8))
            roads = getattr(self, "_roads_mask", np.zeros((H, W), dtype=bool))
            self._fire_model = FireSpreadModel(
                landuse_map=landuse,
                roads_mask=roads,
                wind_dir=cfg.wind_direction,
                wind_speed=cfg.wind_speed,
                rng=self._rng,
                n_ignition=cfg.fire_ignition_points,
            )

        if cfg.enable_traffic:
            roads = getattr(self, "_roads_mask", np.zeros((H, W), dtype=bool))
            self._traffic_model = TrafficModel(
                roads_mask=roads,
                num_vehicles=cfg.num_emergency_vehicles,
                rng=self._rng,
            )

        # ---------- Parameters for start/goal placement ----------
        safe_alt = int(getattr(cfg, "safe_altitude", options.get("safe_altitude", self.max_altitude)))
        safe_alt = int(np.clip(safe_alt, 0, self.max_altitude))

        min_l1 = int(getattr(cfg, "min_start_goal_l1", options.get("min_start_goal_l1", max(2, H // 2))))

        # ---------- Collect free cells ----------
        free_mask = (self._heightmap == 0.0) & (~self._no_fly_mask)
        free_cells = np.argwhere(free_mask)  # array of [y, x]

        if free_cells.shape[0] < 2:
            raise ValueError(
                "Not enough free cells to place start/goal. "
                "Reduce building_density / no-fly constraints or increase map_size."
            )

        def _nearest_free(x: int, y: int, *, endpoint: str) -> tuple[int, int]:
            """Map fixed endpoints to nearest free cell when exact cell is blocked."""
            if 0 <= x < W and 0 <= y < H and bool(free_mask[y, x]):
                return x, y
            dists = np.abs(free_cells[:, 1] - x) + np.abs(free_cells[:, 0] - y)
            idx = int(np.argmin(dists))
            ny, nx = map(int, free_cells[idx])
            self.log_event(
                "fixed_endpoint_relaxed",
                endpoint=endpoint,
                requested=(x, y),
                used=(nx, ny),
            )
            return nx, ny

        from collections import deque

        largest_component_cells: np.ndarray | None = None

        def _component_mask(seed_xy: tuple[int, int]) -> np.ndarray:
            sx0, sy0 = seed_xy
            comp = np.zeros_like(free_mask, dtype=bool)
            if not (0 <= sx0 < W and 0 <= sy0 < H):
                return comp
            if not free_mask[sy0, sx0]:
                return comp
            q = deque([(sx0, sy0)])
            comp[sy0, sx0] = True
            while q:
                x, y = q.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H:
                        if free_mask[ny, nx] and not comp[ny, nx]:
                            comp[ny, nx] = True
                            q.append((nx, ny))
            return comp

        def _largest_component() -> np.ndarray:
            nonlocal largest_component_cells
            if largest_component_cells is not None:
                return largest_component_cells
            visited = np.zeros_like(free_mask, dtype=bool)
            best_cells = free_cells[:1]
            best_size = 0
            for yx in free_cells:
                sy0, sx0 = map(int, yx)
                if visited[sy0, sx0]:
                    continue
                q = deque([(sx0, sy0)])
                visited[sy0, sx0] = True
                cells: list[tuple[int, int]] = [(sy0, sx0)]
                while q:
                    x, y = q.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            if free_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                q.append((nx, ny))
                                cells.append((ny, nx))
                if len(cells) > best_size:
                    best_size = len(cells)
                    best_cells = np.array(cells, dtype=np.int32)
            largest_component_cells = best_cells
            return largest_component_cells

        # ---------- Sample start ----------
        if cfg.fixed_start_xy is not None:
            sx, sy = _nearest_free(
                int(cfg.fixed_start_xy[0]),
                int(cfg.fixed_start_xy[1]),
                endpoint="start",
            )
        else:
            start_candidates = _largest_component()
            si = int(self._rng.integers(0, start_candidates.shape[0]))
            sy, sx = map(int, start_candidates[si])

        start_component_mask = _component_mask((sx, sy))
        start_component_cells = np.argwhere(start_component_mask)
        required_component_cells = 2
        if cfg.paper_track == "dynamic" or int(cfg.force_replan_count) > 0:
            required_component_cells = 24
        if start_component_cells.shape[0] < required_component_cells:
            start_candidates = _largest_component()
            if start_candidates.shape[0] < required_component_cells:
                raise ValueError("No connected free component with at least two cells")
            prev_start = (sx, sy)
            si = int(self._rng.integers(0, start_candidates.shape[0]))
            sy, sx = map(int, start_candidates[si])
            start_component_mask = _component_mask((sx, sy))
            start_component_cells = np.argwhere(start_component_mask)
            self.log_event(
                "fixed_endpoint_relaxed",
                endpoint="start",
                requested=prev_start,
                used=(sx, sy),
            )

        # ---------- Sample goal with minimum L1 distance in same component ----------
        candidate_cells = start_component_cells[
            (start_component_cells[:, 1] != sx) | (start_component_cells[:, 0] != sy)
        ]
        if candidate_cells.shape[0] == 0:
            raise ValueError("Cannot sample goal: start component has no alternative cells")

        if cfg.fixed_goal_xy is not None:
            req_goal = (int(cfg.fixed_goal_xy[0]), int(cfg.fixed_goal_xy[1]))
            goal_x, goal_y = _nearest_free(req_goal[0], req_goal[1], endpoint="goal")
            if (goal_x, goal_y) == (sx, sy) or (not start_component_mask[goal_y, goal_x]):
                dists = np.abs(candidate_cells[:, 1] - sx) + np.abs(candidate_cells[:, 0] - sy)
                idx = int(np.argmax(dists))
                goal_y, goal_x = map(int, candidate_cells[idx])
                self.log_event(
                    "fixed_endpoint_relaxed",
                    endpoint="goal",
                    requested=req_goal,
                    used=(goal_x, goal_y),
                )
        else:
            dists = np.abs(candidate_cells[:, 1] - sx) + np.abs(candidate_cells[:, 0] - sy)
            eligible = candidate_cells[dists >= min_l1]
            if eligible.shape[0] > 0:
                gi = int(self._rng.integers(0, eligible.shape[0]))
                goal_y, goal_x = map(int, eligible[gi])
            else:
                idx = int(np.argmax(dists))
                goal_y, goal_x = map(int, candidate_cells[idx])

        if cfg.paper_track == "dynamic" or int(cfg.force_replan_count) > 0:
            l1_goal = abs(goal_x - sx) + abs(goal_y - sy)
            if l1_goal < 8:
                dists = np.abs(candidate_cells[:, 1] - sx) + np.abs(candidate_cells[:, 0] - sy)
                idx = int(np.argmax(dists))
                goal_y, goal_x = map(int, candidate_cells[idx])

        gx, gy = goal_x, goal_y

        # ---------- Set internal state ----------
        self._agent_pos = np.array([sx, sy, safe_alt], dtype=np.int32)
        self._goal_pos = np.array([gx, gy, safe_alt], dtype=np.int32)

        # ---------- Late-init dynamics that need start/goal ----------
        if cfg.enable_moving_target:
            from uavbench.dynamics.moving_target import MovingTargetModel
            roads = getattr(self, "_roads_mask", np.zeros((H, W), dtype=bool))
            # Spawn target on a road cell far from UAV start
            road_yx = np.argwhere(roads)
            if len(road_yx) > 2:
                dists = np.abs(road_yx[:, 1] - sx) + np.abs(road_yx[:, 0] - sy)
                far_idx = int(np.argmax(dists))
                spawn_x, spawn_y = int(road_yx[far_idx, 1]), int(road_yx[far_idx, 0])
            else:
                spawn_x, spawn_y = H // 4, W // 4
            self._moving_target = MovingTargetModel(
                roads_mask=roads,
                start_pos=(spawn_x, spawn_y),
                goal_pos=(gx, gy),
                speed=cfg.target_speed,
                buffer_radius=cfg.target_buffer_radius,
                rng=self._rng,
            )

        if cfg.enable_intruders:
            from uavbench.dynamics.intruder import IntruderModel
            self._intruder_model = IntruderModel(
                map_shape=(H, W),
                num_intruders=cfg.num_intruders,
                spawn_zone=cfg.intruder_spawn_zone,
                target_area=(gx, gy),
                speed=cfg.intruder_speed,
                rng=self._rng,
            )

        if cfg.enable_dynamic_nfz:
            from uavbench.dynamics.dynamic_nfz import DynamicNFZModel
            self._dynamic_nfz = DynamicNFZModel(
                map_shape=(H, W),
                uav_start=(sx, sy),
                uav_goal=(gx, gy),
                num_zones=cfg.num_nfz_zones,
                expansion_rate=cfg.nfz_expansion_rate,
                max_radius=cfg.nfz_max_radius,
                rng=self._rng,
            )

        # ---------- Non-blocking risk layers + causal interaction engine ----------
        extra_cfg = cfg.extra or {}
        self._interactions_enabled = not bool(extra_cfg.get("disable_interactions", False))
        self._population_risk_enabled = not bool(extra_cfg.get("disable_population_risk", False))
        self._guardrail_enabled = not bool(extra_cfg.get("disable_feasibility_guardrail", False))

        base_risk = getattr(self, "_risk_map", None)
        self._population_model = PopulationRiskModel(
            map_shape=(H, W),
            base_risk=base_risk,
            rng=self._rng,
        )

        adv_enabled = bool(extra_cfg.get("enable_adversarial_uav", False))
        adv_count = int(extra_cfg.get("num_adversarial_uavs", 0))
        adv_radius = int(extra_cfg.get("adversarial_safety_radius", 6))
        self._adversarial_uav: AdversarialUAVModel | None = None
        if adv_enabled and adv_count > 0:
            self._adversarial_uav = AdversarialUAVModel(
                map_shape=(H, W),
                num_uavs=adv_count,
                safety_radius=adv_radius,
                rng=self._rng,
            )

        self._interaction_engine = InteractionEngine(
            map_shape=(H, W),
            roads_mask=getattr(self, "_roads_mask", None),
            coupling_strength=float(extra_cfg.get("interaction_coupling_strength", 1.0)),
        )

        # ---------- Forced path interdiction scheduler (paper protocol) ----------
        self._init_forced_interdictions((sx, sy), (gx, gy))
        self._emergency_corridor_mask = self._build_emergency_corridor_mask((sx, sy), (gx, gy), width=2)

        # ---------- Build observation + info ----------
        obs = self._build_observation()

        info: dict[str, Any] = {
            "scenario_name": getattr(cfg, "name", "unknown"),
            "domain": getattr(cfg.domain, "value", str(cfg.domain)),
            "difficulty": getattr(cfg.difficulty, "value", str(cfg.difficulty)),
            "paper_track": cfg.paper_track,
            "map_source": cfg.map_source,
            "safe_altitude": int(safe_alt),
            "min_start_goal_l1": int(min_l1),
            "free_cells": int(free_cells.shape[0]),
            "map_shape": (H, W),
        }
        self.log_event(
            "paper_protocol_initialized",
            paper_track=cfg.paper_track,
            interdiction_reference_planner=cfg.interdiction_reference_planner.value,
            plan_budget_ms=(
                cfg.plan_budget_dynamic_ms if cfg.paper_track == "dynamic" else cfg.plan_budget_static_ms
            ),
            replan_every_steps=cfg.replan_every_steps,
            max_replans_per_episode=cfg.max_replans_per_episode,
            interactions_enabled=self._interactions_enabled,
            guardrail_enabled=self._guardrail_enabled,
        )
        return obs, info

    # --------------- Map generation / loading ----------------

    def _generate_synthetic_map(self, options: dict[str, Any]) -> None:
        """Generate a synthetic heightmap with random buildings (original behavior)."""
        cfg = self.config
        H = W = int(self.map_size)

        if H < 5:
            raise ValueError("map_size must be >= 5 for meaningful Urban scenarios.")
        if self.max_altitude < 1:
            raise ValueError("max_altitude must be >= 1.")

        building_density = float(getattr(cfg, "building_density", options.get("building_density", 0.30)))
        building_density = float(np.clip(building_density, 0.0, 1.0))

        building_level = int(getattr(cfg, "building_level", options.get("building_level", self.max_altitude)))
        building_level = int(np.clip(building_level, 0, self.max_altitude))

        extra_density_medium = float(getattr(cfg, "extra_density_medium", options.get("extra_density_medium", 0.10)))
        extra_density_hard = float(getattr(cfg, "extra_density_hard", options.get("extra_density_hard", 0.20)))
        no_fly_radius = int(getattr(cfg, "no_fly_radius", options.get("no_fly_radius", max(1, H // 8))))

        # Build base heightmap
        base_mask = (self._rng.random((H, W)) < building_density)
        self._heightmap = np.zeros((H, W), dtype=np.float32)
        self._heightmap[base_mask] = float(building_level)
        self._no_fly_mask = np.zeros((H, W), dtype=bool)

        # Difficulty-specific tweaks
        if cfg.difficulty == Difficulty.MEDIUM:
            extra_mask = (self._rng.random((H, W)) < extra_density_medium)
            self._heightmap[extra_mask] = float(building_level)

        elif cfg.difficulty == Difficulty.HARD:
            extra_mask = (self._rng.random((H, W)) < extra_density_hard)
            self._heightmap[extra_mask] = float(building_level)

            cy, cx = H // 2, W // 2
            yy, xx = np.ogrid[:H, :W]
            circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= no_fly_radius ** 2
            self._no_fly_mask[circle] = True

    def _load_osm_tile(self, tile_id: str) -> None:
        """Load a pre-rasterized OSM tile from data/maps/{tile_id}.npz.

        Converts meter-based heightmap to altitude levels (10m per level)
        and stores extra layers for future use.
        """
        npz_path = Path("data/maps") / f"{tile_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"OSM tile not found: {npz_path}. "
                f"Run: python -m tools.osm_pipeline.fetch --tile {tile_id} && "
                f"python -m tools.osm_pipeline.rasterize --tile {tile_id}"
            )

        data = np.load(str(npz_path))

        # Convert meters → altitude levels (10m per level, clamped to max_altitude)
        meters_per_level = 10.0
        raw_m = data["heightmap"]
        self._heightmap = np.clip(
            np.ceil(raw_m / meters_per_level), 0, self.max_altitude
        ).astype(np.float32)

        self._no_fly_mask = data["nfz_mask"].astype(bool)

        # Update map_size to match tile dimensions
        self.map_size = self._heightmap.shape[0]

        # Store extra layers for future phases (dynamics, visualization)
        self._roads_mask = data["roads_mask"].astype(bool)
        self._landuse_map = data["landuse_map"]
        self._risk_map = data["risk_map"]


    def _step_impl(self, action: int):
        """Urban step: 2D moves + explicit altitude up/down, with building/no-fly constraints.

        Returns:
            obs, reward, terminated, truncated, info
        """

        # ---------- 0) Hyperparameters (baseline) ----------
        w_z = 1.0           # weight for altitude mismatch in distance
        k_progress = 0.2    # shaping gain
        step_cost = 1.0     # per-step penalty
        p_building = 5.0    # penalty for attempted building collision
        p_no_fly = 8.0      # penalty for attempted no-fly violation
        r_goal = 50.0       # goal bonus
        max_steps = 4 * int(self.map_size)
        next_step_idx = self._step_count + 1

        # Paper protocol: deterministic interdictions that force replans.
        self._maybe_trigger_interdictions(next_step_idx)

        # ---------- 1) Read current state ----------
        ax, ay, az = map(int, self._agent_pos)
        gx, gy, gz = map(int, self._goal_pos)

        # Weighted L1 distance BEFORE the move
        prev_dist = float(abs(ax - gx) + abs(ay - gy) + w_z * abs(az - gz))

        # ---------- 2) Propose next position from action ----------
        # Recommended action mapping (Discrete(6)):
        # 0: up (y-1), 1: down (y+1), 2: left (x-1), 3: right (x+1),
        # 4: ascend (z+1), 5: descend (z-1)
        nx, ny, nz = ax, ay, az

        if action == 0:
            ny -= 1
        elif action == 1:
            ny += 1
        elif action == 2:
            nx -= 1
        elif action == 3:
            nx += 1
        elif action == 4:
            nz += 1
        elif action == 5:
            nz -= 1
        else:
            raise ValueError(f"Invalid action={action}. Expected 0..5.")

        # ---------- 3) Clamp to bounds ----------
        nx = int(np.clip(nx, 0, self.map_size - 1))
        ny = int(np.clip(ny, 0, self.map_size - 1))
        nz = int(np.clip(nz, 0, self.max_altitude))

        # ---------- 4) Constraint checks (no-fly, buildings) ----------
        attempted_no_fly = bool(self._no_fly_mask[ny, nx])

        terrain_h = float(self._heightmap[ny, nx])  # levels-only: 0..max_altitude
        attempted_building_collision = bool(nz <= terrain_h)

        # Decide whether to accept the move
        accepted = True
        attempted_forced_block = False
        attempted_traffic_closure = False
        if attempted_no_fly:
            accepted = False
            self.log_event("no_fly_violation_attempt", x=nx, y=ny, z=nz)
        elif attempted_building_collision:
            accepted = False
            self.log_event(
                "collision_building_attempt",
                x=nx, y=ny, z=nz, height=terrain_h
            )

        # Forced interdictions from paper protocol
        if accepted and self._forced_block_mask[ny, nx]:
            accepted = False
            attempted_forced_block = True
            self.log_event("forced_interdiction_block", x=nx, y=ny, z=nz)

        # Road-closure mask from interaction engine
        if accepted and self._traffic_closure_mask[ny, nx]:
            accepted = False
            attempted_traffic_closure = True
            self.log_event("traffic_closure_block", x=nx, y=ny, z=nz)

        # Fire blocks movement (only when config flag is set)
        attempted_fire_block = False
        if (accepted and self.config.fire_blocks_movement
                and self._fire_model is not None
                and self._fire_model.fire_mask[ny, nx]):
            accepted = False
            attempted_fire_block = True
            self.log_event("fire_block", x=nx, y=ny, z=nz)

        # Traffic blocks movement (vehicle buffer zones reject movement)
        attempted_traffic_block = False
        if (accepted and self.config.traffic_blocks_movement
                and self._traffic_model is not None):
            for vy, vx in self._traffic_model.vehicle_positions:
                dist = abs(nx - int(vx)) + abs(ny - int(vy))
                if dist <= 5:
                    accepted = False
                    attempted_traffic_block = True
                    self.log_event("traffic_block", x=nx, y=ny, z=nz,
                                   vx=int(vx), vy=int(vy), dist=dist)
                    break

        # Moving target buffer blocks movement
        attempted_target_block = False
        if accepted and self._moving_target is not None:
            tp = self._moving_target.current_position
            dist = abs(nx - int(tp[0])) + abs(ny - int(tp[1]))
            if dist <= self._moving_target.buffer_radius:
                accepted = False
                attempted_target_block = True
                self.log_event("target_block", x=nx, y=ny, z=nz)

        # Intruder buffer blocks movement
        attempted_intruder_block = False
        if accepted and self._intruder_model is not None:
            for pos in self._intruder_model.active_positions:
                dist = abs(nx - int(pos[0])) + abs(ny - int(pos[1]))
                if dist <= self._intruder_model.buffer_radius:
                    accepted = False
                    attempted_intruder_block = True
                    self.log_event("intruder_block", x=nx, y=ny, z=nz)
                    break

        # Dynamic NFZ blocks movement
        attempted_nfz_block = False
        if accepted and self._dynamic_nfz is not None:
            nfz_mask = self._dynamic_nfz.get_nfz_mask()
            if nfz_mask[ny, nx]:
                accepted = False
                attempted_nfz_block = True
                self.log_event("dynamic_nfz_block", x=nx, y=ny, z=nz)

        if accepted:
            self._agent_pos = np.array([nx, ny, nz], dtype=np.int32)

        # ---------- 5) Build observation AFTER (potential) move ----------
        obs = self._build_observation()

        # ---------- 6) Distance AFTER the move ----------
        ax2, ay2, az2 = map(int, self._agent_pos)
        new_dist = float(abs(ax2 - gx) + abs(ay2 - gy) + w_z * abs(az2 - gz))

        # ---------- 6b) Advance dynamic layers ----------
        fire_exposure = False
        traffic_proximity = False
        fire_cells = 0
        vehicles_near = 0
        H, W = self._heightmap.shape

        smoke_intensity = 0.0

        if self._fire_model is not None:
            self._fire_model.step(dt=1.0)
            fire_cells = int(self._fire_model.fire_mask.sum())
            if self._fire_model.fire_mask[ay2, ax2]:
                fire_exposure = True
                self.log_event("fire_exposure", x=ax2, y=ay2, z=az2)
            # Smoke exposure check
            smoke_intensity = float(self._fire_model.smoke_mask[ay2, ax2])
            if smoke_intensity > 0.3:
                self.log_event("smoke_exposure", x=ax2, y=ay2, z=az2,
                               intensity=round(smoke_intensity, 3))

        if self._traffic_model is not None:
            self._traffic_model.step(dt=1.0)
            traffic_buffer = self._traffic_model.get_occupancy_mask((H, W))
            if traffic_buffer[ay2, ax2]:
                traffic_proximity = True
                vehicles_near = int(len(self._traffic_model.vehicle_positions))
                self.log_event("traffic_proximity", x=ax2, y=ay2, z=az2)

        # Step new dynamics
        if self._moving_target is not None:
            self._moving_target.step(dt=1.0)
        if self._intruder_model is not None:
            self._intruder_model.step(dt=1.0)
        if self._dynamic_nfz is not None:
            self._dynamic_nfz.step(dt=1.0)

        # Causal interactions update
        fire_mask = self._fire_model.fire_mask if self._fire_model is not None else None
        traffic_positions = (
            self._traffic_model.vehicle_positions if self._traffic_model is not None else None
        )

        # Risk layers (non-blocking)
        if self._population_risk_enabled:
            self._population_model.step(
                fire_mask=fire_mask,
                traffic_positions=traffic_positions,
            )
        if self._adversarial_uav is not None:
            self._adversarial_uav.step(dt=1.0)

        if self._interactions_enabled:
            self._interaction_metrics_last = self._interaction_engine.update(
                step_idx=next_step_idx,
                fire_mask=fire_mask,
                traffic_positions=traffic_positions,
                dynamic_nfz=self._dynamic_nfz,
                risk_map=(self._population_model.risk_map if self._population_risk_enabled else np.zeros((H, W), dtype=np.float32)),
            )
            self._traffic_closure_mask = self._interaction_engine.traffic_closure_mask
        else:
            self._interaction_metrics_last = {
                "step_idx": float(next_step_idx),
                "fire_cells": float(np.sum(fire_mask)) if fire_mask is not None else 0.0,
                "fire_fraction": float(np.mean(fire_mask)) if fire_mask is not None else 0.0,
                "traffic_closure_cells": 0.0,
                "interaction_fire_nfz_overlap_ratio": 0.0,
                "interaction_fire_road_closure_rate": 0.0,
                "interaction_congestion_risk_corr": 0.0,
                "dynamic_block_entropy": 0.0,
                "traffic_congestion_cells": 0.0,
                "nfz_cells": 0.0,
            }
            self._traffic_closure_mask.fill(False)

        pop_risk = (
            self._population_model.risk_map
            if self._population_risk_enabled
            else np.zeros((H, W), dtype=np.float32)
        )
        adv_risk = (
            self._adversarial_uav.get_risk_map((H, W))
            if self._adversarial_uav is not None
            else np.zeros((H, W), dtype=np.float32)
        )
        smoke_risk = (
            self._fire_model.smoke_mask if self._fire_model is not None else np.zeros((H, W), dtype=np.float32)
        )
        w_pop = float(self.config.risk_weight_population)
        w_adv = float(self.config.risk_weight_adversarial)
        w_smoke = float(self.config.risk_weight_smoke)
        self._risk_cost_map = np.clip(
            w_pop * pop_risk + w_adv * adv_risk + w_smoke * smoke_risk,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        # Feasibility guardrail: dynamic updates must not make the world unsolvable.
        guardrail_depth = 0
        if self._guardrail_enabled:
            guardrail_depth = self._enforce_feasibility_guardrail((ax2, ay2), (gx, gy))
            guardrail_status = dict(self._last_guardrail_status)
            if bool(guardrail_status.get("reachability_failed_before_relax", False)):
                self.log_event(
                    "feasibility_relaxation_applied",
                    step=next_step_idx,
                    **guardrail_status,
                )
        else:
            runtime_mask = self._build_runtime_blocking_mask()
            free = (self._heightmap <= 0) & (~self._no_fly_mask) & (~runtime_mask)
            guardrail_status = {
                "reachability_failed_before_relax": False,
                "relaxation_applied": {},
                "corridor_fallback_used": False,
                "feasible_after_guardrail": bool(self._is_reachable(free, (ax2, ay2), (gx, gy))),
                "guardrail_disabled": True,
            }
            self._last_guardrail_status = dict(guardrail_status)

        runtime_mask = self._build_runtime_blocking_mask()
        layer_weights = np.array(
            [
                float(np.mean(self._forced_block_mask)),
                float(np.mean(self._traffic_closure_mask)),
                float(np.mean(runtime_mask)),
                float(np.mean(fire_mask)) if fire_mask is not None else 0.0,
            ],
            dtype=np.float64,
        )
        layer_weights = layer_weights[layer_weights > 0.0]
        env_entropy = 0.0
        if layer_weights.size > 0:
            probs = layer_weights / float(np.sum(layer_weights))
            env_entropy = float(
                -np.sum(probs * np.log(probs)) / np.log(float(max(len(probs), 2)))
            )
        self._interaction_metrics_last["dynamic_block_entropy_env"] = float(np.clip(env_entropy, 0.0, 1.0))
        self._interaction_metrics_last["interdiction_hit_rate"] = float(
            self._interdiction_hits / max(self._interdiction_total, 1)
        )
        self._interaction_metrics_last["interdictions_triggered"] = float(self._interdiction_total)
        self._interaction_metrics_history.append(dict(self._interaction_metrics_last))

        # ---------- 7) Reward (robust baseline) ----------
        # Step cost encourages shorter paths.
        reward = -float(step_cost)

        # Progress shaping (dense signal)
        reward += float(k_progress) * (prev_dist - new_dist)

        # Safety penalties (only if attempt happened)
        if attempted_no_fly:
            reward -= float(p_no_fly)
        elif attempted_building_collision:
            reward -= float(p_building)

        # Collision termination penalty (UAV-ON standard)
        collision = bool(
            self.config.terminate_on_collision
            and (attempted_building_collision or attempted_no_fly)
        )
        if collision:
            reward -= 25.0
            ctype = "collision_building" if attempted_building_collision else "collision_nfz"
            self.log_event("collision_terminated", x=nx, y=ny, z=nz,
                           collision_type=ctype)

        # Dynamic layer penalties
        if fire_exposure:
            reward -= 20.0
        if smoke_intensity > 0.3:
            reward -= 5.0 * smoke_intensity
        if traffic_proximity:
            reward -= 5.0
        reward -= 2.0 * float(self._risk_cost_map[ay2, ax2])

        # Goal achievement
        reached = bool(np.array_equal(self._agent_pos, self._goal_pos))
        if reached:
            reward += float(r_goal)

        # ---------- 8) Termination / truncation ----------
        terminated = reached or collision

        # IMPORTANT: base.step() increments _step_count AFTER this returns.
        # So the next step index will be current + 1.
        truncated = bool((next_step_idx >= max_steps) and (not terminated))

        # ---------- 9) Info ----------
        # Compute proximity/interaction metrics (Objective 8)
        hazard_proximity_time = 0.0
        smoke_exposure_duration = 0.0
        vehicle_near_miss_count = 0
        nfz_violation_time = 0.0

        # Fire proximity: blocking + thermal + smoke radii
        if self._fire_model is not None:
            fm = self._fire_model.fire_mask
            # Thermal radius (within 3 cells of fire)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    fy, fx = ay2 + dy, ax2 + dx
                    if 0 <= fy < H and 0 <= fx < W and abs(dy) + abs(dx) <= 3:
                        if fm[fy, fx]:
                            hazard_proximity_time += 1.0
                            break
                if hazard_proximity_time > 0:
                    break
            if smoke_intensity > 0.1:
                smoke_exposure_duration = smoke_intensity

        # Vehicle near-miss (within 2 cells)
        if self._traffic_model is not None:
            for vy, vx in self._traffic_model.vehicle_positions:
                dist = abs(ax2 - int(vx)) + abs(ay2 - int(vy))
                if dist <= 2:
                    vehicle_near_miss_count += 1

        # Dynamic NFZ violation time
        if self._dynamic_nfz is not None:
            nfz_mask = self._dynamic_nfz.get_nfz_mask()
            if nfz_mask[ay2, ax2]:
                nfz_violation_time = 1.0

        info: dict[str, Any] = {
            # Episode / task
            "scenario_name": self.config.name,
            "domain": self.config.domain.value,
            "difficulty": self.config.difficulty.value,

            # Goal status
            "reached_goal": reached,
            "distance_to_goal": float(new_dist),
            "collision_terminated": collision,
            "termination_reason": (
                ("collision_building" if attempted_building_collision else "collision_nfz")
                if collision
                else ("success" if reached else ("timeout" if truncated else "in_progress"))
            ),

            # Safety / constraints
            "attempted_building_collision": attempted_building_collision,
            "attempted_no_fly": attempted_no_fly,
            "attempted_fire_block": attempted_fire_block,
            "attempted_traffic_block": attempted_traffic_block,
            "attempted_forced_block": attempted_forced_block,
            "attempted_traffic_closure": attempted_traffic_closure,
            "attempted_target_block": attempted_target_block,
            "attempted_intruder_block": attempted_intruder_block,
            "attempted_nfz_block": attempted_nfz_block,
            "accepted_move": bool(accepted),

            # State for metrics (path length, energy proxy κ.λπ. τα χτίζεις έξω)
            "agent_pos": (int(ax2), int(ay2), int(az2)),
            "goal_pos": (int(gx), int(gy), int(gz)),
            "terrain_height": float(terrain_h),

            # Dynamic layers
            "fire_active": self._fire_model is not None,
            "traffic_active": self._traffic_model is not None,
            "fire_exposure": fire_exposure,
            "smoke_intensity": smoke_intensity,
            "traffic_proximity": traffic_proximity,
            "fire_cells": fire_cells,
            "vehicles_near": vehicles_near,
            "risk_cost": float(self._risk_cost_map[ay2, ax2]),
            "forced_replans_triggered": int(self._forced_replans_triggered),
            "interdiction_hit_rate": float(self._interdiction_hits / max(self._interdiction_total, 1)),
            "feasible_after_guardrail": bool(guardrail_status.get("feasible_after_guardrail", True)),
            "guardrail_corridor_fallback_used": bool(guardrail_status.get("corridor_fallback_used", False)),
            "guardrail_depth": int(guardrail_status.get("guardrail_depth", 0)),

            # Drone-dynamic interaction metrics (Objective 8)
            "hazard_proximity_time": float(hazard_proximity_time),
            "smoke_exposure_duration": float(smoke_exposure_duration),
            "vehicle_near_miss_count": int(vehicle_near_miss_count),
            "nfz_violation_time": float(nfz_violation_time),

            # Optional debugging
            "step_index": int(self._step_count + 1),
        }
        return obs, float(reward), terminated, truncated, info

    def _init_forced_interdictions(self, start_xy: GridPos, goal_xy: GridPos) -> None:
        """Prepare deterministic 1-2 path interdictions for dynamic paper track."""
        cfg = self.config
        self._forced_interdictions = []
        self._reference_path = []
        self._reference_corridor_mask.fill(False)
        if bool((cfg.extra or {}).get("disable_forced_interdictions", False)):
            return

        force_count = int(cfg.force_replan_count)
        if force_count == 0 and cfg.paper_track == "dynamic":
            force_count = 2
        force_count = int(np.clip(force_count, 0, 2))
        if force_count == 0:
            return

        ref_planner = cfg.interdiction_reference_planner.value
        planner_cls: type[Any]
        if ref_planner == "astar":
            planner_cls = AStarPlanner
        else:
            planner_cls = ThetaStarPlanner

        planner = planner_cls(self._heightmap, self._no_fly_mask)
        plan = planner.plan(start_xy, goal_xy)
        if plan.success and len(plan.path) >= 6:
            base_path = list(plan.path)
        else:
            # Deterministic fallback corridor to keep forced-replan protocol active.
            x0, y0 = start_xy
            x1, y1 = goal_xy
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            line_path: list[GridPos] = []
            for k in range(steps + 1):
                t = k / steps
                px = int(round(x0 + (x1 - x0) * t))
                py = int(round(y0 + (y1 - y0) * t))
                if not line_path or line_path[-1] != (px, py):
                    line_path.append((px, py))
            if len(line_path) < 6:
                return
            base_path = line_path
            self.log_event(
                "interdiction_path_fallback",
                reason="initial_plan_unavailable",
                path_length=len(base_path),
                reference_planner=ref_planner,
            )
        self._reference_path = list(base_path)
        for x, y in base_path:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self._reference_corridor_mask[y, x] = True

        cut_a = int(0.30 * (len(base_path) - 1))
        cut_b = int(0.65 * (len(base_path) - 1))
        cut_a = int(np.clip(cut_a, 2, len(base_path) - 3))
        cut_b = int(np.clip(cut_b, cut_a + 2, len(base_path) - 2))
        cut_points = [base_path[cut_a], base_path[cut_b]]

        t1 = cfg.event_t1 if cfg.event_t1 is not None else 12
        t2 = cfg.event_t2 if cfg.event_t2 is not None else 28
        times = [int(t1), int(t2)]

        for idx in range(force_count):
            radius_scale = float((cfg.extra or {}).get("interdiction_radius_scale", 1.0))
            base_radius = 3 if idx == 0 else 4
            radius = int(np.clip(round(base_radius * radius_scale), 2, 8))
            self._forced_interdictions.append(
                {
                    "name": f"path_interdiction_{idx + 1}",
                    "step": times[idx],
                    "point": cut_points[idx],
                    "radius": radius,
                    "triggered": False,
                    "reference_planner": ref_planner,
                }
            )

    def _maybe_trigger_interdictions(self, step_idx: int) -> None:
        if not self._forced_interdictions:
            return

        for event in self._forced_interdictions:
            if event["triggered"] or step_idx < int(event["step"]):
                continue
            cx, cy = event["point"]
            radius = int(event["radius"])
            hit_reference = False
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) <= radius:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                            self._forced_block_mask[ny, nx] = True
                            if self._reference_corridor_mask[ny, nx]:
                                hit_reference = True
            event["triggered"] = True
            self._forced_replans_triggered += 1
            self._interdiction_total += 1
            if hit_reference:
                self._interdiction_hits += 1
            self.log_event(
                event["name"],
                x=int(cx),
                y=int(cy),
                step=step_idx,
                reference_planner=event.get("reference_planner", "theta_star"),
                reference_hit=bool(hit_reference),
            )
            self.log_event("forced_replan_triggered", reason=event["name"], step=step_idx)

    def _build_emergency_corridor_mask(
        self,
        start_xy: GridPos,
        goal_xy: GridPos,
        width: int = 2,
    ) -> np.ndarray:
        mask = np.zeros((self.map_size, self.map_size), dtype=bool)
        planner = AStarPlanner(self._heightmap, self._no_fly_mask)
        static_path = planner.plan(start_xy, goal_xy).path
        if len(static_path) < 2:
            x0, y0 = start_xy
            x1, y1 = goal_xy
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            static_path = []
            for k in range(steps + 1):
                t = k / steps
                x = int(round(x0 + (x1 - x0) * t))
                y = int(round(y0 + (y1 - y0) * t))
                static_path.append((x, y))

        for x, y in static_path:
            for dy in range(-width, width + 1):
                for dx in range(-width, width + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.map_size and 0 <= nx < self.map_size:
                        if abs(dx) + abs(dy) <= width and self._heightmap[ny, nx] <= 0 and not self._no_fly_mask[ny, nx]:
                            mask[ny, nx] = True
        return mask

    def _build_runtime_blocking_mask(self) -> np.ndarray:
        mask = np.zeros((self.map_size, self.map_size), dtype=bool)
        mask |= self._forced_block_mask
        mask |= self._traffic_closure_mask
        if self._fire_model is not None and self.config.fire_blocks_movement:
            mask |= self._fire_model.fire_mask
        if self._traffic_model is not None and self.config.traffic_blocks_movement:
            mask |= self._traffic_model.get_occupancy_mask((self.map_size, self.map_size), buffer_radius=5)
        if self._intruder_model is not None:
            mask |= self._intruder_model.get_buffer_mask((self.map_size, self.map_size))
        if self._dynamic_nfz is not None:
            mask |= self._dynamic_nfz.get_nfz_mask()
        if self._emergency_corridor_active and self.config.emergency_corridor_enabled:
            mask &= ~self._emergency_corridor_mask
        return mask

    def _enforce_feasibility_guardrail(self, current_xy: GridPos, goal_xy: GridPos) -> int:
        """Ensure dynamic updates do not make path permanently unreachable.

        Returns guardrail_depth: 0=no action, 1=forced blocks cleared,
        2=NFZ/closures relaxed, 3=emergency corridor fallback.
        """
        status: dict[str, Any] = {
            "reachability_failed_before_relax": False,
            "relaxation_applied": {},
            "corridor_fallback_used": False,
            "feasible_after_guardrail": True,
            "guardrail_depth": 0,
        }
        relaxation: dict[str, Any] = {}
        relaxed = False
        depth = 0

        runtime_mask = self._build_runtime_blocking_mask()
        free = (self._heightmap <= 0) & (~self._no_fly_mask) & (~runtime_mask)
        reachable = self._is_reachable(free, current_xy, goal_xy)
        if reachable:
            self._guardrail_unreachable_streak = 0
            self._last_guardrail_status = status
            return 0

        status["reachability_failed_before_relax"] = True

        # Step 1 (depth=1): relax forced interdiction this tick.
        depth = 1
        forced_before = int(np.sum(self._forced_block_mask))
        if forced_before > 0:
            self._forced_block_mask[:] = False
            relaxation["forced_blocks_cleared"] = forced_before
            relaxed = True
            runtime_mask = self._build_runtime_blocking_mask()
            free = (self._heightmap <= 0) & (~self._no_fly_mask) & (~runtime_mask)
            reachable = self._is_reachable(free, current_xy, goal_xy)
            if reachable:
                status["relaxation_applied"] = relaxation
                status["feasible_after_guardrail"] = True
                status["guardrail_depth"] = depth
                self._guardrail_unreachable_streak = 0
                self._last_guardrail_status = status
                return depth

        # Step 2 (depth=2): reduce NFZ growth / closures this tick.
        depth = 2
        nfz_rate_delta = 0.0
        nfz_radius_delta = 0.0
        if self._dynamic_nfz is not None:
            if hasattr(self._dynamic_nfz, "expansion_rate"):
                prev_rate = float(getattr(self._dynamic_nfz, "expansion_rate"))
                new_rate = max(0.1, prev_rate * 0.70)
                setattr(self._dynamic_nfz, "expansion_rate", float(new_rate))
                nfz_rate_delta = prev_rate - new_rate
            if hasattr(self._dynamic_nfz, "radii"):
                prev_radii = np.asarray(getattr(self._dynamic_nfz, "radii"), dtype=np.float32)
                new_radii = np.maximum(prev_radii - 2.0, 4.0)
                setattr(self._dynamic_nfz, "radii", new_radii)
                nfz_radius_delta = float(np.mean(prev_radii - new_radii))
        closures_removed = 0
        if np.any(self._traffic_closure_mask):
            to_relax = self._traffic_closure_mask & self._emergency_corridor_mask
            closures_removed = int(np.sum(to_relax))
            if closures_removed > 0:
                self._traffic_closure_mask[to_relax] = False
            else:
                closures_removed = int(np.sum(self._traffic_closure_mask))
                self._traffic_closure_mask[:] = False
        if nfz_rate_delta > 0.0 or nfz_radius_delta > 0.0 or closures_removed > 0:
            relaxation["nfz_rate_delta"] = round(float(nfz_rate_delta), 6)
            relaxation["nfz_radius_delta"] = round(float(nfz_radius_delta), 6)
            relaxation["closures_removed"] = int(closures_removed)
            relaxed = True

        runtime_mask = self._build_runtime_blocking_mask()
        free = (self._heightmap <= 0) & (~self._no_fly_mask) & (~runtime_mask)
        reachable = self._is_reachable(free, current_xy, goal_xy)
        if reachable:
            status["relaxation_applied"] = relaxation
            status["feasible_after_guardrail"] = True
            status["guardrail_depth"] = depth
            self._guardrail_unreachable_streak = 0
            self._last_guardrail_status = status
            return depth

        # Step 3 (depth=3): emergency corridor fallback.
        depth = 3
        if self.config.emergency_corridor_enabled:
            self._emergency_corridor_active = True
            status["corridor_fallback_used"] = True
            self._forced_block_mask[self._emergency_corridor_mask] = False
            self._traffic_closure_mask[self._emergency_corridor_mask] = False
            runtime_mask = self._build_runtime_blocking_mask()
            free = (self._heightmap <= 0) & (~self._no_fly_mask) & (~runtime_mask)
            reachable = self._is_reachable(free, current_xy, goal_xy)
            if not reachable and self._guardrail_unreachable_streak >= 1:
                # Hard deconfliction fallback: emergency authority opens all dynamic blocks.
                self._forced_block_mask[:] = False
                self._traffic_closure_mask[:] = False
                if self._dynamic_nfz is not None and hasattr(self._dynamic_nfz, "radii"):
                    self._dynamic_nfz.radii = np.full_like(self._dynamic_nfz.radii, 4.0)
                self._emergency_corridor_mask |= self._build_emergency_corridor_mask(current_xy, goal_xy, width=3)
                runtime_mask = self._build_runtime_blocking_mask()
                free = (self._heightmap <= 0) & (~self._no_fly_mask) & (~runtime_mask)
                reachable = self._is_reachable(free, current_xy, goal_xy)
                relaxation["hard_deconfliction"] = True

        status["relaxation_applied"] = relaxation
        status["feasible_after_guardrail"] = bool(reachable)
        status["guardrail_depth"] = depth
        if reachable:
            self._guardrail_unreachable_streak = 0
        else:
            self._guardrail_unreachable_streak += 1
        self._last_guardrail_status = status
        return depth

    def _is_reachable(self, free_mask: np.ndarray, start_xy: GridPos, goal_xy: GridPos) -> bool:
        from collections import deque

        sx, sy = start_xy
        gx, gy = goal_xy
        if not (0 <= sx < self.map_size and 0 <= sy < self.map_size):
            return False
        if not (0 <= gx < self.map_size and 0 <= gy < self.map_size):
            return False
        if not free_mask[sy, sx] or not free_mask[gy, gx]:
            return False

        q = deque([(sx, sy)])
        visited = np.zeros_like(free_mask, dtype=bool)
        visited[sy, sx] = True
        while q:
            x, y = q.popleft()
            if (x, y) == (gx, gy):
                return True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if free_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((nx, ny))
        return False


    # --------------- Helpers ----------------

    def _build_observation(self) -> np.ndarray:
        x, y, z = self._agent_pos
        gx, gy, gz = self._goal_pos
        h = float(self._heightmap[int(y), int(x)])
        return np.array(
            [x, y, z, gx, gy, gz, h],
            dtype=np.float32,
        )

    def export_planner_inputs(self) -> tuple[np.ndarray, np.ndarray, GridPos, GridPos]:
        """Return planner-ready inputs: (heightmap, no_fly_mask, start_xy, goal_xy).

        - heightmap: float32 [H,W]
        - no_fly_mask: bool [H,W]
        - start_xy: (x,y)
        - goal_xy: (x,y)

        Notes:
        - Assumes env has been reset() so internal state is initialized.
        - Returns copies to avoid accidental external mutation.
        """
        if not hasattr(self, "_heightmap") or self._heightmap is None:
            raise RuntimeError("export_planner_inputs() called before reset(): heightmap not initialized.")
        if not hasattr(self, "_no_fly_mask") or self._no_fly_mask is None:
            raise RuntimeError("export_planner_inputs() called before reset(): no_fly_mask not initialized.")
        if not hasattr(self, "_agent_pos") or self._agent_pos is None:
            raise RuntimeError("export_planner_inputs() called before reset(): agent_pos not initialized.")
        if not hasattr(self, "_goal_pos") or self._goal_pos is None:
            raise RuntimeError("export_planner_inputs() called before reset(): goal_pos not initialized.")

        # internal positions are [x,y,z]
        ax, ay = int(self._agent_pos[0]), int(self._agent_pos[1])
        gx, gy = int(self._goal_pos[0]), int(self._goal_pos[1])

        heightmap = np.array(self._heightmap, copy=True)
        no_fly = np.array(self._no_fly_mask, copy=True)

        return heightmap, no_fly, (ax, ay), (gx, gy)

    def get_dynamic_state(self) -> dict[str, Any]:
        """Return all dynamic obstacle state for adaptive planners."""
        H, W = self._heightmap.shape
        state: dict[str, Any] = {
            "fire_mask": None,
            "burned_mask": None,
            "smoke_mask": None,
            "traffic_positions": None,
            "traffic_closure_mask": None,
            "moving_target_pos": None,
            "moving_target_buffer": None,
            "intruder_positions": None,
            "intruder_buffer": None,
            "dynamic_nfz_mask": None,
            "forced_block_mask": None,
            "emergency_corridor_mask": None,
            "population_risk_map": None,
            "adversarial_positions": None,
            "adversarial_risk_map": None,
            "risk_cost_map": None,
            "interaction_metrics": None,
            "interaction_metrics_history": None,
            "guardrail_status": None,
            "protocol_metrics": None,
        }
        if self._fire_model is not None:
            state["fire_mask"] = self._fire_model.fire_mask.copy()
            state["burned_mask"] = self._fire_model.burned_mask.copy()
            state["smoke_mask"] = self._fire_model.smoke_mask.copy()
        if self._traffic_model is not None:
            state["traffic_positions"] = self._traffic_model.vehicle_positions.copy()
        state["traffic_closure_mask"] = self._traffic_closure_mask.copy()
        if self._moving_target is not None:
            state["moving_target_pos"] = self._moving_target.current_position
            state["moving_target_buffer"] = self._moving_target.get_buffer_mask((H, W))
        if self._intruder_model is not None:
            state["intruder_positions"] = self._intruder_model.active_positions
            state["intruder_buffer"] = self._intruder_model.get_buffer_mask((H, W))
        if self._dynamic_nfz is not None:
            state["dynamic_nfz_mask"] = self._dynamic_nfz.get_nfz_mask()
        state["forced_block_mask"] = self._forced_block_mask.copy()
        state["emergency_corridor_mask"] = self._emergency_corridor_mask.copy()
        if self._population_risk_enabled:
            state["population_risk_map"] = self._population_model.risk_map.copy()
        else:
            state["population_risk_map"] = np.zeros((H, W), dtype=np.float32)
        if self._adversarial_uav is not None:
            state["adversarial_positions"] = self._adversarial_uav.positions
            state["adversarial_risk_map"] = self._adversarial_uav.get_risk_map((H, W))
        state["risk_cost_map"] = self._risk_cost_map.copy()
        state["interaction_metrics"] = dict(self._interaction_metrics_last)
        state["interaction_metrics_history"] = [dict(v) for v in self._interaction_metrics_history]
        state["guardrail_status"] = dict(self._last_guardrail_status)
        state["protocol_metrics"] = {
            "forced_replans_triggered": int(self._forced_replans_triggered),
            "interdiction_hit_rate": float(self._interdiction_hits / max(self._interdiction_total, 1)),
            "interdictions_triggered": int(self._interdiction_total),
            "reference_path_length": int(len(self._reference_path)),
            "emergency_corridor_active": bool(self._emergency_corridor_active),
        }
        return state
