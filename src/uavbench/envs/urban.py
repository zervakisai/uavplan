from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uavbench.envs.base import UAVBenchEnv
from uavbench.scenarios.schema import ScenarioConfig, Domain, Difficulty


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
            self._load_osm_tile(cfg.osm_tile_id)
        else:
            self._generate_synthetic_map(options)

        H, W = self._heightmap.shape

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

        # ---------- Sample start ----------
        si = int(self._rng.integers(0, free_cells.shape[0]))
        sy, sx = map(int, free_cells[si])

        # ---------- Sample goal with minimum L1 distance ----------
        gx = gy = None

        for _ in range(200):
            gi = int(self._rng.integers(0, free_cells.shape[0]))
            ty, tx = map(int, free_cells[gi])
            if (tx, ty) == (sx, sy):
                continue
            l1 = abs(tx - sx) + abs(ty - sy)
            if l1 >= min_l1:
                gx, gy = tx, ty
                break

        # Fallback: choose farthest free cell
        if gx is None:
            dists = np.abs(free_cells[:, 1] - sx) + np.abs(free_cells[:, 0] - sy)
            idx = int(np.argmax(dists))
            gy, gx = map(int, free_cells[idx])

        # ---------- Set internal state ----------
        self._agent_pos = np.array([sx, sy, safe_alt], dtype=np.int32)
        self._goal_pos = np.array([gx, gy, safe_alt], dtype=np.int32)

        # ---------- Build observation + info ----------
        obs = self._build_observation()

        info: dict[str, Any] = {
            "scenario_name": getattr(cfg, "name", "unknown"),
            "domain": getattr(cfg.domain, "value", str(cfg.domain)),
            "difficulty": getattr(cfg.difficulty, "value", str(cfg.difficulty)),
            "map_source": cfg.map_source,
            "safe_altitude": int(safe_alt),
            "min_start_goal_l1": int(min_l1),
            "free_cells": int(free_cells.shape[0]),
            "map_shape": (H, W),
        }
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
        if attempted_no_fly:
            accepted = False
            self.log_event("no_fly_violation_attempt", x=nx, y=ny, z=nz)
        elif attempted_building_collision:
            accepted = False
            self.log_event(
                "collision_building_attempt",
                x=nx, y=ny, z=nz, height=terrain_h
            )

        if accepted:
            self._agent_pos = np.array([nx, ny, nz], dtype=np.int32)

        # ---------- 5) Build observation AFTER (potential) move ----------
        obs = self._build_observation()

        # ---------- 6) Distance AFTER the move ----------
        ax2, ay2, az2 = map(int, self._agent_pos)
        new_dist = float(abs(ax2 - gx) + abs(ay2 - gy) + w_z * abs(az2 - gz))

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

        # Goal achievement
        reached = bool(np.array_equal(self._agent_pos, self._goal_pos))
        if reached:
            reward += float(r_goal)

        # ---------- 8) Termination / truncation ----------
        terminated = reached

        # IMPORTANT: base.step() increments _step_count AFTER this returns.
        # So the next step index will be current + 1.
        next_step_idx = self._step_count + 1
        truncated = bool((next_step_idx >= max_steps) and (not reached))

        # ---------- 9) Info ----------
        info: dict[str, Any] = {
            # Episode / task
            "scenario_name": self.config.name,
            "domain": self.config.domain.value,
            "difficulty": self.config.difficulty.value,

            # Goal status
            "reached_goal": reached,
            "distance_to_goal": float(new_dist),

            # Safety / constraints
            "attempted_building_collision": attempted_building_collision,
            "attempted_no_fly": attempted_no_fly,
            "accepted_move": bool(accepted),

            # State for metrics (path length, energy proxy κ.λπ. τα χτίζεις έξω)
            "agent_pos": (int(ax2), int(ay2), int(az2)),
            "goal_pos": (int(gx), int(gy), int(gz)),
            "terrain_height": float(terrain_h),

            # Optional debugging
            "step_index": int(self._step_count + 1),
        }
        return obs, float(reward), terminated, truncated, info
   


    

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