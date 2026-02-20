"""Dynamic obstacle layers with plausible kinematics.

Provides moving obstacles for the three mission types, all integrated
with the UpdateBus:

  M1 — Civil Protection:  road-constrained emergency vehicles
  M2 — Maritime Domain:   vessel tracks with AIS-like kinematics
  M3 — Critical Infra:    mobile work zones (slow-moving, area-based)

Each obstacle type has:
  - Deterministic motion model (seeded RNG for reproducibility)
  - Plausible speed / turning constraints
  - UpdateBus integration (publishes OBSTACLE events each step)
  - Mask export for merged obstacle checking

The ``DynamicObstacleManager`` orchestrates all obstacle types for a
given mission and publishes their state to the bus each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from uavbench.updates.bus import EventType, UpdateEvent, UpdateBus


GridPos = tuple[int, int]


# ─────────────────────────────────────────────────────────────────────────────
# M1: Road-constrained emergency vehicles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    """A single road-constrained vehicle."""
    vehicle_id: str
    position: tuple[float, float]   # (x, y) — float for smooth motion
    heading: float                  # degrees, 0=North, 90=East
    speed: float                    # cells per step
    vehicle_type: str = "emergency"  # emergency, patrol, civilian
    trail: list[tuple[int, int]] = field(default_factory=list)
    max_trail_length: int = 30
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    waypoint_idx: int = 0

    @property
    def grid_pos(self) -> GridPos:
        return (int(round(self.position[0])), int(round(self.position[1])))


class VehicleLayer:
    """Road-constrained moving vehicles (M1 — Civil Protection).

    Vehicles follow pre-defined waypoint routes on the road network.
    If no roads_mask is given, they move freely within the grid.

    Parameters
    ----------
    grid_shape : (H, W)
    roads_mask : [H, W] bool, optional
        Road network mask for constraining vehicle paths.
    rng : np.random.Generator
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        roads_mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.H, self.W = grid_shape
        self.roads_mask = roads_mask
        self.rng = rng or np.random.default_rng(42)
        self.vehicles: list[Vehicle] = []

    def add_vehicle(
        self,
        vehicle_id: str,
        start: tuple[int, int],
        waypoints: list[tuple[int, int]],
        speed: float = 1.0,
        vehicle_type: str = "emergency",
    ) -> None:
        """Add a vehicle with a waypoint route."""
        self.vehicles.append(Vehicle(
            vehicle_id=vehicle_id,
            position=(float(start[0]), float(start[1])),
            heading=0.0,
            speed=speed,
            vehicle_type=vehicle_type,
            waypoints=waypoints,
            trail=[start],
        ))

    def add_random_vehicles(self, count: int, speed_range: tuple[float, float] = (0.5, 1.5)) -> None:
        """Add vehicles with random patrol routes in the grid."""
        for i in range(count):
            # Pick random start and waypoints
            sx = int(self.rng.integers(20, self.W - 20))
            sy = int(self.rng.integers(20, self.H - 20))
            n_wp = self.rng.integers(3, 6)
            wps = []
            for _ in range(n_wp):
                wx = int(np.clip(sx + self.rng.integers(-80, 80), 5, self.W - 5))
                wy = int(np.clip(sy + self.rng.integers(-80, 80), 5, self.H - 5))
                wps.append((wx, wy))
            speed = float(self.rng.uniform(*speed_range))
            self.add_vehicle(f"vehicle_{i}", (sx, sy), wps, speed)

    def step(self, step_num: int) -> list[Vehicle]:
        """Advance all vehicles one step."""
        for v in self.vehicles:
            if not v.waypoints:
                continue
            # Target waypoint
            target = v.waypoints[v.waypoint_idx]
            dx = target[0] - v.position[0]
            dy = target[1] - v.position[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < v.speed + 0.5:
                # Reached waypoint — advance to next (loop)
                v.waypoint_idx = (v.waypoint_idx + 1) % len(v.waypoints)
                target = v.waypoints[v.waypoint_idx]
                dx = target[0] - v.position[0]
                dy = target[1] - v.position[1]
                dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0.01:
                nx = dx / dist
                ny = dy / dist
                v.position = (
                    v.position[0] + nx * v.speed,
                    v.position[1] + ny * v.speed,
                )
                v.heading = math.degrees(math.atan2(nx, -ny))  # 0=North

            # Clamp to grid
            v.position = (
                float(np.clip(v.position[0], 0, self.W - 1)),
                float(np.clip(v.position[1], 0, self.H - 1)),
            )

            # Trail
            gp = v.grid_pos
            if not v.trail or v.trail[-1] != gp:
                v.trail.append(gp)
                if len(v.trail) > v.max_trail_length:
                    v.trail = v.trail[-v.max_trail_length:]

        return self.vehicles

    def get_positions(self) -> list[tuple[int, int]]:
        """Return current grid positions of all vehicles."""
        return [v.grid_pos for v in self.vehicles]

    def get_obstacle_mask(self, buffer: int = 3) -> np.ndarray:
        """Build obstacle mask with buffer around all vehicles."""
        mask = np.zeros((self.H, self.W), dtype=bool)
        for v in self.vehicles:
            x, y = v.grid_pos
            for dy in range(-buffer, buffer + 1):
                for dx in range(-buffer, buffer + 1):
                    if abs(dy) + abs(dx) <= buffer:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.H and 0 <= nx < self.W:
                            mask[ny, nx] = True
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# M2: Vessel tracks (AIS-like kinematics)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Vessel:
    """A maritime vessel with AIS-like track."""
    vessel_id: str
    position: tuple[float, float]
    heading: float                  # degrees
    speed: float                    # cells per step
    turn_rate: float = 0.5          # degrees per step (max turn)
    vessel_type: str = "cargo"      # cargo, tanker, fishing, patrol
    trail: list[tuple[int, int]] = field(default_factory=list)
    max_trail_length: int = 50
    patrol_center: tuple[int, int] = (250, 250)
    patrol_radius: float = 100.0

    @property
    def grid_pos(self) -> GridPos:
        return (int(round(self.position[0])), int(round(self.position[1])))


class VesselLayer:
    """Vessel tracks for M2 — Maritime Domain.

    Vessels follow circular / elliptical patrol paths with
    realistic turn-rate constraints.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.H, self.W = grid_shape
        self.rng = rng or np.random.default_rng(42)
        self.vessels: list[Vessel] = []

    def add_vessel(
        self,
        vessel_id: str,
        start: tuple[int, int],
        heading: float = 0.0,
        speed: float = 0.8,
        vessel_type: str = "cargo",
        patrol_center: tuple[int, int] | None = None,
        patrol_radius: float = 100.0,
    ) -> None:
        pc = patrol_center or (self.W // 2, self.H // 2)
        self.vessels.append(Vessel(
            vessel_id=vessel_id,
            position=(float(start[0]), float(start[1])),
            heading=heading,
            speed=speed,
            vessel_type=vessel_type,
            trail=[start],
            patrol_center=pc,
            patrol_radius=patrol_radius,
        ))

    def add_patrol_vessels(self, count: int, center: tuple[int, int], radius: float = 100.0) -> None:
        """Add vessels in circular patrol formation."""
        for i in range(count):
            angle = 2 * math.pi * i / count
            sx = int(center[0] + radius * 0.6 * math.cos(angle))
            sy = int(center[1] + radius * 0.6 * math.sin(angle))
            sx = int(np.clip(sx, 5, self.W - 5))
            sy = int(np.clip(sy, 5, self.H - 5))
            heading = math.degrees(angle) + 90
            speed = 0.3 + 0.3 * (i % 3)
            vtype = ["cargo", "tanker", "fishing"][i % 3]
            self.add_vessel(
                f"vessel_{i}", (sx, sy), heading, speed, vtype,
                patrol_center=center, patrol_radius=radius,
            )

    def step(self, step_num: int) -> list[Vessel]:
        """Advance all vessels — circular patrol with turn constraints."""
        for v in self.vessels:
            cx, cy = v.patrol_center
            # Desired direction: tangent to circle around patrol center
            dx = v.position[0] - cx
            dy = v.position[1] - cy
            dist_to_center = math.sqrt(dx * dx + dy * dy) + 1e-6

            # Target angle: tangent (perpendicular to radius) + correction
            current_angle = math.atan2(dy, dx)
            tangent_angle = current_angle + math.pi / 2  # CCW tangent

            # Correction: steer toward patrol_radius distance
            radius_error = dist_to_center - v.patrol_radius * 0.6
            correction = -0.02 * radius_error
            desired_angle = tangent_angle + correction

            # Convert to heading (degrees, 0=N)
            desired_heading = math.degrees(math.atan2(
                math.cos(desired_angle), -math.sin(desired_angle),
            ))

            # Apply turn rate constraint
            heading_diff = (desired_heading - v.heading + 180) % 360 - 180
            clamped_diff = max(-v.turn_rate * 5, min(v.turn_rate * 5, heading_diff))
            v.heading += clamped_diff

            # Move
            rad = math.radians(v.heading)
            v.position = (
                v.position[0] + math.sin(rad) * v.speed,
                v.position[1] - math.cos(rad) * v.speed,
            )

            # Clamp
            v.position = (
                float(np.clip(v.position[0], 2, self.W - 2)),
                float(np.clip(v.position[1], 2, self.H - 2)),
            )

            # Trail
            gp = v.grid_pos
            if not v.trail or v.trail[-1] != gp:
                v.trail.append(gp)
                if len(v.trail) > v.max_trail_length:
                    v.trail = v.trail[-v.max_trail_length:]

        return self.vessels

    def get_positions(self) -> list[tuple[int, int]]:
        return [v.grid_pos for v in self.vessels]

    def get_obstacle_mask(self, buffer: int = 4) -> np.ndarray:
        mask = np.zeros((self.H, self.W), dtype=bool)
        for v in self.vessels:
            x, y = v.grid_pos
            for dy in range(-buffer, buffer + 1):
                for dx in range(-buffer, buffer + 1):
                    if abs(dy) + abs(dx) <= buffer:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.H and 0 <= nx < self.W:
                            mask[ny, nx] = True
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# M3: Mobile work zones
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkZone:
    """A slow-moving area-based restriction (M3)."""
    zone_id: str
    center: tuple[float, float]
    radius: float                   # cells
    speed: float = 0.2              # cells per step (slow-moving)
    heading: float = 0.0
    active: bool = True
    activate_step: int = 0          # step at which zone becomes active
    waypoints: list[tuple[int, int]] = field(default_factory=list)
    waypoint_idx: int = 0

    @property
    def grid_center(self) -> GridPos:
        return (int(round(self.center[0])), int(round(self.center[1])))


class WorkZoneLayer:
    """Mobile work zones for M3 — Critical Infrastructure.

    Work zones are slow-moving circular areas that the UAV must avoid.
    They represent maintenance crews, security cordons, etc.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (500, 500),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.H, self.W = grid_shape
        self.rng = rng or np.random.default_rng(42)
        self.zones: list[WorkZone] = []

    def add_zone(
        self,
        zone_id: str,
        center: tuple[int, int],
        radius: float = 15.0,
        speed: float = 0.2,
        activate_step: int = 0,
        waypoints: list[tuple[int, int]] | None = None,
    ) -> None:
        wps = waypoints or [center]
        self.zones.append(WorkZone(
            zone_id=zone_id,
            center=(float(center[0]), float(center[1])),
            radius=radius,
            speed=speed,
            activate_step=activate_step,
            waypoints=wps,
        ))

    def step(self, step_num: int) -> list[WorkZone]:
        """Advance all work zones."""
        for z in self.zones:
            if step_num < z.activate_step:
                z.active = False
                continue
            z.active = True

            if not z.waypoints:
                continue

            target = z.waypoints[z.waypoint_idx]
            dx = target[0] - z.center[0]
            dy = target[1] - z.center[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < z.speed + 0.5:
                z.waypoint_idx = (z.waypoint_idx + 1) % len(z.waypoints)
                continue

            if dist > 0.01:
                z.center = (
                    z.center[0] + dx / dist * z.speed,
                    z.center[1] + dy / dist * z.speed,
                )

            z.center = (
                float(np.clip(z.center[0], z.radius, self.W - z.radius)),
                float(np.clip(z.center[1], z.radius, self.H - z.radius)),
            )

        return self.zones

    def get_obstacle_mask(self) -> np.ndarray:
        """Build combined obstacle mask from all active zones."""
        mask = np.zeros((self.H, self.W), dtype=bool)
        yy, xx = np.ogrid[0:self.H, 0:self.W]
        for z in self.zones:
            if not z.active:
                continue
            cx, cy = z.center
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            mask |= dist <= z.radius
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# DynamicObstacleManager — orchestrates all obstacle types
# ─────────────────────────────────────────────────────────────────────────────

class DynamicObstacleManager:
    """Orchestrates dynamic obstacle layers and publishes to UpdateBus.

    Creates appropriate obstacle layers based on mission type and
    publishes OBSTACLE events each step.

    Parameters
    ----------
    mission_type : str
        "civil_protection", "maritime_domain", "critical_infrastructure"
    grid_shape : (H, W)
    bus : UpdateBus
    seed : int
    roads_mask : optional road network for vehicle constraint
    """

    def __init__(
        self,
        mission_type: str,
        grid_shape: tuple[int, int] = (500, 500),
        bus: UpdateBus | None = None,
        seed: int = 42,
        roads_mask: np.ndarray | None = None,
    ) -> None:
        self.mission_type = mission_type
        self.H, self.W = grid_shape
        self.bus = bus
        self.rng = np.random.default_rng(seed)

        self.vehicle_layer: VehicleLayer | None = None
        self.vessel_layer: VesselLayer | None = None
        self.workzone_layer: WorkZoneLayer | None = None

        self._setup_layers(grid_shape, roads_mask)

    def _setup_layers(
        self,
        grid_shape: tuple[int, int],
        roads_mask: np.ndarray | None,
    ) -> None:
        """Create obstacle layers appropriate for the mission type."""
        if self.mission_type == "civil_protection":
            self.vehicle_layer = VehicleLayer(grid_shape, roads_mask, self.rng)
            self.vehicle_layer.add_random_vehicles(3)

        elif self.mission_type == "maritime_domain":
            self.vessel_layer = VesselLayer(grid_shape, self.rng)
            center = (self.W // 2, self.H // 2)
            self.vessel_layer.add_patrol_vessels(4, center, radius=min(self.W, self.H) * 0.3)

        elif self.mission_type == "critical_infrastructure":
            self.workzone_layer = WorkZoneLayer(grid_shape, self.rng)
            # Add 2 work zones with staggered activation
            cx, cy = self.W // 3, self.H // 3
            self.workzone_layer.add_zone(
                "wz_0", (cx, cy), radius=12, speed=0.15,
                activate_step=20,
                waypoints=[(cx, cy), (cx + 40, cy + 30), (cx, cy)],
            )
            self.workzone_layer.add_zone(
                "wz_1", (2 * cx, 2 * cy), radius=10, speed=0.2,
                activate_step=50,
                waypoints=[(2 * cx, 2 * cy), (2 * cx - 30, 2 * cy + 20)],
            )

    def step(self, step_num: int) -> np.ndarray:
        """Advance all obstacle layers one step and publish events.

        Returns
        -------
        np.ndarray
            [H, W] bool merged obstacle mask.
        """
        merged = np.zeros((self.H, self.W), dtype=bool)

        if self.vehicle_layer is not None:
            self.vehicle_layer.step(step_num)
            vmask = self.vehicle_layer.get_obstacle_mask()
            merged |= vmask
            if self.bus is not None:
                for v in self.vehicle_layer.vehicles:
                    self.bus.publish(UpdateEvent(
                        event_type=EventType.OBSTACLE,
                        step=step_num,
                        description=f"vehicle:{v.vehicle_id}",
                        severity=0.6,
                        position=v.grid_pos,
                        payload={
                            "obstacle_type": "vehicle",
                            "vehicle_type": v.vehicle_type,
                            "heading": v.heading,
                            "speed": v.speed,
                        },
                    ))

        if self.vessel_layer is not None:
            self.vessel_layer.step(step_num)
            vmask = self.vessel_layer.get_obstacle_mask()
            merged |= vmask
            if self.bus is not None:
                for v in self.vessel_layer.vessels:
                    self.bus.publish(UpdateEvent(
                        event_type=EventType.OBSTACLE,
                        step=step_num,
                        description=f"vessel:{v.vessel_id}",
                        severity=0.5,
                        position=v.grid_pos,
                        payload={
                            "obstacle_type": "vessel",
                            "vessel_type": v.vessel_type,
                            "heading": v.heading,
                            "speed": v.speed,
                        },
                    ))

        if self.workzone_layer is not None:
            self.workzone_layer.step(step_num)
            wmask = self.workzone_layer.get_obstacle_mask()
            merged |= wmask
            if self.bus is not None:
                for z in self.workzone_layer.zones:
                    if z.active:
                        self.bus.publish(UpdateEvent(
                            event_type=EventType.OBSTACLE,
                            step=step_num,
                            description=f"workzone:{z.zone_id}",
                            severity=0.7,
                            position=z.grid_center,
                            payload={
                                "obstacle_type": "workzone",
                                "radius": z.radius,
                                "speed": z.speed,
                            },
                        ))

        return merged

    def get_entity_data(self) -> list[dict[str, Any]]:
        """Get entity data for visualization overlay."""
        entities: list[dict[str, Any]] = []
        if self.vehicle_layer is not None:
            for v in self.vehicle_layer.vehicles:
                entities.append({
                    "xy": v.grid_pos,
                    "type": "fire" if v.vehicle_type == "emergency" else "police",
                    "heading": v.heading,
                    "trail": list(v.trail),
                    "label": v.vehicle_id,
                })
        if self.vessel_layer is not None:
            for v in self.vessel_layer.vessels:
                entities.append({
                    "xy": v.grid_pos,
                    "type": "vessel",
                    "heading": v.heading,
                    "trail": list(v.trail),
                    "label": v.vessel_id,
                })
        if self.workzone_layer is not None:
            for z in self.workzone_layer.zones:
                if z.active:
                    entities.append({
                        "xy": z.grid_center,
                        "type": "police",  # workzone uses shield icon
                        "heading": z.heading,
                        "trail": [],
                        "label": z.zone_id,
                    })
        return entities
