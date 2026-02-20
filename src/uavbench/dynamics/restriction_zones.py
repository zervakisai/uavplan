"""Mission-grounded dynamic restriction zones (replaces DynamicNFZModel).

Each mission type produces operationally motivated restriction zones:

- **Civil Protection (TFR):** Temporary Flight Restriction derived from fire
  perimeter — dilated convex hull of each fire cluster.  Inspired by the
  2018 Attica (Penteli/Mati) wildfire and 2024 Attica wildfire response.

- **Maritime Domain (SAR Box + Port Exclusion):** Rectangular SAR
  coordination box that drifts with deterministic current, plus port
  exclusion sector.  Inspired by the *Agia Zoni II* oil-tanker sinking
  in the Saronic Gulf (Sept 2017).

- **Critical Infrastructure (Security Cordon):** Street-following polygon
  cordon that expands via BFS on the road network.  Inspired by the
  Dec 2021 Athens Metro bomb-threat evacuation (Syntagma/Monastiraki).

All zone dynamics are:
  - **activate → expand → hold** (no oscillation),
  - placed at operationally motivated locations (NOT along start→goal),
  - capped at a configurable max coverage fraction,
  - deterministic given the env RNG seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RestrictionZone:
    """Structured description of a single temporary restriction zone."""

    zone_id: str                          # "TFR-1", "SAR-BOX-1", "CORDON-1"
    zone_type: str                        # "tfr", "sar_box", "port_exclusion", "security_cordon"
    mission_type: str                     # "civil_protection", "maritime_domain", ...
    active: bool = False
    activation_step: int = 0
    severity: float = 1.0                 # 0..1 (1 = hard block)
    label: str = ""                       # Human-readable COP label
    source_incident: str = ""             # e.g., "fire_cluster_0", "sar_alert"
    expires_step: Optional[int] = None    # auto-deactivate at this step (None=permanent)
    risk_buffer_px: int = 10              # graded annulus width outside core mask
    # Geometry (at least one should be set)
    center: tuple[int, int] = (0, 0)      # (x, y)
    bbox: Optional[tuple[int, int, int, int]] = None   # x0, y0, x1, y1
    polygon: Optional[list[tuple[int, int]]] = None
    _mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def mask(self) -> Optional[np.ndarray]:
        return self._mask

    @mask.setter
    def mask(self, v: np.ndarray) -> None:
        self._mask = v


# ─────────────────────────────────────────────────────────────────────────────
# Mission Restriction Model
# ─────────────────────────────────────────────────────────────────────────────


class MissionRestrictionModel:
    """Mission-grounded dynamic restriction zones.

    Drop-in replacement for DynamicNFZModel.  Provides the same
    ``get_nfz_mask()`` API for backward compatibility, plus structured
    ``get_zones()`` for the renderer.

    Args:
        map_shape: (H, W)
        mission_type: one of "civil_protection", "maritime_domain",
            "critical_infrastructure"
        roads_mask: bool [H, W] — road network (for cordon BFS)
        heightmap: int [H, W] — building heights (for cordon anchoring)
        num_zones: target zone count
        max_coverage: peak fraction of map that zones may cover (hard cap)
        buffer_px: TFR buffer around fire perimeter (pixels)
        event_t1: activation step for first zone group
        event_t2: activation step for second zone group (expansion trigger)
        current_vec: (vx, vy) deterministic drift for SAR box (px/step)
        incident_point: (x, y) override for cordon centre
        rng: deterministic RNG
    """

    def __init__(
        self,
        map_shape: tuple[int, int],
        mission_type: str,
        *,
        roads_mask: Optional[np.ndarray] = None,
        heightmap: Optional[np.ndarray] = None,
        num_zones: int = 3,
        max_coverage: float = 0.30,
        buffer_px: int = 15,
        event_t1: int = 30,
        event_t2: int = 80,
        current_vec: tuple[float, float] = (0.0, 0.3),
        incident_point: Optional[tuple[int, int]] = None,
        update_bus: object | None = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.height, self.width = map_shape
        self.mission_type = mission_type
        self._roads = (
            roads_mask.astype(bool) if roads_mask is not None
            else np.zeros(map_shape, dtype=bool)
        )
        self._heightmap = (
            heightmap if heightmap is not None
            else np.zeros(map_shape, dtype=np.int32)
        )
        self.num_zones = num_zones
        self.max_coverage = max_coverage
        self.buffer_px = buffer_px
        self.event_t1 = event_t1
        self.event_t2 = event_t2
        self._current_vec = current_vec
        self._incident_point = incident_point
        self._update_bus = update_bus  # Optional UpdateBus for lifecycle events
        self._rng = rng if rng is not None else np.random.default_rng()
        self._total_cells = max(1, self.height * self.width)

        self.step_count = 0
        self._zones: list[RestrictionZone] = []
        self._peak_coverage: float = 0.0
        self._zone_violations: int = 0  # external counter (set by env)

        # Pre-compute activation schedule (staggered)
        self._activation_steps = self._stagger_activations(num_zones, event_t1, event_t2)

        # Mission dispatch
        mt = mission_type.lower().replace(" ", "_")
        if "civil" in mt:
            self._init_civil()
        elif "maritime" in mt:
            self._init_maritime()
        elif "critical" in mt or "infra" in mt:
            self._init_critical()
        else:
            self._init_civil()  # default fallback

        # Cached union mask + risk buffer
        self._cached_mask = np.zeros(map_shape, dtype=bool)
        self._cached_risk_buffer = np.zeros(map_shape, dtype=np.float32)

    # ── Backward compat ──────────────────────────────────────────────

    # These attributes are checked by InteractionEngine / UrbanEnv
    # via hasattr().  We expose them but they are read-only stubs.
    @property
    def expansion_rate(self) -> float:
        return 0.0  # no oscillation — zones grow via step()

    @expansion_rate.setter
    def expansion_rate(self, _v: float) -> None:
        pass  # ignore external mutation (InteractionEngine compat)

    @property
    def radii(self) -> np.ndarray:
        return np.zeros(self.num_zones, dtype=np.float32)

    @radii.setter
    def radii(self, _v: np.ndarray) -> None:
        pass  # ignore external mutation

    @property
    def active_zones(self) -> int:
        return sum(1 for z in self._zones if z.active)

    # ── Activation schedule ──────────────────────────────────────────

    def _stagger_activations(
        self, n: int, t1: int, t2: int,
    ) -> list[int]:
        """Spread zone activations between t1 and midpoint(t1, t2)."""
        if n <= 1:
            return [t1]
        mid = (t1 + t2) // 2
        return [int(t1 + (mid - t1) * i / (n - 1)) for i in range(n)]

    # ── Mission-specific init ────────────────────────────────────────

    def _init_civil(self) -> None:
        """Civil Protection: TFR zones derived from fire perimeter.

        Zone centres are placed at random non-building cells in the map
        interior.  During step(), zones grow based on fire_mask proximity.
        """
        for i in range(self.num_zones):
            # Place in map interior, avoiding edges and buildings
            cx = int(self._rng.integers(self.width // 5, 4 * self.width // 5))
            cy = int(self._rng.integers(self.height // 5, 4 * self.height // 5))
            # Push away from buildings
            for _ in range(20):
                if self._heightmap[cy, cx] == 0:
                    break
                cx = int(self._rng.integers(self.width // 5, 4 * self.width // 5))
                cy = int(self._rng.integers(self.height // 5, 4 * self.height // 5))
            zone = RestrictionZone(
                zone_id=f"TFR-{i + 1}",
                zone_type="tfr",
                mission_type="civil_protection",
                active=False,
                activation_step=self._activation_steps[min(i, len(self._activation_steps) - 1)],
                severity=1.0,
                label=f"TFR: Firefighting Ops",
                source_incident=f"fire_cluster_{i}",
                center=(cx, cy),
            )
            self._zones.append(zone)

    def _init_maritime(self) -> None:
        """Maritime Domain: SAR coordination box + port exclusion."""
        # SAR box: large rectangle in the middle-to-south area
        cx = self.width // 2 + int(self._rng.integers(-30, 31))
        cy = self.height // 2 + int(self._rng.integers(0, self.height // 5))
        half_w = max(30, self.width // 8)
        half_h = max(25, self.height // 10)
        sar_zone = RestrictionZone(
            zone_id="SAR-BOX-1",
            zone_type="sar_box",
            mission_type="maritime_domain",
            active=False,
            activation_step=self._activation_steps[0],
            severity=1.0,
            label="SAR COORD BOX",
            source_incident="sar_alert",
            center=(cx, cy),
            bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        )
        self._zones.append(sar_zone)

        # Port exclusion: small zone at top-left quadrant (port mouth heuristic)
        px = max(20, self.width // 6)
        py = max(20, self.height // 6)
        port_zone = RestrictionZone(
            zone_id="PORT-EXCL-1",
            zone_type="port_exclusion",
            mission_type="maritime_domain",
            active=False,
            activation_step=self._activation_steps[min(1, len(self._activation_steps) - 1)],
            severity=1.0,
            label="PORT EXCLUSION",
            source_incident="port_security",
            center=(px, py),
        )
        self._zones.append(port_zone)

    def _init_critical(self) -> None:
        """Critical Infrastructure: security cordons at tall buildings."""
        # Find tallest building locations
        if np.any(self._heightmap > 0):
            max_h = int(self._heightmap.max())
            tall_mask = self._heightmap >= max(1, max_h - 1)
            tall_ys, tall_xs = np.where(tall_mask)
            if len(tall_xs) > 0:
                # Cluster by taking N spread-out points
                n = min(self.num_zones, len(tall_xs))
                step = max(1, len(tall_xs) // n)
                indices = list(range(0, len(tall_xs), step))[:n]
                for i, idx in enumerate(indices):
                    cx, cy = int(tall_xs[idx]), int(tall_ys[idx])
                    zone = RestrictionZone(
                        zone_id=f"CORDON-{i + 1}",
                        zone_type="security_cordon",
                        mission_type="critical_infrastructure",
                        active=False,
                        activation_step=self._activation_steps[min(i, len(self._activation_steps) - 1)],
                        severity=1.0,
                        label="SECURITY CORDON",
                        source_incident=f"security_alert_{i}",
                        center=(cx, cy),
                    )
                    self._zones.append(zone)
                return

        # Fallback: use incident_point or random
        if self._incident_point is not None:
            cx, cy = self._incident_point
        else:
            cx = self.width // 2
            cy = self.height // 2
        for i in range(self.num_zones):
            ox = int(self._rng.integers(-40, 41))
            oy = int(self._rng.integers(-40, 41))
            zone = RestrictionZone(
                zone_id=f"CORDON-{i + 1}",
                zone_type="security_cordon",
                mission_type="critical_infrastructure",
                active=False,
                activation_step=self._activation_steps[min(i, len(self._activation_steps) - 1)],
                severity=1.0,
                label="SECURITY CORDON",
                source_incident=f"security_alert_{i}",
                center=(
                    int(np.clip(cx + ox, 20, self.width - 20)),
                    int(np.clip(cy + oy, 20, self.height - 20)),
                ),
            )
            self._zones.append(zone)

    # ── Step logic ───────────────────────────────────────────────────

    def step(
        self,
        dt: float = 1.0,
        fire_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Advance restriction zones by one step.

        For Civil Protection, zones track fire perimeter (dilated hull).
        For Maritime, the SAR box drifts.
        For Critical Infrastructure, cordons expand along roads.
        """
        self.step_count += 1

        for zone in self._zones:
            # Expiration check
            if zone.active and zone.expires_step is not None and self.step_count >= zone.expires_step:
                zone.active = False
                self._emit_bus_event(zone, "RESTRICTION_LIFTED")
                continue

            # Activation
            if not zone.active and self.step_count >= zone.activation_step:
                zone.active = True
                self._emit_bus_event(zone, "GEOFENCE_ACTIVATED")

            if not zone.active:
                continue

            # Track previous mask size for expansion detection
            prev_cells = int(np.sum(zone._mask)) if zone._mask is not None else 0

            # Age since activation
            age = self.step_count - zone.activation_step

            if zone.zone_type == "tfr":
                self._step_tfr(zone, age, fire_mask)
            elif zone.zone_type == "sar_box":
                self._step_sar_box(zone, age)
            elif zone.zone_type == "port_exclusion":
                self._step_port_exclusion(zone, age)
            elif zone.zone_type == "security_cordon":
                self._step_cordon(zone, age)

            # Emit expansion event if mask grew significantly
            new_cells = int(np.sum(zone._mask)) if zone._mask is not None else 0
            if new_cells > prev_cells + 20:
                self._emit_bus_event(zone, "GEOFENCE_EXPANDED")

        # Rebuild cached mask with coverage cap
        self._rebuild_mask()

    def _step_tfr(
        self, zone: RestrictionZone, age: int,
        fire_mask: Optional[np.ndarray],
    ) -> None:
        """TFR: derive zone from fire perimeter + buffer."""
        H, W = self.height, self.width
        cx, cy = zone.center

        if fire_mask is not None and np.any(fire_mask):
            # Find fire cluster nearest to this zone's anchor
            zone_mask = self._fire_cluster_tfr(fire_mask, cx, cy)
        else:
            # No fire yet: expand a small circle from the anchor point
            radius = min(self.buffer_px, 5 + age // 3)
            zone_mask = self._circle_mask(cx, cy, radius)

        zone._mask = zone_mask

    def _fire_cluster_tfr(
        self, fire_mask: np.ndarray, cx: int, cy: int,
    ) -> np.ndarray:
        """Derive TFR mask from fire cluster nearest to (cx, cy).

        Steps: label connected components of fire → pick cluster nearest
        to anchor → dilate by buffer_px → return mask.
        """
        H, W = self.height, self.width

        # Label connected components (simple 4-connected via sequential scan)
        try:
            from scipy.ndimage import label as scipy_label, binary_dilation
            labeled, n_clusters = scipy_label(fire_mask)
        except ImportError:
            # Fallback: just dilate entire fire mask
            return self._dilate_mask(fire_mask, self.buffer_px)

        if n_clusters == 0:
            return self._circle_mask(cx, cy, self.buffer_px)

        # Find nearest cluster to anchor (cx, cy)
        best_id = 1
        best_dist = float("inf")
        for cid in range(1, n_clusters + 1):
            ys, xs = np.where(labeled == cid)
            if len(xs) == 0:
                continue
            mean_x, mean_y = float(np.mean(xs)), float(np.mean(ys))
            dist = (mean_x - cx) ** 2 + (mean_y - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = cid

        cluster = labeled == best_id

        # Dilate by buffer
        try:
            struct = np.ones((3, 3), dtype=bool)
            dilated = binary_dilation(cluster, structure=struct,
                                      iterations=self.buffer_px)
            return dilated.astype(bool)
        except Exception:
            return self._dilate_mask(cluster, self.buffer_px)

    def _step_sar_box(self, zone: RestrictionZone, age: int) -> None:
        """SAR box: drift centre with deterministic current vector."""
        cx, cy = zone.center
        vx, vy = self._current_vec
        new_cx = int(np.clip(cx + vx, 10, self.width - 10))
        new_cy = int(np.clip(cy + vy, 10, self.height - 10))
        zone.center = (new_cx, new_cy)

        # Box dimensions grow slightly over time then hold
        half_w = max(30, self.width // 8) + min(age // 5, 15)
        half_h = max(25, self.height // 10) + min(age // 5, 10)
        x0 = max(0, new_cx - half_w)
        y0 = max(0, new_cy - half_h)
        x1 = min(self.width, new_cx + half_w)
        y1 = min(self.height, new_cy + half_h)
        zone.bbox = (x0, y0, x1, y1)

        # Rasterize rectangle
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        zone._mask = mask

    def _step_port_exclusion(self, zone: RestrictionZone, age: int) -> None:
        """Port exclusion: fixed position, grows slightly then holds."""
        cx, cy = zone.center
        radius = min(25, 10 + age // 4)
        zone._mask = self._circle_mask(cx, cy, radius)

    def _step_cordon(self, zone: RestrictionZone, age: int) -> None:
        """Security cordon: BFS expansion along roads from anchor."""
        cx, cy = zone.center
        # Expansion radius grows with age, cap at 60px
        reach = min(60, 8 + age // 2)
        mask = self._road_bfs_cordon(cx, cy, reach)
        zone._mask = mask

    # ── Geometry helpers ─────────────────────────────────────────────

    def _circle_mask(self, cx: int, cy: int, r: int) -> np.ndarray:
        """Rasterize a filled circle at (cx, cy) with radius r."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        r = max(1, r)
        y0, y1 = max(0, cy - r), min(self.height, cy + r + 1)
        x0, x1 = max(0, cx - r), min(self.width, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return mask
        ys = np.arange(y0, y1)
        xs = np.arange(x0, x1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        dist_sq = (yy - cy) ** 2 + (xx - cx) ** 2
        mask[yy[dist_sq <= r * r], xx[dist_sq <= r * r]] = True
        return mask

    def _dilate_mask(self, mask: np.ndarray, iters: int) -> np.ndarray:
        """Simple numpy dilation fallback (no scipy)."""
        out = mask.copy()
        for _ in range(iters):
            out = out | np.roll(out, 1, 0) | np.roll(out, -1, 0)
            out = out | np.roll(out, 1, 1) | np.roll(out, -1, 1)
        return out

    def _road_bfs_cordon(self, cx: int, cy: int, reach: int) -> np.ndarray:
        """BFS expansion from (cx, cy) along roads, then dilate.

        The cordon follows streets (road_mask=True cells) up to `reach`
        steps, then dilates by 3 pixels to form a blocky polygon.
        """
        H, W = self.height, self.width
        mask = np.zeros((H, W), dtype=bool)

        # BFS on road network
        from collections import deque
        queue: deque[tuple[int, int, int]] = deque()
        queue.append((cy, cx, 0))
        visited = np.zeros((H, W), dtype=bool)
        visited[cy, cx] = True
        mask[cy, cx] = True

        while queue:
            y, x, d = queue.popleft()
            if d >= reach:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                    visited[ny, nx] = True
                    # Expand on roads OR adjacent to road
                    if self._roads[ny, nx] or self._roads[y, x]:
                        mask[ny, nx] = True
                        queue.append((ny, nx, d + 1))

        # Dilate the road-path by 3px to form corridor width
        mask = self._dilate_mask(mask, 3)
        return mask

    # ── Mask rebuild with coverage cap ───────────────────────────────

    def _rebuild_mask(self) -> None:
        """Rebuild union mask from active zones, enforcing coverage cap."""
        H, W = self.height, self.width
        union = np.zeros((H, W), dtype=bool)

        for zone in self._zones:
            if zone.active and zone._mask is not None:
                union |= zone._mask

        coverage = float(np.sum(union)) / self._total_cells
        if coverage > self.max_coverage:
            # Scale down: disable youngest zones until under cap
            for zone in sorted(
                [z for z in self._zones if z.active and z._mask is not None],
                key=lambda z: z.activation_step,
                reverse=True,
            ):
                union &= ~zone._mask
                zone.active = False
                coverage = float(np.sum(union)) / self._total_cells
                if coverage <= self.max_coverage:
                    break

        self._cached_mask = union

        # Track peak coverage for metrics
        coverage = float(np.sum(union)) / self._total_cells
        self._peak_coverage = max(self._peak_coverage, coverage)

        # Build graded risk buffer (annulus around active zone cores)
        self._cached_risk_buffer = self._build_risk_buffer(union)

    def _build_risk_buffer(self, core_mask: np.ndarray) -> np.ndarray:
        """Build a graded [0..1] float32 annulus around the core restriction mask.

        The buffer extends ``risk_buffer_px`` pixels outside the core,
        linearly decaying from 1.0 (at core edge) to 0.0 (at buffer edge).
        """
        if not np.any(core_mask):
            return np.zeros((self.height, self.width), dtype=np.float32)

        # Determine max buffer radius from zones
        max_buf = max((z.risk_buffer_px for z in self._zones if z.active), default=10)
        if max_buf <= 0:
            return np.zeros((self.height, self.width), dtype=np.float32)

        # Dilate iteratively to compute distance-from-edge
        buf = np.zeros((self.height, self.width), dtype=np.float32)
        expanded = core_mask.copy()
        for d in range(1, max_buf + 1):
            expanded = expanded | np.roll(expanded, 1, 0) | np.roll(expanded, -1, 0)
            expanded = expanded | np.roll(expanded, 1, 1) | np.roll(expanded, -1, 1)
            ring = expanded & ~core_mask
            alpha = 1.0 - (d / (max_buf + 1))
            buf[ring & (buf == 0.0)] = alpha

        # Zero out the core itself (buffer is only outside)
        buf[core_mask] = 0.0
        return buf

    # ── UpdateBus event emission ──────────────────────────────────────

    def _emit_bus_event(self, zone: RestrictionZone, event_desc: str) -> None:
        """Emit a CONSTRAINT event to the UpdateBus if connected."""
        if self._update_bus is None:
            return
        try:
            from uavbench.updates.bus import UpdateEvent, EventType
            event = UpdateEvent(
                event_type=EventType.CONSTRAINT,
                step=self.step_count,
                description=f"{event_desc}: {zone.zone_id}",
                severity=zone.severity,
                position=zone.center,
                payload={
                    "zone_id": zone.zone_id,
                    "zone_type": zone.zone_type,
                    "lifecycle": event_desc,
                    "source_incident": zone.source_incident,
                },
                parent_id=zone.source_incident or None,
            )
            self._update_bus.publish(event)
        except ImportError:
            pass

    # ── Public API ───────────────────────────────────────────────────

    def get_zones(self) -> list[RestrictionZone]:
        """Return all zones (active and inactive) for renderer/logging."""
        return list(self._zones)

    def get_mask(self) -> np.ndarray:
        """[H, W] bool — union of active zone core masks."""
        return self._cached_mask.copy()

    def get_nfz_mask(self) -> np.ndarray:
        """Backward-compatible alias for get_mask()."""
        return self.get_mask()

    def get_risk_buffer(self) -> np.ndarray:
        """[H, W] float32 — graded annulus [0..1] around active zones."""
        return self._cached_risk_buffer.copy()

    @property
    def peak_coverage(self) -> float:
        """Peak fraction of map covered by restriction zones so far."""
        return self._peak_coverage

    @property
    def zone_violations(self) -> int:
        """Number of UAV steps inside active restriction zones."""
        return self._zone_violations

    @zone_violations.setter
    def zone_violations(self, v: int) -> None:
        self._zone_violations = v
