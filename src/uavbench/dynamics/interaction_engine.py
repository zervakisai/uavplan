"""Causal interaction rules between fire, NFZ, traffic and population risk."""

from __future__ import annotations

import numpy as np


class InteractionEngine:
    """Deterministic interaction graph for operational 2D dynamics."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        roads_mask: np.ndarray | None = None,
        nfz_alpha: float = 0.002,
        coupling_strength: float = 1.0,
    ) -> None:
        self.height, self.width = map_shape
        self._roads = roads_mask.astype(bool) if roads_mask is not None else np.zeros(map_shape, dtype=bool)
        self._nfz_alpha = float(max(0.0, nfz_alpha))
        self._coupling = float(max(0.1, coupling_strength))
        self._traffic_closure_mask = np.zeros(map_shape, dtype=bool)
        self._downstream_congestion_mask = np.zeros(map_shape, dtype=np.float32)
        self._base_nfz_expansion_rate: float | None = None
        self._fire_traffic_feedback_events: int = 0
        self._fire_traffic_closure_adjustments: int = 0
        self._downstream_congestion_amplifications: int = 0

    @property
    def traffic_closure_mask(self) -> np.ndarray:
        return self._traffic_closure_mask.copy()

    def _apply_manhattan_disk(
        self, mask: np.ndarray, positions: np.ndarray, radius: int,
        roads_only: bool = False,
    ) -> None:
        """Set cells within Manhattan radius of positions on mask.

        Vectorized helper to replace per-position nested Python loops.
        """
        H, W = self.height, self.width
        for py, px in positions:
            iy, ix = int(py), int(px)
            y_lo = max(0, iy - radius)
            y_hi = min(H, iy + radius + 1)
            x_lo = max(0, ix - radius)
            x_hi = min(W, ix + radius + 1)
            for ny in range(y_lo, y_hi):
                for nx in range(x_lo, x_hi):
                    if abs(ny - iy) + abs(nx - ix) <= radius:
                        if not roads_only or self._roads[ny, nx]:
                            mask[ny, nx] = True

    def update(
        self,
        *,
        step_idx: int,
        fire_mask: np.ndarray | None,
        traffic_positions: np.ndarray | None,
        dynamic_nfz: object | None,
        risk_map: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Apply causal interactions and return summary metrics."""
        fire = fire_mask.astype(bool) if fire_mask is not None else np.zeros((self.height, self.width), dtype=bool)
        fire_cells = int(np.sum(fire))
        fire_fraction = float(fire_cells / max(1, self.height * self.width))

        # Read restriction zone mask
        nfz_mask = np.zeros((self.height, self.width), dtype=bool)
        if dynamic_nfz is not None and hasattr(dynamic_nfz, "get_nfz_mask"):
            nfz_mask = np.asarray(dynamic_nfz.get_nfz_mask(), dtype=bool)

        # Fire + traffic -> road closures.
        self._traffic_closure_mask.fill(False)
        hot_zone = np.zeros((self.height, self.width), dtype=bool)
        if np.any(self._roads):
            if fire_cells > 0:
                hot_zone = fire.copy()
                dilation_steps = int(np.clip(round(self._coupling * 2.0), 1, 4))
                for _ in range(dilation_steps):
                    hot_zone |= np.roll(hot_zone, 1, axis=0)
                    hot_zone |= np.roll(hot_zone, -1, axis=0)
                    hot_zone |= np.roll(hot_zone, 1, axis=1)
                    hot_zone |= np.roll(hot_zone, -1, axis=1)
                self._traffic_closure_mask |= (hot_zone & self._roads)

            if traffic_positions is not None and len(traffic_positions) > 0:
                tr = int(np.clip(round(2.0 * self._coupling), 1, 5))
                self._apply_manhattan_disk(
                    self._traffic_closure_mask, traffic_positions, tr,
                    roads_only=True,
                )

        closure_count = int(np.sum(self._traffic_closure_mask))

        # Bidirectional fire-traffic feedback
        fire_traffic_feedback = 0
        if fire_cells > 0 and traffic_positions is not None and len(traffic_positions) > 0:
            # Vectorized: check which vehicles are near fire (within 2 cells)
            for py, px in traffic_positions:
                iy, ix = int(py), int(px)
                y_lo = max(0, iy - 2)
                y_hi = min(self.height, iy + 3)
                x_lo = max(0, ix - 2)
                x_hi = min(self.width, ix + 3)
                vehicle_near_fire = bool(np.any(fire[y_lo:y_hi, x_lo:x_hi]))

                if vehicle_near_fire:
                    fire_traffic_feedback += 1
                    self._fire_traffic_closure_adjustments += 1

                    # Close roads near fire-adjacent vehicles
                    r3 = 3
                    y_lo3 = max(0, iy - r3)
                    y_hi3 = min(self.height, iy + r3 + 1)
                    x_lo3 = max(0, ix - r3)
                    x_hi3 = min(self.width, ix + r3 + 1)
                    for ny in range(y_lo3, y_hi3):
                        for nx in range(x_lo3, x_hi3):
                            if abs(ny - iy) + abs(nx - ix) <= r3 and self._roads[ny, nx]:
                                self._traffic_closure_mask[ny, nx] = True

                    # Downstream congestion
                    congestion_radius = int(np.clip(round(3.0 * self._coupling), 2, 6))
                    y_loc = max(0, iy - congestion_radius)
                    y_hic = min(self.height, iy + congestion_radius + 1)
                    x_loc = max(0, ix - congestion_radius)
                    x_hic = min(self.width, ix + congestion_radius + 1)
                    for ny in range(y_loc, y_hic):
                        for nx in range(x_loc, x_hic):
                            if abs(ny - iy) + abs(nx - ix) <= congestion_radius:
                                self._downstream_congestion_mask[ny, nx] += 0.2 * self._coupling
                                self._downstream_congestion_amplifications += 1

        self._fire_traffic_feedback_events += fire_traffic_feedback
        np.clip(self._downstream_congestion_mask, 0.0, 1.0, out=self._downstream_congestion_mask)

        hot_road = hot_zone & self._roads
        hot_road_count = int(np.sum(hot_road))
        fire_road_closure_rate = (
            float(np.sum(self._traffic_closure_mask & hot_road)) / float(hot_road_count)
            if hot_road_count > 0
            else 0.0
        )

        nfz_overlap = int(np.sum(nfz_mask & fire))
        fire_nfz_overlap_ratio = float(nfz_overlap / fire_cells) if fire_cells > 0 else 0.0

        # Congestion map — vectorized with bounded slicing
        congestion_map = np.zeros((self.height, self.width), dtype=np.float32)
        if traffic_positions is not None and len(traffic_positions) > 0:
            for py, px in traffic_positions:
                iy, ix = int(py), int(px)
                y_lo = max(0, iy - 3)
                y_hi = min(self.height, iy + 4)
                x_lo = max(0, ix - 3)
                x_hi = min(self.width, ix + 4)
                for ny in range(y_lo, y_hi):
                    for nx in range(x_lo, x_hi):
                        if abs(ny - iy) + abs(nx - ix) <= 3:
                            congestion_map[ny, nx] += 1.0
        congestion_cells = int(np.sum(congestion_map > 0.0))

        congestion_risk_corr = 0.0
        if risk_map is not None:
            rm = np.asarray(risk_map, dtype=np.float32)
            if rm.shape == congestion_map.shape:
                c = congestion_map.ravel()
                r = rm.ravel()
                if float(np.std(c)) > 1e-6 and float(np.std(r)) > 1e-6:
                    corr = np.corrcoef(c, r)
                    congestion_risk_corr = float(np.nan_to_num(corr[0, 1], nan=0.0))

        nfz_fraction = float(np.sum(nfz_mask) / max(1, self.height * self.width))
        closure_fraction = float(closure_count / max(1, self.height * self.width))
        weights = np.array([fire_fraction, nfz_fraction, closure_fraction], dtype=np.float64)
        dynamic_block_entropy = 0.0
        if float(np.sum(weights)) > 0.0:
            probs = weights / float(np.sum(weights))
            probs = probs[probs > 0.0]
            dynamic_block_entropy = float(-np.sum(probs * np.log(probs)) / np.log(3.0))

        return {
            "step_idx": float(step_idx),
            "fire_cells": float(fire_cells),
            "fire_fraction": fire_fraction,
            "traffic_closure_cells": float(closure_count),
            "interaction_fire_nfz_overlap_ratio": float(np.clip(fire_nfz_overlap_ratio, 0.0, 1.0)),
            "interaction_fire_road_closure_rate": float(np.clip(fire_road_closure_rate, 0.0, 1.0)),
            "interaction_congestion_risk_corr": float(np.clip(congestion_risk_corr, -1.0, 1.0)),
            "dynamic_block_entropy": float(np.clip(dynamic_block_entropy, 0.0, 1.0)),
            "traffic_congestion_cells": float(congestion_cells),
            "nfz_cells": float(np.sum(nfz_mask)),
            "fire_traffic_feedback_events": float(self._fire_traffic_feedback_events),
            "fire_traffic_feedback_rate": float(fire_traffic_feedback / max(1, len(traffic_positions) if traffic_positions is not None else 1)),
            "fire_traffic_closure_adjustments": float(self._fire_traffic_closure_adjustments),
            "downstream_congestion_amplifications": float(self._downstream_congestion_amplifications),
            "downstream_congestion_intensity": float(np.mean(self._downstream_congestion_mask)),
            "congestion_amplification_rate": float(closure_count / max(1, int(np.sum(self._roads)))) if int(np.sum(self._roads)) > 0 else 0.0,
        }
