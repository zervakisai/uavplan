"""Population density risk model for operationally realistic 2D scenarios.

This layer is non-blocking by design. It produces a risk cost map in [0, 1]
that can be consumed by risk-aware planners.
"""

from __future__ import annotations

import numpy as np


class PopulationRiskModel:
    """Crowd/population risk model with deterministic diffusion + hazard drift."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        base_risk: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.height, self.width = map_shape
        self._rng = rng if rng is not None else np.random.default_rng()

        if base_risk is not None:
            risk = np.asarray(base_risk, dtype=np.float32)
            if risk.shape != (self.height, self.width):
                raise ValueError("base_risk shape must match map_shape")
            rmin = float(np.min(risk))
            rmax = float(np.max(risk))
            if rmax > rmin:
                self._risk = (risk - rmin) / (rmax - rmin)
            else:
                self._risk = np.zeros_like(risk, dtype=np.float32)
        else:
            self._risk = np.zeros((self.height, self.width), dtype=np.float32)

    def step(
        self,
        fire_mask: np.ndarray | None = None,
        traffic_positions: np.ndarray | None = None,
    ) -> None:
        """Advance one simulation tick.

        Fire and traffic increase local crowd risk. The map diffuses with light
        temporal persistence to avoid abrupt oscillations.
        """
        source = np.zeros((self.height, self.width), dtype=np.float32)

        if fire_mask is not None:
            fire = fire_mask.astype(np.float32)
            # Fire-adjacent crowd risk rises around the front.
            source += 0.6 * (
                fire
                + np.roll(fire, 1, axis=0)
                + np.roll(fire, -1, axis=0)
                + np.roll(fire, 1, axis=1)
                + np.roll(fire, -1, axis=1)
            )

        if traffic_positions is not None and len(traffic_positions) > 0:
            for py, px in traffic_positions:
                iy = int(py)
                ix = int(px)
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width:
                            if abs(dy) + abs(dx) <= 3:
                                source[ny, nx] += 0.2

        # Diffusion (5-point stencil) + persistence.
        nbr = (
            np.roll(self._risk, 1, axis=0)
            + np.roll(self._risk, -1, axis=0)
            + np.roll(self._risk, 1, axis=1)
            + np.roll(self._risk, -1, axis=1)
        ) / 4.0
        updated = 0.70 * self._risk + 0.20 * nbr + 0.10 * source
        self._risk = np.clip(updated, 0.0, 1.0).astype(np.float32, copy=False)

    @property
    def risk_map(self) -> np.ndarray:
        """Return current crowd/population risk map in [0, 1]."""
        return self._risk.copy()
