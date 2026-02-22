"""Adversarial/other-UAV near-miss risk model (non-blocking by default)."""

from __future__ import annotations

import numpy as np


class AdversarialUAVModel:
    """Moving aerial actors that create local near-miss risk bubbles."""

    def __init__(
        self,
        map_shape: tuple[int, int],
        num_uavs: int = 2,
        safety_radius: int = 6,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.height, self.width = map_shape
        self.num_uavs = max(0, int(num_uavs))
        self.safety_radius = max(1, int(safety_radius))
        self._rng = rng if rng is not None else np.random.default_rng()

        if self.num_uavs == 0:
            self._positions = np.empty((0, 2), dtype=np.float32)
            self._vel = np.empty((0, 2), dtype=np.float32)
        else:
            ys = self._rng.integers(0, self.height, size=self.num_uavs)
            xs = self._rng.integers(0, self.width, size=self.num_uavs)
            self._positions = np.stack([ys, xs], axis=1).astype(np.float32)
            angles = self._rng.uniform(0.0, 2.0 * np.pi, size=self.num_uavs)
            self._vel = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)

    def step(self, dt: float = 1.0) -> None:
        if self.num_uavs == 0:
            return

        jitter = self._rng.normal(0.0, 0.12, size=self._vel.shape).astype(np.float32)
        self._vel = 0.90 * self._vel + 0.10 * jitter

        self._positions += self._vel * float(dt)
        self._positions[:, 0] = np.clip(self._positions[:, 0], 0, self.height - 1)
        self._positions[:, 1] = np.clip(self._positions[:, 1], 0, self.width - 1)

        # Reflect when touching boundaries.
        hit_top_bottom = (self._positions[:, 0] <= 0.0) | (self._positions[:, 0] >= self.height - 1)
        hit_left_right = (self._positions[:, 1] <= 0.0) | (self._positions[:, 1] >= self.width - 1)
        self._vel[hit_top_bottom, 0] *= -1.0
        self._vel[hit_left_right, 1] *= -1.0

    @property
    def positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_risk_map(self, shape: tuple[int, int]) -> np.ndarray:
        """Return near-miss risk field in [0, 1]."""
        H, W = shape
        risk = np.zeros((H, W), dtype=np.float32)
        if self.num_uavs == 0:
            return risk

        r = self.safety_radius
        r_inv = 1.0 / max(r, 1)
        for py, px in self._positions:
            iy = int(py)
            ix = int(px)
            y_lo = max(0, iy - r)
            y_hi = min(H, iy + r + 1)
            x_lo = max(0, ix - r)
            x_hi = min(W, ix + r + 1)
            for ny in range(y_lo, y_hi):
                for nx in range(x_lo, x_hi):
                    d = abs(ny - iy) + abs(nx - ix)
                    if d <= r:
                        v = float(1.0 - d * r_inv)
                        if v > risk[ny, nx]:
                            risk[ny, nx] = v
        return np.clip(risk, 0.0, 1.0)
