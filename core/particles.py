"""
Particle System — vectorised with NumPy.
All state lives in contiguous arrays for cache efficiency.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ParticleConfig:
    count:   int   = 300
    speed:   float = 2.0
    size:    float = 2.5
    decay:   float = 0.96
    gravity: float = 0.05


class ParticleSystem:
    """
    Vectorised 2-D particle simulation.

    Positions, velocities and metadata are stored as (N,) or (N,2)
    NumPy arrays so that every physics step runs as a single BLAS call
    with no Python-level loops.
    """

    def __init__(self, width: int, height: int, cfg: ParticleConfig | None = None):
        self.width  = width
        self.height = height
        self.cfg    = cfg or ParticleConfig()
        self._init_state()

    # ------------------------------------------------------------------
    def _init_state(self):
        n = self.cfg.count
        angles = np.random.uniform(0, 2 * np.pi, n)
        speeds = np.random.uniform(0.5, 1.5, n) * self.cfg.speed

        self.pos = np.random.uniform(
            [0, 0], [self.width, self.height], (n, 2)
        ).astype(np.float64)

        self.vel = np.stack(
            [np.cos(angles) * speeds, np.sin(angles) * speeds], axis=1
        )

        self.hue = np.random.uniform(0, 360, n)

    # ------------------------------------------------------------------
    def step(self, mouse_pos: tuple[float, float] | None = None,
             attract: bool = True) -> None:
        """Advance simulation by one tick."""
        cfg = self.cfg

        # Gravity
        self.vel[:, 1] += cfg.gravity

        # Velocity decay (air resistance / damping)
        self.vel *= cfg.decay

        # Mouse interaction
        if mouse_pos is not None:
            mx, my = mouse_pos
            diff   = np.array([mx, my]) - self.pos          # (N,2)
            dist   = np.linalg.norm(diff, axis=1, keepdims=True)  # (N,1)
            mask   = (dist[:, 0] < 120) & (dist[:, 0] > 0.1)
            safe   = np.where(dist > 0.1, dist, 1.0)
            force  = (1.0 if attract else -1.0) * 0.8 / safe
            self.vel[mask] += (diff * force)[mask]

        # Integrate
        self.pos += self.vel

        # Boundary bounce
        for axis, limit in enumerate([self.width, self.height]):
            lo = self.pos[:, axis] < 0
            hi = self.pos[:, axis] > limit
            self.pos[lo, axis] = 0
            self.pos[hi, axis] = limit
            self.vel[lo | hi, axis] *= -0.8

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        speeds  = np.linalg.norm(self.vel, axis=1)
        return {
            "count":     int(self.cfg.count),
            "avg_speed": float(np.mean(speeds)),
            "kinetic_e": float(0.5 * np.sum(speeds ** 2)),
            "max_speed": float(np.max(speeds)),
        }

    # ------------------------------------------------------------------
    def state_json(self) -> dict:
        return {
            "positions":  self.pos.tolist(),
            "velocities": self.vel.tolist(),
            "hues":       self.hue.tolist(),
            "size":       self.cfg.size,
        }

    # ------------------------------------------------------------------
    def apply_config(self, updates: dict) -> None:
        for k, v in updates.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, type(getattr(self.cfg, k))(v))
        self._init_state()
