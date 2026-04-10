"""
N-Body Gravitational Simulation.

Uses a direct O(N²) summation with a softening length ε to prevent
singularities when bodies pass close to each other.  For N ≤ ~20 this
is faster than a Barnes-Hut tree and trivially vectorisable with NumPy.

  F_ij = G * m_i * m_j / (|r_ij|² + ε²)  ·  r̂_ij
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class GravityConfig:
    n_bodies:   int   = 5
    G:          float = 1.0      # gravitational constant (simulation units)
    softening:  float = 10.0     # ε — prevents singularity
    trail_len:  int   = 60       # history steps kept per body
    dt:         float = 1.0      # time step


class GravitySystem:
    """
    Direct-summation N-body integrator (Leapfrog / Störmer-Verlet scheme).
    """

    def __init__(self, width: int, height: int, cfg: GravityConfig | None = None):
        self.width  = width
        self.height = height
        self.cfg    = cfg or GravityConfig()
        self._init_state()

    # ------------------------------------------------------------------
    def _init_state(self):
        n    = self.cfg.n_bodies
        cx   = self.width  / 2
        cy   = self.height / 2
        rmax = min(self.width, self.height) * 0.35

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radii  = rmax * (0.4 + 0.6 * np.random.rand(n))

        self.pos    = np.column_stack([
            cx + radii * np.cos(angles),
            cy + radii * np.sin(angles),
        ]).astype(np.float64)

        self.mass   = 5.0 + np.random.uniform(0, 25, n)

        # Approximate circular orbit velocity
        orbit_speed = np.sqrt(self.cfg.G * self.mass.sum() / (radii + 1e-9)) * 0.5
        self.vel    = np.column_stack([
            -np.sin(angles) * orbit_speed,
             np.cos(angles) * orbit_speed,
        ])

        self.accel  = np.zeros_like(self.pos)
        self.trails: list[list[tuple]] = [[] for _ in range(n)]

    # ------------------------------------------------------------------
    def _compute_accel(self) -> np.ndarray:
        """Vectorised O(N²) gravitational acceleration."""
        G, eps2 = self.cfg.G, self.cfg.softening ** 2
        # diff[i,j] = pos[j] - pos[i]
        diff  = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :]  # (N,N,2)
        dist2 = (diff ** 2).sum(axis=2) + eps2                            # (N,N)
        dist3 = dist2 ** 1.5

        # Acceleration from each pair: a_i += G * m_j * r_ij / |r_ij|³
        coeff = G * self.mass[np.newaxis, :] / dist3                      # (N,N)
        np.fill_diagonal(coeff, 0)
        return (coeff[:, :, np.newaxis] * diff).sum(axis=1)               # (N,2)

    # ------------------------------------------------------------------
    def step(self, mouse_pos: tuple[float, float] | None = None) -> None:
        dt = self.cfg.dt

        # Leapfrog half-kick → drift → recompute → half-kick
        self.vel  += 0.5 * self.accel * dt
        self.pos  += self.vel * dt
        self.accel = self._compute_accel()

        # Mouse pull
        if mouse_pos is not None:
            delta  = np.array(mouse_pos) - self.pos
            d      = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1)
            self.accel += delta / d * 2.0

        self.vel += 0.5 * self.accel * dt

        # Wrap boundaries
        self.pos[:, 0] %= self.width
        self.pos[:, 1] %= self.height

        # Update trails
        for i, trail in enumerate(self.trails):
            trail.append((float(self.pos[i, 0]), float(self.pos[i, 1])))
            if len(trail) > self.cfg.trail_len:
                trail.pop(0)

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        speeds = np.linalg.norm(self.vel, axis=1)
        ke     = 0.5 * (self.mass * speeds ** 2).sum()
        return {
            "count":     int(self.cfg.n_bodies),
            "avg_speed": float(np.mean(speeds)),
            "kinetic_e": float(ke),
            "max_speed": float(np.max(speeds)),
        }

    # ------------------------------------------------------------------
    def state_json(self) -> dict:
        return {
            "positions": self.pos.tolist(),
            "masses":    self.mass.tolist(),
            "trails":    self.trails,
        }

    # ------------------------------------------------------------------
    def apply_config(self, updates: dict) -> None:
        for k, v in updates.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, type(getattr(self.cfg, k))(v))
        self._init_state()
