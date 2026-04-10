"""
Boids Flocking Simulation (Craig Reynolds, 1987).

Three steering behaviours implemented with NumPy distance matrices:
  1. Cohesion   — steer toward average position of neighbours
  2. Separation — avoid crowding neighbours
  3. Alignment  — steer toward average heading of neighbours
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FlockConfig:
    count:      int   = 150
    cohesion:   float = 0.5
    separation: float = 1.5
    alignment:  float = 1.0
    perception: float = 50.0
    max_speed:  float = 4.0


class FlockSystem:
    """
    Vectorised Boids simulation.

    Distance matrix is computed once per step (O(N²) memory) and reused
    by all three rules, avoiding redundant sqrt calls.
    """

    def __init__(self, width: int, height: int, cfg: FlockConfig | None = None):
        self.width  = width
        self.height = height
        self.cfg    = cfg or FlockConfig()
        self._init_state()

    # ------------------------------------------------------------------
    def _init_state(self):
        n = self.cfg.count
        self.pos = np.random.uniform(
            [0, 0], [self.width, self.height], (n, 2)
        ).astype(np.float64)
        self.vel = np.random.uniform(-2, 2, (n, 2))

    # ------------------------------------------------------------------
    def _distance_matrix(self) -> np.ndarray:
        """Return (N, N) pairwise Euclidean distance matrix."""
        diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2)   # (N, N)

    # ------------------------------------------------------------------
    def step(self, mouse_pos: tuple[float, float] | None = None) -> None:
        cfg  = self.cfg
        n    = cfg.count
        dist = self._distance_matrix()   # (N, N)

        # Boolean neighbour mask within perception radius (exclude self)
        np.fill_diagonal(dist, np.inf)
        neighbour = dist < cfg.perception
        close     = dist < cfg.perception * 0.3

        accel = np.zeros_like(self.vel)

        # --- Cohesion ---
        n_count = neighbour.sum(axis=1, keepdims=True).clip(min=1)
        avg_pos = (neighbour[:, :, np.newaxis] * self.pos[np.newaxis]).sum(axis=1) / n_count
        accel  += (avg_pos - self.pos) * 0.001 * cfg.cohesion

        # --- Alignment ---
        avg_vel = (neighbour[:, :, np.newaxis] * self.vel[np.newaxis]).sum(axis=1) / n_count
        accel  += (avg_vel - self.vel) * 0.01 * cfg.alignment

        # --- Separation ---
        safe_dist = np.where(dist > 0, dist, 1.0)
        diff      = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]   # (N,N,2)
        sep_vec   = (diff / safe_dist[:, :, np.newaxis])                       # unit vectors
        sep_force = (close[:, :, np.newaxis] * sep_vec).sum(axis=1)
        accel    += sep_force * 0.05 * cfg.separation

        # --- Mouse guidance ---
        if mouse_pos is not None:
            delta = np.array(mouse_pos) - self.pos
            d     = np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1)
            mask  = d[:, 0] < 150
            accel[mask] += (delta / d)[mask] * 0.5

        # Integrate
        self.vel += accel
        speed     = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel  = np.where(
            speed > cfg.max_speed,
            self.vel / speed * cfg.max_speed,
            self.vel
        )
        self.pos += self.vel

        # Wrap boundaries (toroidal)
        self.pos[:, 0] %= self.width
        self.pos[:, 1] %= self.height

    # ------------------------------------------------------------------
    def metrics(self) -> dict:
        speeds = np.linalg.norm(self.vel, axis=1)
        return {
            "count":     int(self.cfg.count),
            "avg_speed": float(np.mean(speeds)),
            "kinetic_e": float(0.5 * np.sum(speeds ** 2)),
            "order":     float(np.linalg.norm(self.vel.mean(axis=0)) / (self.cfg.max_speed + 1e-9)),
        }

    # ------------------------------------------------------------------
    def state_json(self) -> dict:
        headings = np.arctan2(self.vel[:, 1], self.vel[:, 0])
        speeds   = np.linalg.norm(self.vel, axis=1)
        return {
            "positions": self.pos.tolist(),
            "headings":  headings.tolist(),
            "speeds":    speeds.tolist(),
        }

    # ------------------------------------------------------------------
    def apply_config(self, updates: dict) -> None:
        for k, v in updates.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, type(getattr(self.cfg, k))(v))
        self._init_state()
