import numpy as np
from core import Solver
from generator import Generator
from particle import VerletObject


class SolverAdapter:
    """Non-blocking wrapper around :class:`Solver`."""

    def __init__(self, particles: int = 10, radius: float = 50.0, dt: float = 1/60, colour_mode: str = "velocity"):
        self.particles = particles
        self.radius = radius
        self.dt = dt
        self.colour_mode = colour_mode
        self.solver = self._build_solver()

    def _build_solver(self) -> Solver:
        gen = Generator(VerletObject)
        parts = list(gen.rnd_particle_gen(self.particles, self.radius))
        if parts:
            parts[0].fixated = True
        solver = Solver(parts)
        solver.dt = self.dt
        solver.runtime = 0.0
        return solver

    def reset(self):
        self.solver = self._build_solver()

    def set_params(self, particles: int | None = None, radius: float | None = None,
                   dt: float | None = None, colour_mode: str | None = None):
        if particles is not None:
            self.particles = particles
        if radius is not None:
            self.radius = radius
        if dt is not None:
            self.dt = dt
        if colour_mode is not None:
            self.colour_mode = colour_mode
        self.reset()

    def step(self):
        self.solver.update()
        self.solver.runtime += self.solver.dt

    def buffer(self) -> bytes:
        pos = self.solver.positions[:, 1]
        rad = self.solver.radii
        vel = self.solver.velocities[:, 1]
        data = np.concatenate([pos, rad[:, None], vel], axis=1).astype(np.float32)
        return data.tobytes()
