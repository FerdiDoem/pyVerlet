import time
import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from particle import VerletObject
from solver import Solver
import simulation


def test_collision_resolution():
    p1 = VerletObject(1, pos_x=0, pos_y=0, radius=2)
    p2 = VerletObject(2, pos_x=1, pos_y=0, radius=2)
    solver = Solver([p1, p2])
    solver.dt = 0.1
    solver.runtime = 0
    initial = np.linalg.norm(p1.position[1] - p2.position[1])
    solver.solveCollisionCummulative()
    dist = np.linalg.norm(p1.position[1] - p2.position[1])
    assert dist > initial


def test_run_simulation_bounding():
    np.random.seed(0)
    solver = simulation.setup_solver(n_particles=10, bounding_box_radius=20,
                                     time=0.2, substeps=10)
    solver.bounding_radius = 20
    results = solver.run_simulation(0.2, 10)
    positions = np.vstack(results[-1][1])[:, :2]
    dist = np.linalg.norm(positions, axis=1)
    assert np.all(dist <= 20 + 1e-6)


@pytest.mark.performance
def test_simulation_speed():
    solver = simulation.setup_solver(n_particles=100, bounding_box_radius=50,
                                     time=0.5, substeps=100)
    start = time.perf_counter()
    solver.run_simulation(0.5, 100)
    runtime = time.perf_counter() - start
    assert runtime < 1.0
