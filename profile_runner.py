#!/usr/bin/env python
"""Run the particle simulation under cProfile and save the results."""

import argparse
import cProfile
import time

import simulation
from grid import HashGrid


def run_profile(n_particles: int, substeps: int, cell_size: float, sim_time: float) -> None:
    """Run the simulation with profiling enabled."""
    # Patch HashGrid.build to use a fixed cell size
    original_build = HashGrid.build

    def build(self, points, _):
        return original_build(self, points, cell_size)

    HashGrid.build = build

    solver = simulation.setup_solver(n_particles=n_particles,
                                     bounding_box_radius=50,
                                     time=sim_time,
                                     substeps=substeps)

    for _ in solver.run_simulation_iter(sim_time, substeps):
        pass

    HashGrid.build = original_build


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the particle simulation")
    parser.add_argument("--n_particles", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=1000)
    parser.add_argument("--cell_size", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=1.0, help="Simulated time")
    parser.add_argument("--output", type=str, default="stats.prof",
                        help="Filename for the profiling output")
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    run_profile(args.n_particles, args.substeps, args.cell_size, args.time)
    profiler.disable()
    profiler.dump_stats(args.output)


if __name__ == "__main__":
    main()
