#!/usr/bin/env python
"""Parameter sweep harness for FPS tuning."""

from __future__ import annotations

import argparse
import itertools
import time

import simulation
from grid import HashGrid


def run_case(n_particles: int, substeps: int, cell_size: float, sim_time: float) -> float:
    """Return frames per second for a single parameter combination."""
    original_build = HashGrid.build

    def build(self, points, _):
        return original_build(self, points, cell_size)

    HashGrid.build = build

    solver = simulation.setup_solver(n_particles=n_particles,
                                     bounding_box_radius=50,
                                     time=sim_time,
                                     substeps=substeps)

    start = time.perf_counter()
    for _ in solver.run_simulation_iter(sim_time, substeps):
        pass
    elapsed = time.perf_counter() - start

    HashGrid.build = original_build

    if elapsed == 0:
        return 0.0
    return substeps / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an FPS parameter sweep")
    parser.add_argument("--particles", nargs="+", type=int,
                        default=[50, 100, 200])
    parser.add_argument("--substeps", nargs="+", type=int,
                        default=[100, 500, 1000])
    parser.add_argument("--cell_sizes", nargs="+", type=float,
                        default=[1.0, 2.0, 3.0])
    parser.add_argument("--time", type=float, default=1.0,
                        help="Simulated time per run")
    args = parser.parse_args()

    header = ["n_particles", "substeps", "cell_size", "fps"]
    print("\t".join(header))
    for n, s, c in itertools.product(args.particles, args.substeps, args.cell_sizes):
        fps = run_case(n, s, c, args.time)
        print(f"{n}\t{s}\t{c}\t{fps:.2f}")


if __name__ == "__main__":
    main()
