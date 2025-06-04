# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:50:35 2022

@author: Ferdinand Dömling


"""
import time
import numpy as np

from simulation import (
    setup_solver,
    animate_simulation,
    plot_kinetic_energy,
)


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def main():
    """Run a demo simulation and return the results."""
    sim_time = 5
    substeps = 800.0
    n_particles = 10
    bounding_box_radius = 50

    solver = setup_solver(n_particles, bounding_box_radius,
                          sim_time, substeps)
    results = solver.run_simulation(sim_time, substeps)

    animation, kinetic = animate_simulation(results, bounding_box_radius,
                                            sim_time, substeps)
    plot_kinetic_energy(kinetic)

    return results, animation


if __name__ == '__main__':
    start = time.time()
    _, _ = main()
    end = time.time()
    print(f'Done in {round(end-start, 2)} s!')
