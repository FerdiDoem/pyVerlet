# -*- coding: utf-8 -*-
"""Utility helpers for building and running particle simulations."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from particle import VerletObject
from generator import Generator
from solver import Solver


def setup_solver(n_particles: int = 10,
                 bounding_box_radius: float = 50,
                 time: float = 5,
                 substeps: float = 800.0) -> Solver:
    """Create particles and return a configured solver."""
    v0 = np.array([0.0, 0.0]) * time / substeps

    gen1 = Generator(VerletObject)
    particles = list(gen1.rnd_particle_gen(n_particles,
                                           bounding_box_radius,
                                           v0=v0))

    gen2 = Generator(VerletObject)
    particles.extend(gen2.rnd_particle_gen(n_particles,
                                           bounding_box_radius,
                                           v0=v0,
                                           start_ID=n_particles + 1))

    gen1.chain_particles()

    if particles:
        particles[0].fixated = True

    return Solver(list(particles))


def animate_simulation(results, bounding_box_radius: float,
                       time: float, substeps: float):
    """Return a matplotlib animation for the given results."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_facecolor("darkgray")
    circle = plt.Circle((0, 0), bounding_box_radius,
                        edgecolor="black",
                        facecolor="white",
                        fill=True)
    ax.add_patch(circle)
    fig.canvas.draw()
    px_per_scale = (ax.get_window_extent().width /
                    (2 * bounding_box_radius + 2) * 72.0 / fig.dpi)

    frames = []
    kinetic = []
    for step_data in results:
        particles = np.vstack(step_data[1])
        frame = ax.scatter(particles[:, 0], particles[:, 1],
                           c=np.linalg.norm(particles[:, 6:8], axis=1),
                           cmap="gist_rainbow",
                           edgecolors="white",
                           s=(px_per_scale * 2 * particles[:, 7]) ** 2,
                           linewidth=0)
        energy = sum(0.5 * particles[:, 7] *
                     np.linalg.norm(particles[:, 2:4], axis=1) ** 2)
        frames.append([frame])
        kinetic.append(energy)

    interval = time * 1000 // substeps
    animation = anim.ArtistAnimation(fig, frames, blit=True,
                                     interval=interval, repeat=True)
    return animation, kinetic


def plot_kinetic_energy(kinetic):
    """Plot the pseudo kinetic energy over time."""
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, len(kinetic), 1), kinetic, label="System")
    ax.set_xlabel("substeps")
    ax.set_ylabel("$equv. E_{Kin}$")
    plt.legend()
    plt.close(fig)
    return fig
