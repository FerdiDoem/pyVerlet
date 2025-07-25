# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:25:06 2022

@author: Ferdinand Dömling

A simple physics simulation inspired by https://www.youtube.com/c/PezzzasWork

"""

import numpy as np
from tqdm import tqdm
from typing import Callable
from scipy import constants
from scipy.spatial import cKDTree
from grid import HashGrid
from buffers import init_buffers
import math
"""
Unit System:
    m,s,N,Kg
"""

GRAVITY = -constants.g
BBOX_CENTER_X = 0.
BBOX_CENTER_Y = 0.
BBOX_ROUND_RADIUS = 50.
CONST_FORCE_VEC_X = 0.
CONST_FORCE_VEC_Y = 0.
ACCELERATION_DURATION = 100


class Solver:

    def __init__(self, particles: list, use_grid: bool = True):
        self.gravity = np.array([0, GRAVITY])
        self.force = np.array([CONST_FORCE_VEC_X, CONST_FORCE_VEC_Y])
        self.particles = particles
        self.use_grid = use_grid
        self.bounding_center = np.array([BBOX_CENTER_X, BBOX_CENTER_Y])
        self.bounding_radius: float = BBOX_ROUND_RADIUS
        self.runtime: float = 0.0
        self.dt: float = 0.0
        self.results = []

        # initialize buffers holding particle data and work arrays
        self.buffers = init_buffers(self.particles)
        self.positions = self.buffers["positions"]
        self.velocities = self.buffers["velocities"]
        self.accelerations = self.buffers["accelerations"]
        self.radii = self.buffers["radii"]
        self.masses = self.buffers["masses"]
        self.fixated_mask = self.buffers["fixated_mask"]
        self.work_vec2 = self.buffers["work_vec2"]
        self.work_vec2b = self.buffers["work_vec2b"]
        self.work_scalar = self.buffers["work_scalar"]
        self.work_scalar_b = self.buffers["work_scalar_b"]
        self.accum = self.buffers["accum"]

        # pre-compute linkage pairs
        self.linkage_pairs = self._compute_linkage_pairs()

    def _compute_linkage_pairs(self):
        """Return a list of particle index pairs that are linked."""
        id_to_idx = {p.ID: idx for idx, p in enumerate(self.particles)}
        pairs = set()
        for idx, p in enumerate(self.particles):
            for link_id in p.linkage:
                j = id_to_idx.get(link_id)
                if j is None:
                    continue
                if idx == j:
                    continue
                pair = tuple(sorted((idx, j)))
                pairs.add(pair)
        if not pairs:
            return []
        return np.array(sorted(pairs), dtype=int)

    def _sync_arrays(self):
        """Load particle attributes into numpy arrays for fast updates."""
        if not self.particles:
            self.positions = np.empty((0, 2, 2))
            self.velocities = np.empty((0, 2, 2))
            self.accelerations = np.empty((0, 2))
            self.radii = np.empty((0,))
            self.masses = np.empty((0,))
            self.fixated_mask = np.empty((0,), dtype=bool)
            return
        self.positions = np.array([p.position for p in self.particles], dtype=float)
        self.velocities = np.array([p.velocity for p in self.particles], dtype=float)
        self.accelerations = np.array([p.acceleration for p in self.particles], dtype=float)
        self.radii = np.array([p.radius for p in self.particles], dtype=float)
        self.masses = np.array([p.mass for p in self.particles], dtype=float)
        self.fixated_mask = np.array([p.fixated for p in self.particles], dtype=bool)

    def _sync_particles(self):
        """Write numpy array state back to the particle objects."""
        for idx, p in enumerate(self.particles):
            p.position = self.positions[idx]
            p.velocity = self.velocities[idx]
            p.acceleration = self.accelerations[idx]
            p.radius = self.radii[idx]
            p.mass = self.masses[idx]
            p.fixated = bool(self.fixated_mask[idx])

    def update(self):
        """Update particle states using stored numpy arrays."""

        if not self.particles:
            return

        fixated_mask = self.fixated_mask
        stored_prev = self.positions[fixated_mask, 0].copy()
        stored_cur = self.positions[fixated_mask, 1].copy()

        # v = pos_cur - pos_prev
        np.subtract(self.positions[:, 1], self.positions[:, 0],
                    out=self.velocities[:, 1])
        velocity = self.velocities[:, 1]

        # reset and apply acceleration
        self.accelerations.fill(0.0)
        if self.runtime <= ACCELERATION_DURATION:
            self.accelerations += self.gravity
        acceleration = self.accelerations

        self.positions[:, 0] = self.positions[:, 1]
        self.positions[:, 1] += velocity
        np.multiply(acceleration, self.dt ** 2, out=self.work_vec2)
        self.positions[:, 1] += self.work_vec2

        # distance from center
        np.subtract(self.positions[:, 1], self.bounding_center,
                    out=self.work_vec2)
        np.square(self.work_vec2, out=self.work_vec2b)
        np.sum(self.work_vec2b, axis=1, out=self.work_scalar)
        np.sqrt(self.work_scalar, out=self.work_scalar)
        dist = self.work_scalar
        np.divide(self.work_vec2, dist[:, None], out=self.work_vec2b,
                  where=dist[:, None] != 0)

        np.subtract(self.bounding_radius, self.radii, out=self.work_scalar_b)
        limit = self.work_scalar_b

        mask = dist > limit
        self.work_vec2[mask] = self.work_vec2b[mask]
        self.work_vec2[mask] *= limit[mask, None]
        self.work_vec2[mask] += self.bounding_center
        self.positions[mask, 1] = self.work_vec2[mask]

        # handle collisions and linkages on the arrays
        self.solveCollisionCummulative()
        self.updateLinkageCumulative(2)

        # restore fixated particle positions
        self.positions[fixated_mask, 0] = stored_prev
        self.positions[fixated_mask, 1] = stored_cur

    def applyConstantForce(self, particle: Callable, force: np.array) -> np.array:
        acc = force/particle.mass
        return acc

    def applyNewtonGravitationalPotential(self, particle, potential_center: tuple, Force: float):
        to_center = particle.position[1]-np.array(potential_center)
        dist = np.sqrt(to_center.dot(to_center))
        n = to_center/dist
        acc = n * Force / (particle.mass * dist**2)
        return acc

    def applyOszillatingForce(self, particle: Callable, force: float, freq: float) -> np.array:
        acc = force*np.sin(freq*2*np.pi*self.runtime)/particle.mass
        return acc

    def applyCircleConstraint(self, particle, constraint_radius: float, constraint_center: tuple, bbox=False):

        to_center = particle.position[1]-np.array(constraint_center)
        dist = np.sqrt(to_center.dot(to_center))
        n = to_center/dist

        if bbox:
            if dist > constraint_radius-particle.radius:
                #particle.position[0] = particle.position[1]
                particle.position[1] = constraint_center + \
                    n*(constraint_radius-particle.radius)
        else:
            if dist < constraint_radius+particle.radius:
                #particle.position[0] = particle.position[1]
                particle.position[1] = constraint_center + \
                    n*(constraint_radius+particle.radius)

    def solveCollision(self, par1, particle_list):
        """Handling collision of two particles via brute force algr"""
        if len(self.particles) < 2:
            return

        for par2 in particle_list:
            if par1 == par2:
                continue
            # calculate distance between particles and possible collision axis
            min_coll_dist = par1.radius+par2.radius
            coll_axis = par1.position[1]-par2.position[1]
            dist = np.sqrt(coll_axis.dot(coll_axis))
            # skip to next particles when distance is bigger than sum of radii
            if dist > min_coll_dist:
                continue

            # calculate the normalized vector
            n = coll_axis/dist
            # calculat the overlap
            delta = min_coll_dist-dist
            # calculate massfraction
            mass_ratio = par1.mass/(par1.mass + par2.mass)
            # move each particle according to the massratio
            par1.position[1] += mass_ratio*n*delta
            par2.position[1] -= (1-mass_ratio)*n*delta

    def solveCollisionCummulative(self):
        """Resolve particle collisions using a KDTree based approach."""
        if len(self.positions) < 2:
            return

        positions = self.positions[:, 1]
        radii = self.radii
        mass = self.masses

        search_radius = 2 * radii.max()
        if self.use_grid:
            grid = HashGrid()
            grid.build(positions, search_radius)
            pairs = grid.query_pairs(search_radius)
        else:
            tree = cKDTree(positions)
            pairs = np.array(list(tree.query_pairs(search_radius)))
        if len(pairs) == 0:
            return

        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]

        delta = positions[i_idx] - positions[j_idx]
        dist = np.linalg.norm(delta, axis=1)
        min_dist = radii[i_idx] + radii[j_idx]

        mask = dist < min_dist
        if not np.any(mask):
            return

        delta = delta[mask]
        dist = dist[mask]
        min_dist = min_dist[mask]
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]

        n_axis = delta / dist[:, None]
        overlap = min_dist - dist
        mass_ratio = mass[i_idx] / (mass[i_idx] + mass[j_idx])

        shift_i = mass_ratio[:, None] * n_axis * overlap[:, None]
        shift_j = -(1 - mass_ratio)[:, None] * n_axis * overlap[:, None]

        self.accum.fill(0.0)
        np.add.at(self.accum, i_idx, shift_i)
        np.add.at(self.accum, j_idx, shift_j)

        self.positions[:, 1] += self.accum
    
    def updateLinkageCumulative(self, target_distance):
        """Resolve linkage constraints for all linked particle pairs."""
        if len(self.linkage_pairs) == 0:
            return

        positions = self.positions[:, 1]

        i_idx = self.linkage_pairs[:, 0]
        j_idx = self.linkage_pairs[:, 1]

        delta = positions[i_idx] - positions[j_idx]
        dist = np.linalg.norm(delta, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_axis = np.divide(delta, dist[:, None], out=np.zeros_like(delta), where=dist[:, None] != 0)

        shift = 0.5 * (target_distance - dist)[:, None] * n_axis

        self.accum.fill(0.0)
        np.add.at(self.accum, i_idx, shift)
        np.add.at(self.accum, j_idx, -shift)

        self.positions[:, 1] += self.accum

    def extract_results(self):
        self._sync_particles()
        current_data = []
        for idx in range(len(self.particles)):
            current_data.append((
                *self.positions[idx, 1],
                *self.velocities[idx, 1],
                *self.accelerations[idx],
                self.masses[idx],
                self.radii[idx],
            ))
        return current_data

    def run_simulation(self, time: float, steps: int) -> list:
        self.dt = time/steps
        self.runtime = self.dt
        with tqdm(total=steps) as pbar:
            #self.particle_init_delay(steps, delay=PARTICLE_ADDITION_INTERVAL)
            while self.runtime <= time:
                self.update()
                # save the result of the current substep
                self.results.append((self.runtime, self.extract_results()))
                self.runtime += self.dt
                pbar.update(1)
        return self.results

    def run_simulation_iter(self, time: float, steps: int):
        """Yield results step by step while running the simulation."""
        self.dt = time / steps
        self.runtime = self.dt
        for _ in range(steps):
            self.update()
            yield self.runtime, self.extract_results()
            self.runtime += self.dt
