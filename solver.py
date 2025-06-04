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

    def __init__(self, particles: list,):
        self.gravity = np.array([0, GRAVITY])
        self.force = np.array([CONST_FORCE_VEC_X, CONST_FORCE_VEC_Y])
        self.particles = particles
        self.bounding_center = np.array([BBOX_CENTER_X, BBOX_CENTER_Y])
        self.bounding_radius: float = BBOX_ROUND_RADIUS
        self.runtime: float
        self.dt: float
        self.results = []
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

    def update(self):
        """Update particle states using numpy operations."""

        if not self.particles:
            return

        fixated_mask = np.array([p.fixated for p in self.particles])
        stored_prev = np.array([p.position[0] for p in self.particles])[fixated_mask]
        stored_cur = np.array([p.position[1] for p in self.particles])[fixated_mask]

        prev_pos = np.array([p.position[0] for p in self.particles])
        cur_pos = np.array([p.position[1] for p in self.particles])
        radii = np.array([p.radius for p in self.particles])

        velocity = cur_pos - prev_pos
        acceleration = np.zeros_like(cur_pos)
        if self.runtime <= ACCELERATION_DURATION:
            acceleration += self.gravity

        prev_pos = cur_pos
        cur_pos = cur_pos + velocity + acceleration * (self.dt ** 2)

        to_center = cur_pos - self.bounding_center
        dist = np.linalg.norm(to_center, axis=1)
        n = to_center / dist[:, None]
        limit = self.bounding_radius - radii
        mask = dist > limit
        cur_pos[mask] = self.bounding_center + n[mask] * limit[mask, None]

        for idx, p in enumerate(self.particles):
            p.position[0] = prev_pos[idx]
            p.position[1] = cur_pos[idx]
            p.velocity[1] = velocity[idx]
            p.acceleration = acceleration[idx]

        # handle collisions and linkages
        self.solveCollisionCummulative()
        self.updateLinkageCumulative(2)

        # restore fixated particle positions
        fix_indices = np.where(fixated_mask)[0]
        for store_idx, obj_idx in enumerate(fix_indices):
            self.particles[obj_idx].position[0] = stored_prev[store_idx]
            self.particles[obj_idx].position[1] = stored_cur[store_idx]

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
        if len(self.particles) < 2:
            return

        positions = np.array([p.position[1] for p in self.particles])
        radii = np.array([p.radius for p in self.particles])
        mass = np.array([p.mass for p in self.particles])

        tree = cKDTree(positions)
        search_radius = 2 * radii.max()
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

        accum = np.zeros_like(positions)
        np.add.at(accum, i_idx, shift_i)
        np.add.at(accum, j_idx, shift_j)

        positions += accum

        for p, new in zip(self.particles, positions):
            p.position[1] = new
    
    def updateLinkageCumulative(self, target_distance):
        """Resolve linkage constraints for all linked particle pairs."""
        if len(self.linkage_pairs) == 0:
            return

        positions = np.array([p.position[1] for p in self.particles])

        i_idx = self.linkage_pairs[:, 0]
        j_idx = self.linkage_pairs[:, 1]

        delta = positions[i_idx] - positions[j_idx]
        dist = np.linalg.norm(delta, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            n_axis = np.divide(delta, dist[:, None], out=np.zeros_like(delta), where=dist[:, None] != 0)

        shift = 0.5 * (target_distance - dist)[:, None] * n_axis

        accum = np.zeros_like(positions)
        np.add.at(accum, i_idx, shift)
        np.add.at(accum, j_idx, -shift)

        positions += accum

        for p, new in zip(self.particles, positions):
            p.position[1] = new

    def extract_results(self):
        current_data = []
        for particle in self.particles:
            current_data.append((*particle.position[1],
                                 *particle.velocity[1],
                                 *particle.acceleration,
                                 particle.mass,
                                 particle.radius))
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
