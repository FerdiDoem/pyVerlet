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
from itertools import combinations
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

    def update(self):
        """Updates the particle instance for each substep."""
        fixated_particles = []
        for particle in self.particles:
            #taking care of fixated particles
            if particle.fixated:
                pos_stored = np.array(particle.position)
                fixated_particles.append([particle,pos_stored])
            # update position
            particle.updatePosition(self.dt)
            # calc accleration for this step
            particle.acceleration = np.zeros(2)
            if self.runtime <= ACCELERATION_DURATION:
                particle.updateAcceleration(self.gravity)
                #particle.updateAcceleration(self.applyNewtonGravitationalPotential(particle, (0, 0), -500))
                pass

            # Check Constraints here
            self.applyCircleConstraint(
                particle, self.bounding_radius, self.bounding_center, bbox=True)
            #self.applyCircleConstraint(particle, 10, (0, 0))
            #self.applyCircleConstraint(particle, 10, (-50, -25))
            #self.applyCircleConstraint(particle, 10, (50, -50))

            # check collision
            #self.solveCollision(particle, self.particles)
        
        # check collision at the end of the cycle
        self.solveCollisionCummulative()
        
        # constraint particles by linkage
        self.updateLinkageCumulative(2)
        

        
        #reset the position of the fixated particle
        for part_pos in fixated_particles:
            part_pos[0].position = part_pos[1]

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
        """Resolve particle collisions using a vectorized approach."""
        if len(self.particles) < 2:
            return

        positions = np.array([p.position[1] for p in self.particles])
        radii = np.array([p.radius for p in self.particles])
        mass = np.array([p.mass for p in self.particles])

        delta = positions[:, None, :] - positions[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        min_dist = radii[:, None] + radii[None, :]

        mask = (dist < min_dist) & np.triu(np.ones_like(dist, dtype=bool), k=1)
        if not np.any(mask):
            return

        safe_dist = np.where(mask, dist, 1.0)
        n_axis = delta / safe_dist[..., None]
        shift = (min_dist - dist)[..., None] * n_axis * mask[..., None]

        mass_ratio = mass[:, None] / (mass[:, None] + mass[None, :])
        pos_shift = (
            np.sum(mass_ratio[..., None] * shift, axis=1) -
            np.sum((1 - mass_ratio)[..., None] * shift, axis=0)
        )

        positions += pos_shift

        for p, new in zip(self.particles, positions):
            p.position[1] = new
    
    def updateLinkageCumulative(self,target_distance):
        for par1,par2 in combinations(self.particles,2):
            if par1.ID not in par2.linkage:
                continue
            min_coll_dist = par1.radius+par2.radius
            coll_axis = par1.position[1]-par2.position[1]
            dist = np.sqrt(coll_axis.dot(coll_axis))
            # calculate the normalized vector
            n = coll_axis/dist
            # calculat the overlap
            delta = target_distance-dist
            # calculate massfraction
            radius_ratio = par1.radius/(par1.radius + par2.radius)
            # move each particle according to the massratio
            par1.position[1] += 0.5*n*delta
            par2.position[1] -= 0.5*n*delta

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
