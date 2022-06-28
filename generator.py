# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 23:36:34 2022

@author: Ferdinand Dömling

"""

from typing import Callable
import numpy as np

class Generator:
    def __init__(self, particle_type:Callable):
        self.pt = particle_type
        self.pt_list = []
    
    def rnd_particle_gen(self, n: int,
                         bounding_box_radius: float,
                         start_ID = 0,
                         v0: np.array = np.array((0., 0.)),
                         a0: tuple = (0., 0.)) -> Callable:
        """
        Generates random positioned particles inside the bounding area.
    
        """
        for i in range(1, n+1):
    
            # create particles
            particle = self.pt(i)
    
            # for a circular bounding box
            phi = np.random.random() * np.pi
            r = np.random.random() * (bounding_box_radius-particle.radius)
            pos = r * np.cos(phi), r * np.sin(phi)
    
            # set inital state
            particle.position = np.array((pos, pos))
            particle.position[0] -= v0
            particle.acceleration = np.array(a0)
    
            # set static properties
            particle.density = np.random.randint(1, 7)
            particle.radius = np.random.randint(1, 5)
            particle.mass = particle.density*particle.radius
    
            # debugging
            #particle.mass = 1
            particle.radius = 1
            self.pt_list.append(particle)
            yield particle
        
    
    def chain_particles(self):
        # TODO: Move linked objects closer together
        self.pt_list[0].linkage.append(self.pt_list[1].ID)
        for idx,link in enumerate(self.pt_list[1:len(self.pt_list)-1]):
            link.linkage.append(self.pt_list[idx].ID)
            link.linkage.append(self.pt_list[idx+2].ID)
        self.pt_list[-1].linkage.append(self.pt_list[-2].ID)
        
        
            
        