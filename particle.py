# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:01:50 2022

@author: Ferdinand Dömling

A simple physics simulation inspired by https://www.youtube.com/c/PezzzasWork
"""

import numpy as np

class VerletObject:

    def __init__(self,
                 ID,
                 pos_x: float = 0.,
                 pos_y: float = 0.,
                 v_x: float = 0.,
                 v_y: float = 0.,
                 a_x: float = 0.,
                 a_y: float = 0.,
                 radius: float = 1,
                 density: float = 1,
                 ):

        self.ID = ID
        self.position = np.array([[pos_x, pos_y], [pos_x, pos_y]])
        self.acceleration = np.array([a_x, a_y])
        self.velocity = np.array([[v_x, v_y], [v_x, v_y]])
        self.density = density
        self.mass = density * radius
        self.radius = radius
        self.linkage = []
        self.fixated = False
    
