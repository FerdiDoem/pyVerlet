# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:38:49 2022

@author: Ferdinand Dömling

"""

import numpy as np 
import math
from scipy.spatial import distance

if True:
    #Testing different versions for calculating the length of a 2D Array
    #Winner: npDot closely followed by mathSqrtv2
    
    def npLinalgNorm(array, n):
        for _ in range(n):
            dist = np.linalg.norm(array)
        return dist
            
    def npSqrt(array,n):
        for _ in range(n):
            dist = np.sqrt((array**2).sum())
        return dist
    
    def npSqrtv2(array,n):
        for _ in range(n):
            dist = np.sqrt((array*array).sum())
        return dist
    
    def mathSqrt(array,n):
        for _ in range(n):
            dist = math.sqrt((array**2).sum())
        return dist
    
    def mathSqrtv2(array,n):
        for _ in range(n):
            dist = math.sqrt(sum(array**2))
        return dist
    
    def npDot(array,n):
        for _ in range(n):
            dist = np.sqrt(array.dot(array))
        return dist
            
    n_experiments = 10
    n_sampling = 10000
    
    for _ in range(n_experiments):
        array = np.random.random(2)*100
        dist1 = npLinalgNorm(array, n_sampling)
        dist2 = npSqrt(array, n_sampling)
        dist3 = npSqrtv2(array, n_sampling)
        dist4 = mathSqrt(array, n_sampling)
        dist5 = mathSqrtv2(array, n_sampling)
        dist6 = npDot(array, n_sampling)