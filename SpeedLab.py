# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 21:38:49 2022

@author: Ferdinand Dömling

"""

import numpy as np 
import math

if True:
    def npLinalgNorm(arr, n):
        array = arr
        for _ in range(n):
            dist = np.linalg.norm(array)
        return dist
            
    def npSqrt(arr,n):
        array = arr
        for _ in range(n):
            dist = np.sqrt((array**2).sum())
        return dist
            
    
    n_experiments = 10
    n_sampling = 1000000
    
    for _ in range(n_experiments):
        array = np.random.random(2)
        dist1 = npLinalgNorm(array, n_sampling)
        dist2 = npSqrt(array, n_sampling)