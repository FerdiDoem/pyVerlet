import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from particle import VerletObject
from solver import Solver


def test_update_particle_at_center_no_nan():
    p = VerletObject(1, pos_x=0.0, pos_y=0.0, radius=1)
    solver = Solver([p])
    solver.dt = 0.1
    solver.runtime = 0
    solver.update()
    assert not np.isnan(p.position).any()
