import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from grid import HashGrid


def test_query_pairs_simple():
    pts = np.array([[0, 0], [0.4, 0], [3, 4], [3.3, 4]])
    grid = HashGrid()
    grid.build(pts, cell_size=1.0)
    pairs = grid.query_pairs(radius=0.5)
    assert sorted(map(tuple, pairs)) == [(0, 1), (2, 3)]


def test_no_pairs():
    pts = np.array([[0, 0], [5, 5]])
    grid = HashGrid()
    grid.build(pts, cell_size=1.0)
    pairs = grid.query_pairs(radius=0.5)
    assert pairs.size == 0

