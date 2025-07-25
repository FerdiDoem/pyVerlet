import time
import numpy as np
from particle import VerletObject
from solver import Solver

def main():
    n = 1000
    particles = [VerletObject(i, pos_x=0.0, pos_y=0.0) for i in range(n)]
    s = Solver(particles)
    s.dt = 0.01
    start = time.perf_counter()
    for _ in range(100):
        s.update()
        s.runtime += s.dt
    duration = (time.perf_counter() - start) / 100.0
    print(f'{duration * 1e6:.2f} μs/step')

if __name__ == '__main__':
    main()
