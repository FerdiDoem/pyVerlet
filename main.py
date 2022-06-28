# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:50:35 2022

@author: Ferdinand Dömling


"""
from particle import VerletObject
from solver import Solver
import time
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

from generator import Generator


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def main():
    # basic properties
    time = 5
    substeps = 800.

    # create particles
    n_p = 10
    bounding_box_radius = 50
    v0 = np.array([0., 0.])*time/substeps
    
    Gen1 = Generator(VerletObject)
    particle_system = list(Gen1.rnd_particle_gen(
        n_p, bounding_box_radius, v0=v0))
    
    Gen2 = Generator(VerletObject)
    particle_system.extend(list(Gen2.rnd_particle_gen(
        n_p, bounding_box_radius, v0=v0,start_ID=n_p+1)))
    
    #link particle together
    Gen1.chain_particles()
    
    #modify particles
    particle_system[0].fixated = True
    # define simulation
    solver = Solver(particle_system)

    # run simulation
    res = solver.run_simulation(time, substeps)

    # visualize the results
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_facecolor('darkgray')
    circle1 = plt.Circle((0, 0), bounding_box_radius, color='black', fill='white')
    circle2 = plt.Circle((0, 0), 10, color='r', fill=False)
    circle3 = plt.Circle((-50, -25), 10, color='r', fill=False)
    circle4 = plt.Circle((50, -50), 10, color='r', fill=False)
    ax.add_patch(circle1)
    # ax.add_patch(circle2)
    # ax.add_patch(circle3)
    # ax.add_patch(circle4)
    fig.canvas.draw()
    px_per_scale = (ax.get_window_extent().width /
                    (2*bounding_box_radius+2) * 72./fig.dpi)
    frames = []
    lst_E_Kin = []
    for step_data in res:
        particles = np.vstack(step_data[1])
        frame = ax.scatter(particles[:, 0],
                           particles[:, 1],
                           #c=particles[:, 6],
                           cmap='gist_rainbow',
                           c=np.linalg.norm(particles[:, 6:8], axis=1),
                           edgecolors='white',
                           s=(px_per_scale*2*particles[:, 7])**2,
                           linewidth=0,
                           )
        # not the real kinetic energy, just the delta of the position at each step
        E_kin = sum(0.5*particles[:, 7] *
                    np.linalg.norm(particles[:, 2:4], axis=1)**2)
        frames.append([frame])
        lst_E_Kin.append(E_kin)
    print('Frames created. Animation follows...')

    # export results to external viewer
    dpi = 300
    framerate = substeps/time
    interval = time*1000//substeps
    rendering = anim.ArtistAnimation(
        fig, frames, blit=True, interval=interval, repeat=True)
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(np.arange(0, len(lst_E_Kin), 1), lst_E_Kin, label='System')
    ax2.plot(np.arange(0,len(lst_E_Kin),1))
    ax2.set_xlabel('substeps')
    ax2.set_ylabel('$equv. E_{Kin}$ ')
    plt.legend()
    plt.close()

    return res, rendering


if __name__ == '__main__':
    start = time.time()
    debug, rendering = main()
    #debug = main()
    end = time.time()
    print(f'Done in {round(end-start,2)} s!')
    #plt.show()
    #plt.close()
    
    #writer = anim.writers['ffmpeg'](fps=framerate)
    #rendering.save('PhySim.mp4', writer=writer, dpi=dpi)
