#Simulate Lennard-Jones system (Fig 1.7 in book)
#N=100 particles with diamter 1, mass=1
#Particles mutually interact with L-J potential (Eq 1.1) parameter 1
#System enclosed in squared box with reflecting boundaries
#LxL: size of box
#Initaly particles are position on squared lattice
#Velocity with random orientation and magnituted v=1
#Time step \Delta t=0.001. 
#Magnitude of the interaction force between two particles:
#r=np.sqrt((x[1]-x[2])**2+(y[1]-y[2])**2)
#F(r)=24(parameter/r)*(2*(parameter/r)**12 -(parameter/r)**6))
#Simulate for T_tot=10
#Use leapfrog algorithm
#Pt1:
#Take L=10*parameter
#Plot: configuration of system at t=T_tot, trajectory and mean square displacement MSD of a particle lying n the centre of the system as a function of time t=n \Delta t :
#MSD(n \Delta t)=<(x_(i+n)-x_i)**2+(y_(i+n)-y_i)**2>=(1)/(S-n) \sum_{i=1}^{S-n} (x_{i+n}-x_i)**2+(y_{i+n}-y_i)**2





#Force=m*a


import numpy as np
from matplotlib import pyplot as ply
import random
import math
import time
from scipy.constants import Boltzmann as kB
from tkinter import *

#Initalise variables
N_particles=100
sigma=1
m=1
parameter=1
magnitude=1
dt=0.001
T_tot=10
v0=1
eps = 1 #Energy, should I hav this?

def list_neighbours(x,y,N_particles, cutoff_radius):
    neighbours=[]
    neighbour_number=[]
    for j in range(N_particles):
        distance=np.sqrt((x - x[j]) **2 + (y - y[j]) ** 2)
        neighbour_indices=np.where(distance <= cutoff_radius)
        neighbours.append(neighbour_indices)
        neighbour_number.append(len(neighbour_indices))
    return neighbours, neighbour_number


def total_force_cutoff(x, y, N_particles, sigma, epsilon, neighbours):
    Fx=np.zeros(N_particles)
    Fy=np.zeros(N_particles)
    for i in range(N_particles):
        for j in list(neighbours[i][0]):
            if i != j:
                r2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                r = np.sqrt(r2)
                ka2 = sigma ** 2 / r2

                F = 24 * epsilon / r * (2 * ka2 ** 6 - ka2 ** 3)
                Fx[i] += F * (x[i] - x[j]) / r
                Fy[i] += F * (y[i] - y[j]) / r
    return Fx, Fy

def graph(x, L, y):
    window_size=600
    tk=Tk()
    tk.geometry(f'{window_size+20}x{window_size +20}')
    tk.configure(background = '#000000')
    canvas = Canvas(tk, background = '#ECECEC')
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    specialparticle=canvas.create_oval(
        (x[0]-sigma / 2) / L *window_size +window_size / 2,
        (y[0] - sigma / 2) / L * window_size +window_size / 2,
        (x[0] + sigma / 2) / L * window_size + window_size / 2,
        (y[0] + sigma / 2) / L * window_size + window_size /2,
        outline='#000000',
        fill='#000000', 
    )
    particles = []
    for j in range(1, N_particles):
        particles.append(
            canvas.create_oval(
                (x[j] - sigma / 2) / L * window_size + window_size / 2, 
                (y[j] - sigma / 2) / L * window_size + window_size / 2,
                (x[j] + sigma / 2) / L * window_size + window_size / 2, 
                (y[j] + sigma / 2) / L * window_size + window_size / 2,
                outline='#00C0C0', 
                fill='#00C0C0',
            )
        )
    tk.title(f'Final time')
    tk.update_idletasks
    tk.update()
    time.sleep(1)


def running(x, y, vx, vy, x_half, y_half, nx, ny, nvx, nvy, neighbours, x_min, x_max, y_min, y_max, nphi, cutoff_radius):
    #Picking the particle to follow
    index_to_follow=int(len(x)/2)

    msd_values=[]
    past_values_x=[]
    past_values_y=[]
    past_values_x.append(x[index_to_follow])
    past_values_y.append(y[index_to_follow])

    for t in range(int(T_tot/dt)):
        print(t)
        x_half = x + 0.5 * vx * dt
        y_half = y + 0.5 * vy * dt

        fx, fy= total_force_cutoff(x_half, y_half, N_particles, sigma, eps, neighbours)

        nvx = vx + fx / m * dt
        nvy = vy + fy / m * dt

        nx = x_half + 0.5 * nvx * dt
        ny = y_half + 0.5 * nvy * dt

        #Reflecting boundaries

        for j in range(N_particles):
            if nx[j] < x_min:
                nx[j] = x_min + (x_min - nx[j])
                nvx[j] = - nvx[j]

            if nx[j] > x_max:
                nx[j] = x_max - (nx[j] - x_max)
                nvx[j] = - nvx[j]
            
            if ny[j] < y_min:
                ny[j] = y_min + (y_min - ny[j])
                nvy[j] = - nvy[j]
            
            if ny[j] > y_max:
                ny[j] = y_max - (ny[j] - y_max)
                nvy[j] = - nvy[j]
            
            if j==index_to_follow:
                past_values_x.append(nx)
                past_values_y.append(ny)


            #msd_values[j] += (nx[j]-x[j]) ** 2 + (ny[j] - y[j]) ** 2


        nv = np.sqrt(nvx ** 2 + nvy **2)
        for i in range(N_particles):
            nphi[i] = math.atan2(nvy[i], nvx[i])
        
        if t % 10 == 0:
            neighbours, neighbour_number = list_neighbours(nx, ny, N_particles, cutoff_radius)
        
        #Update variables
        x = nx
        y = ny
        vx = nvx
        vy = nvy
        v = nv
        phi = nphi

    sums=0
    for val in msd_values:
        sums+= val
    msd_final=sums/((int(T_tot/dt))- )


    return x,y, msd_values

import matplotlib.pyplot as plt

def main_part(L):
    #Set initial positions:
    x_min, x_max, y_min, y_max=-L/2, L/2, -L/2, L/2
    cutoff_radius=5*sigma #WHat does this do?
    
    x0, y0=np.meshgrid(
        np.linspace(x_min, x_max, int(np.sqrt(N_particles))),
        np.linspace(y_min, y_max, int(np.sqrt(N_particles)))
    )
    x0=x0.flatten()[:N_particles]
    y0=y0.flatten()[:N_particles]
    phi0=(2*np.random.rand(N_particles)-1)*np.pi

    neighbours, neighbour_number=list_neighbours(x0,y0,N_particles, cutoff_radius)
    #velocity.append([np.cos(angle), np.sin(angle)])
    #Leapfrog algorithm
    #Position is advanced for half a time step to obatin r_{n+1/2}
    #Current tiem
    x=x0
    y=y0
    x_half=np.zeros(N_particles)
    y_half=np.zeros(N_particles)
    v=v0
    phi=phi0
    vx = v0 * np.cos(phi0)
    vy = v0 * np.sin(phi0)

    #Next time
    nx=np.zeros(N_particles)
    ny=np.zeros(N_particles)
    nv=np.zeros(N_particles)
    nphi=np.zeros(N_particles)
    nvx=np.zeros(N_particles)
    nvy = np.zeros(N_particles)

    final_x, final_y, msd=running(x, y, vx, vy, x_half, y_half, nx, ny, nvx, nvy, neighbours, x_min, x_max, y_min, y_max, nphi, cutoff_radius)
    graph(final_x, L, final_y)
    plt.plot(np.arange(0, T_tot, dt), msd)
    plt.show()



#pt 1
main_part(10*sigma)

