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

def running(x, y, vx, vy, x_half, y_half, nx, ny, nvx, nvy, neighbours, x_min, x_max, y_min, y_max, nphi, cutoff_radius):
    for t in range(int(T_tot/dt)):
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

def part_1(L):
    #Set initial positions:
    position=[]
    velocity=[]
    step=L/10
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

    running(x, y, vx, vy, x_half, y_half, nx, ny, nvx, nvy, neighbours, x_min, x_max, y_min, y_max, nphi, cutoff_radius)

    #new_position=[]
    #new_velocity=[]
    #for i in range(N):
    #    for j in range(2):
    #        position_half=position[i][j]+velocity[i][j]*(time_step/2)
    #        force=0
    #        for other in range(N):
    #            if other!=i:
    #                dist=np.sqrt((position[i][j]-position[other][j])**2+(position[i][j]-position[other][j])**2)
    #                force+=24*(parameter/dist)*(2*(diameter/dist)**12-(diameter/dist)**6)#

    #        F_half=F(position_half)    =-(U(r_n+\delta r)- U(r_n-\delta r))/(2 \Delta r)    #F(r)=-\Delta U(r)
    #        new_velocity.append(velocity[i][j]+())

    #Then velocity v_{n+1} is calculated using F_{n+1/2}=F(r_{n+1/2})
    #FInally yhe position is advanced for another half time step to r_{n+1}

part_1(10*sigma)

