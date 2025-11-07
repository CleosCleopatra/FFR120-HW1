#Simulate Lennard-Jones system (Fig 1.7 in book)
#N=100 particles with diamter 1, mass=1
#Particles mutually interact with L-J potential (Eq 1.1) parameter 1
#System enclosed in squared box with reflecting boundaries
#LxL: size of box
#Initaly particles are position on squared lattice
#Velocity with random orientation and magnituted v=1
#Time step \Delta t=0.001. 
#Magnitude of the interaction force between two particles:
r=np.sqrt((x[1]-x[2])**2+(y[1]-y[2])**2)
F(r)=24(parameter/r)*(2*(parameter/r)**12 -(parameter/r)**6))
#Simulate for T_tot=10
#Use leapfrog algorithm
#Pt1:
#Take L=10*parameter
#Plot: configuration of system at t=T_tot, trajectory and mean square displacement MSD of a particle lying n the centre of the system as a function of time t=n \Delta t :
MSD(n \Delta t)=<(x_(i+n)-x_i)**2+(y_(i+n)-y_i)**2>=(1)/(S-n) \sum_{i=1}^{S-n} (x_{i+n}-x_i)**2+(y_{i+n}-y_i)**2





Force=m*a


import numpy as np
from matplotlib import pyplot as ply
import random
import math

#Initalise variables
N=100
diameter=1
mass=1
parameter=1
magnitude=1
time_step=0.001
T_tot=10


def part_1(L):
    #Set initial positions:
    position=[]
    velocity=[]
    step=L/10
    for i in range(10):
        for j in range(10):
            position.append([step*i, step*j])
            angle=random.choice(range(0, 2*math.pi))
            velocity.append([np.cos(angle), np.sin(angle)])
    #Leapfrog algorithm
    #Position is advanced for half a time step to obatin r_{n+1/2}
    new_position=[]
    new_velocity=[]
    for i in N:
        for j in range(2):
            position_half=position[i][j]+velocity[i][j]*(time_step/2)
            F_half=F(position_half)    =-(U(r_n+\delta r)- U(r_n-\delta r))/(2 \Delta r)    #F(r)=-\Delta U(r)
            new_velocity.append(velocity[i][j]+())

    #Then velocity v_{n+1} is calculated using F_{n+1/2}=F(r_{n+1/2})
    #FInally yhe position is advanced for another half time step to r_{n+1}


