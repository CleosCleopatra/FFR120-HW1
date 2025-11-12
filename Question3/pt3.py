import numpy as np
from matplotlib import pyplot as plt
import random
from statistics import stdev

np.random.seed(5)
random.seed(5)

def grow_trees(forest, p):
    """
    Function to grow new trees in the forest.
    
    Parameters
    ==========
    forest: 2-dimensional array
    p: Probability for a tree to be generated in an empty cell
    """

    Ni, Nj = forest.shape #Dimensions of forrest

    new_trees = np.random.rand(Ni, Nj) #Random number in each place to calc whether tree grows

    #print(f"p is {p} and new_trees is {new_trees}")
    new_trees_indices = np.where(new_trees <= p) #The indices at which new trees actually grow
    forest[new_trees_indices] = 1 #Add trees

    return forest

def propagate_fire(forest, i0, j0):
    """
    Function to propagate the fire on a populated forest.
    
    Parameters
    ==========
    forest: 2-dimensional array
    i0: First index of the cell where the fire occurs
    j0: Second index of the cell where the fire occurs
    """

    Ni, Nj = forest.shape #Dimensions of the forest

    fs = 0 #Initalises fire size as 0

    if forest[i0, j0] == 1: #Tree where lightning strikes
        active_i = [i0] #Initalises list of things on fire
        active_j = [j0] #Initalises list of trees on fire
        forest[i0, j0] = -1 #Sets tree on fire
        fs += 1 #Update fire size

        while len(active_i) > 0: #While any tree is still on fire
            next_i = []
            next_j = []
            for n in np.arange(len(active_i)): #Why do we use np.arange here?
                #Coordinates of cell up
                i = (active_i[n] + 1) % Ni 
                j = active_j[n]
                #Check status
                if forest[i, j] ==1:
                    next_i.append(i)
                    next_j.append(j)
                    forest[i, j] = -1 #Set current tree on fire
                    fs += 1 #Update fire size

                #Coordinates of cell down
                i = (active_i[n] - 1) % Ni
                j = active_j[n]
                #Check status
                if forest[i, j] == 1:
                    next_i.append(i)
                    next_j.append(j)
                    forest[i, j] = -1 
                    fs += 1
                
                #Coordinates of cell left
                i = active_i[n]
                j = (active_j[n] - 1) % Nj
                #Check status
                if forest[i, j] == 1:
                    next_i.append(i)
                    next_j.append(j)
                    forest[i, j] = -1
                    fs += 1
                
                #Coordinates of cell right
                i = active_i[n]
                j = (active_j[n] + 1) % Nj

                if forest[i, j] == 1:
                    next_i.append(i) #add to list
                    next_j.append(j)
                    forest[i, j] = -1
                    fs += 1

                #Coordinates of cell right up
                i = (active_i[n] + 1) % Ni 
                j = (active_j[n] + 1) % Nj

                if forest[i, j] == 1:
                    next_i.append(i) #add to list
                    next_j.append(j)
                    forest[i, j] = -1
                    fs += 1
                
                #Coordinates of cell right down
                i = (active_i[n] - 1) % Ni 
                j = (active_j[n] + 1) % Nj

                if forest[i, j] == 1:
                    next_i.append(i) #add to list
                    next_j.append(j)
                    forest[i, j] = -1
                    fs += 1
                
                #Coordinates of cell left down
                i = (active_i[n] - 1) % Ni 
                j = (active_j[n] - 1) % Nj

                if forest[i, j] == 1:
                    next_i.append(i) #add to list
                    next_j.append(j)
                    forest[i, j] = -1
                    fs += 1
                
                #Coordinates of cell left up
                i = (active_i[n] + 1) % Ni 
                j = (active_j[n] - 1) % Nj

                if forest[i, j] == 1:
                    next_i.append(i) #add to list
                    next_j.append(j)
                    forest[i, j] = -1
                    fs += 1
                
            
            active_i = next_i
            active_j = next_j 
        
    return fs, forest

#Initalise system
#N = 100 #Side of the forest
p = 0.01 #Growth probability
f = 0.2 #Lightning strike probability

#Function for complementary cumulative distribution
def complementary_CDF(f, f_max):
    """
    Function to return the complementary cumulative distribution function.
    
    Parameters
    ==========
    f : Sequence of values (as they occur, non necessarily sorted)
    f_max: Integer, maximum possible value for hte values in f
    """

    num_events = len(f)
    s = np.sort(np.array(f)) / f_max #Sort f in ascending order
    c = np.array(np.arange(num_events, 0, -1)) / (num_events) #Descending

    c_CDF = c
    s_rel = s

    return c_CDF, s_rel



#Compare size of fires in fire grown forest vs. randomly 
# grown forest of the same size (same num of trees)

import random

"""
def powerlaw_random(alpha, x_min, num_drawings):
    
    Function that returns numbers drawn from a probability distribution
    P(x) ~ x ** (- alpha) starting from random numbers in [0, 1].
    
    Parameters
    ==========
    alpha : Exponent of the probability distribution. Must be > 1.
    x_min : Minimum value of the domain of P(x).
    num_drawings : Integer. Numbers of random numbers generated. 
    
    
    if alpha <= 1:
        raise ValueError('alpha must be > 1')

    if x_min <= 0:
        raise ValueError('x_min must be > 0')

            
    r = np.random.rand(num_drawings)
    
    random_values = x_min * r ** (1 / (1 - alpha))

    return random_values
"""

N_list=[16, 32, 64, 128, 256, 512]
target_num_fires = 300
repititions = 5

#Determine exponent for eth empirical cCDF by a linear fit
global_min_rel_size = 1e-3
global_max_rel_size = 5e-2

all_avg_alpha = []
all_stdev_alpha = []

fig, axs = plt.subplots(3, 2)
fig.tight_layout(h_pad=3, w_pad=3)
colours = ["red", "green", "black", "blue", "pink"]
min_rel_size_list = []
for idx, N in enumerate(N_list):
    alpha_list = []
    s_rel_list = [] 
    c_CDF_list = []

    for rep in range(repititions):
        forest = np.zeros([N,N]) #Empty forest
        fire_size = [] #Empty list of fire sizes

        Ni, Nj = forest.shape

        fire_history = []

        num_fires = 0

        while num_fires < target_num_fires:
            print(idx, rep, num_fires)

            forest = grow_trees(forest, p)

            p_lightning = np.random.rand()
            if p_lightning < f:
                i0 = np.random.randint(Ni)
                j0 = np.random.randint(Nj)

                #T = int(np.sum(forest)) #Current number of trees

                fs, forest = propagate_fire(forest, i0, j0)
                if fs > 0:
                    fire_size.append(fs)
                    num_fires += 1

            forest[np.where(forest == -1)] = 0
        print(f'Target of {target_num_fires} fire events reached')

        #Lets compare the forests
        c_CDF, s_rel = complementary_CDF(fire_size, forest.size)

        c_CDF_list.append(c_CDF)
        s_rel_list.append(s_rel)

        min_fit = max(global_min_rel_size, s_rel[0] * 1.05)
        max_fit = min(global_max_rel_size, s_rel[-1] * 0.95)


        is_min = np.searchsorted(s_rel, min_fit)
        is_max = np.searchsorted(s_rel, max_fit)

        new_p = np.polyfit(np.log(s_rel[is_min:is_max]), np.log(c_CDF[is_min: is_max]), 1)

        beta = new_p[0]
        print(f'The empirical cCDF has an exponent beta = {beta:.4}')

        alpha = 1 - beta
        print(f'The empirical prob. distr. exponent: -alpha')
        print(f'with alpha = {alpha:.4}')
        alpha_list.append(alpha)
    
    alpha_sum = sum(alpha_list)
    all_avg_alpha.append(alpha_sum / len(alpha_list))

    row = idx // 2
    col = idx % 2
    print(f"N is {idx} which gives {row, col}, ")
    print(s_rel_list)
    for i in range(len(s_rel_list)):
        axs[row, col].loglog(s_rel_list[i], c_CDF_list[i], '.-', markersize=2,
                   label=f'i = {i}', color = colours[i])

    standard_dev=stdev(alpha_list)
    all_stdev_alpha.append(standard_dev)
    axs[row, col].text(
    0.95, 0.95,
    f'avg alpha = {all_avg_alpha[-1]:.3f} +- {standard_dev:.3f}',
    transform=axs[row, col].transAxes,
    ha='right', va='top',
    fontsize=6
    )
    axs[row, col].legend(fontsize = 4)
    axs[row, col].set_title(f'Empirical cCDF for N={N}', fontsize=6)
    axs[row, col].set_xlabel('relative size', fontsize=4)
    axs[row, col].set_ylabel('c CDF', fontsize=4)
    axs[row, col].tick_params(axis='both', which='both', labelsize=6)
    axs[row, col].axvline(min_fit, color='gray', linestyle='--', linewidth=0.8)
    axs[row, col].axvline(max_fit, color='gray', linestyle='--', linewidth=0.8)


plt.show()


#avg_alpha is list
#stdev_alpha is list

N_rev = [x**-1 for x in N_list]
y_error_min = [all_avg_alpha[i] - all_stdev_alpha[i] for i in range(repititions+1)]
y_error_max = [all_avg_alpha[i] + all_stdev_alpha[i] for i in range(repititions+1)]

print(len(y_error_max))
print(len(all_avg_alpha))
y_error = [y_error_min, y_error_max]

plt.errorbar(N_rev, all_avg_alpha, yerr=y_error, fmt = 'o')
plt.show()