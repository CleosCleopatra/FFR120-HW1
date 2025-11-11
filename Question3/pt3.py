#Forest: Square lattice N x N initially empty
#Each cell of the lattic can host a tree
#Each round: Trees can grow in the forest (probability: p) 
#and a lightning can hit (probability: f)
#If the lightning hits a cell with a tree, ti ignites a fire 
#event
#The fire:
# Propagates to the neighbouring cells (Von Neumann neighbourhood)
#that are occupied by a tree
#Is propagated to the neighbouts recursively
#Cannot propagate to empty cells or cells that were already ignited
#Stops whn it doesnt find any more tree to burn
#After fire, caluclate num of trees burned

#0: Empty cell
#1: Cell with tree
#-1: cell on fire during fire event (turns to 0 at end of round)

import numpy as np
from matplotlib import pyplot as plt

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
N = 100 #Side of the forest
p = 0.01 #Growth probability
f = 0.2 #Lightning strike probability

forest = np.zeros([N,N]) #Empty forest
fire_size = [] #Empty list of fire sizes

#Run simulation
Ni, Nj = forest.shape

target_num_fires = 300

fire_history = []

num_fires = 0

"""
#Plot fire history over time
from matplotlib import pyplot as plt
t = np.array(np.arange(len(fire_history)))
fh = np.array(fire_history) / forest.size

plt.plot(t, fh, '.-', color='k', markersize=5)
plt.title('Fire History')
plt.xlabel('step')
plt.ylabel('relative fire size')
plt.show()
#Plot fire sizes over time
max_fire_size = max(fire_size)

bin_width = 100
bins_edges = (np.arange(0, max_fire_size + bin_width, bin_width) - 0.5 * bin_width)
bins = bins_edges[1:] + 0.5 * bin_width

occurrance = np.histogram(fire_size, bins=bins_edges)

plt.bar(bins, occurrance[0], color= 'r', width=0.4, edgecolor='k')
plt.title('Histogram of fire sizes')
plt.xlabel('size')
plt.ylabel('occurrence')
plt.show()
"""

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


"""
c_CDF, s_rel = complementary_CDF(fire_size, forest.size)

plt.loglog(s_rel, c_CDF, ".-", color='k', markersize=5, linewidth= 0.5)
plt.title('Empirical cCDF')
plt.xlabel('relative size')
plt.ylabel('c CDF')
plt.show()
"""

#Compare size of fires in fire grown forest vs. randomly 
# grown forest of the same size (same num of trees)

import random

def random_forest(Ni, Nj, T):
    """
    Function to return a randomly grown forest.
    Returns also the coordinates of the random ignition point
    
    Parameters
    ==========
    Ni: First dimension of the forest array
    Nj: Second dimension fo the forest array
    T: Integer, num of trees in forest
    """

    rf = np.zeros([Ni, Nj])

    nt = random.sample(range(Ni * Nj), T)
    i_list = list(map(lambda x: x % Ni, nt))
    j_list = list(map(lambda x: x // Ni, nt))

    rf[i_list, j_list] = 1

    #Coordinates of where lightning strikes
    ignition = np.random.randint(T)
    i_fire = i_list[ignition]
    j_fire = j_list[ignition] 

    return rf, i_fire, j_fire

#Comparision of the fires:
#N = 100 #Side of the forest
#p = 0.01 #Growth probability
#f = 0.2 #Lightning strike probability
#target_num_fires = 300

#forest = np.zeros([N,N])
#fire_size = []
rf_fire_size = [] #Empty list of fire history for random forest

#num_fires = 0
#Ni, Nj = forest.shape

while num_fires < target_num_fires:
    print(num_fires)

    forest = grow_trees(forest, p)

    p_lightning = np.random.rand()
    if p_lightning < f:
        i0 = np.random.randint(Ni)
        j0 = np.random.randint(Nj)

        T = int(np.sum(forest)) #Current number of trees

        fs, forest = propagate_fire(forest, i0, j0)
        if fs > 0:
            fire_size.append(fs)
            num_fires += 1

            #Generate random forest for a comparision
            rf, i0_rf, j0_rf = random_forest(Ni, Nj, T)
            fs_rf, rf = propagate_fire(rf, i0_rf, j0_rf)
            rf_fire_size.append(fs_rf)

    forest[np.where(forest == -1)] = 0
print(f'Target of {target_num_fires} fire events reached')

#Lets compare the forests
c_CDF, s_rel = complementary_CDF(fire_size, forest.size)

c_CDF_rf, s_rel_rf = complementary_CDF(rf_fire_size, forest.size)


plt.loglog(s_rel, c_CDF, '.-', color='k', markersize=5,
           label='grown with fire')
plt.loglog(s_rel_rf, c_CDF_rf, '.-', color='b', markersize=5,
           label='randomly generated')
plt.title('Empirical cCDF')
plt.xlabel('relative size')
plt.ylabel('c CDF')
plt.legend()
plt.show()

#Determine exponent for eth empirical cCDF by a linear fit
min_rel_size = 1e-3
max_rel_size = 5e-2

is_min = np.searchsorted(s_rel, min_rel_size)
is_max = np.searchsorted(s_rel, max_rel_size)

p = np.polyfit(np.log(s_rel[is_min:is_max]), np.log(c_CDF[is_min: is_max]), 1)

beta = p[0]
print(f'The empirical cCDF has an exponent beta = {beta:.4}')

alpha = 1 - beta
print(f'The empirical prob. distr. exponent: -alpha')
print(f'with alpha = {alpha:.4}')

def powerlaw_random(alpha, x_min, num_drawings):
    """
    Function that returns numbers drawn from a probability distribution
    P(x) ~ x ** (- alpha) starting from random numbers in [0, 1].
    
    Parameters
    ==========
    alpha : Exponent of the probability distribution. Must be > 1.
    x_min : Minimum value of the domain of P(x).
    num_drawings : Integer. Numbers of random numbers generated. 
    """
    
    if alpha <= 1:
        raise ValueError('alpha must be > 1')

    if x_min <= 0:
        raise ValueError('x_min must be > 0')

            
    r = np.random.rand(num_drawings)
    
    random_values = x_min * r ** (1 / (1 - alpha))

    return random_values


x_min = 1  # minimum value for the generated numbers
num_drawings = 5000  

pl_size = powerlaw_random(alpha, x_min, num_drawings)

c_CDF_pl, s_rel_pl = complementary_CDF(pl_size, forest.size)

# Note loglog plot!
plt.loglog(s_rel, c_CDF, '.-', color='k', linewidth=1, 
           label='empirical')
plt.loglog(s_rel_pl, c_CDF_pl, '-', color='g', linewidth=3, 
           label='synthetic data')
plt.xlim([min(s_rel), 1])
plt.title('Comparison with synthetic data')
plt.xlabel('relative size')
plt.ylabel('c CDF')
plt.legend()
plt.show()