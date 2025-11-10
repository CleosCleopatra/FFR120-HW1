#Ising model on N x N squared lattice with periodic boundary conditions
#N>=100
#H=0
#J=1
#K_b=1
#T=2.3 (teemp slightly higher than critical temp)
#Initalise spins randomly with equal probability being +1 or -1
#To simulate the two parallel plates, set spins in row N_1=N/2-d_half
#All equal to s_1 and the spins in the row N_2=N/2+d_half all equal to s_1 and the spins in the row N_2=N/2 + d_half all e


"""

#Implement function to calculate the probability distribution given the energies of the possible states of the system
import numpy as np 
    
def probability_distribution(Ei):

    Function to generate the probability distribution for a system with states 
    with Ei energies.
    
    Parameters
    ==========
    Ei : Array (energies) [kBT].
    
    
    Z = sum(np.exp(- Ei))  # Partition function.
    
    pi = 1 / Z * np.exp(- Ei)  # Probability.
    
    return pi, Z


#Initalising the system

N = 3  # Number of states.
Eb = 2  # Energy barrier [kBT].

xi = np.arange(N)  # States.

Ei = np.array([0, Eb, 0])  # Energies.

pi, Z = probability_distribution(Ei)  # Partition function.


#Plot energy spectrum
from matplotlib import pyplot as plt

#Plot probability distribution
plt.bar(xi, Ei, color='#FF8000', width=0.4, edgecolor='#A04000')
plt.title('Energy spectrum')
plt.xlabel('states')
plt.ylabel('Energy (kBT)')
plt.show()

plt.bar(xi, pi, color='c', width=0.4, edgecolor='k')
plt.title('Probability distribution')
plt.xlabel('states')
plt.ylabel('Probability')
plt.show()

#

def next_state_3(x, pi):
    
    Function to generate the next state given the probability distribution and 
    the current state. 
    Specialized to the case of 3 states (0, 1, 2).
    
    Parameters
    ==========
    x  : current state (between 0 and 2)
    pi : probability distribution
    
    
    if x not in (0, 1, 2):
        raise ValueError('x must be 0 or 1 or 2')
   
    p = np.random.rand()
    
    if p < pi[0]:
        x_next = 0
    elif p < pi[0] + pi[1]:
        x_next = 1
    else:
        x_next = 2
    
    if abs(x - x_next) > 1:
        x_next = x
            
    return x_next


#Let system evolve for certain numbre of time steps

N_steps = 6000

x = np.zeros(N_steps); x[0] = 0
for i in range(N_steps - 1):
    x[i + 1] = next_state_3(x[i], pi)


#Plot the evolution of the system state
t = np.arange(N_steps)

plt.plot(t, x, ".-", color='k', markersize=5, linewidth=0.5)
plt.title('Trajectory')
plt.xlabel('t (s)')
plt.ylabel('x (state label)')
plt.show()


#plot the histograms of the state occupancy in the trajectory
bins_edges = np.arange(4) - 0.5
occupancy = np.histogram(x, bins=bins_edges)

plt.bar(xi, occupancy[0], color='m', width=0.4, edgecolor='k')
plt.xlabel('states')
plt.ylabel('# occurrences')
plt.show()

#Plot experimental probability, compare it with the a priori probability
pi_exp = occupancy[0] / sum(occupancy[0])

plt.bar(xi, pi, color='c', width=0.4, edgecolor='k', label='a priori')
plt.bar(xi, pi_exp, color='m', width=0.1, edgecolor='k', label='experimental')
plt.title('Probability')
plt.xlabel('states')
plt.ylabel('Probability')
plt.legend()
plt.show()







#Simulating system
N = ?  # Size of the spin lattice, supposed to be greater than or equal to 100
H = 0  # External field.
J = 1  # Spin-spin coupling.
T = 2.3  # Temperature. Critical temperature ~2.269.


#Initalise them with equal probability, all equal to s1?
sl = 2 * np.random.randint(2, size=(N, N)) - 1

N_up = np.sum(sl + 1) / 2
N_down = N * N - N_up

print(f"Spin lattice created:  N_up={N_up}  N_down={N_down}")

#Write function that returns the values of the neighbours of the spin position
def neighboring_spins(i_list, j_list, sl):
    Function returning the position of the neighbouring spins of a list of 
    spins identified by their positions in the spin lattice.
    
    Parameters
    ==========
    i_list : Spin position first indices.
    j_list : Spin position second indices.
    sl : Spin lattice.
    

    Ni, Nj = sl.shape  # Shape of the spin lattice.
    
    # Position neighbors right.
    i_r = i_list  
    j_r = list(map(lambda x:(x + 1) % Nj, j_list))   

    # Position neighbors left.
    i_l = i_list  
    j_l = list(map(lambda x:(x - 1) % Nj, j_list))   

    # Position neighbors up.
    i_u = list(map(lambda x:(x - 1) % Ni, i_list))  
    j_u = j_list  

    # Position neighbors down.
    i_d = list(map(lambda x:(x + 1) % Ni, i_list)) 
    j_d = j_list   

    # Spin values.
    sl_u = sl[i_u, j_u]
    sl_d = sl[i_d, j_d]
    sl_l = sl[i_l, j_l]
    sl_r = sl[i_r, j_r]

    return sl_u, sl_d, sl_l, sl_r

def energies_spins(i_list, j_list, sl, H, J):
    
    Function returning the energies of the states for the spins in given 
    positions in the spin lattice.
    
    Parameters
    ==========
    i_list : Spin position first indices.
    j_list : Spin position second indices.
    sl : Spin lattice.
    
    
    sl_u, sl_d, sl_l, sl_r = neighboring_spins(i_list, j_list, sl)
    
    sl_s = sl_u + sl_d + sl_l + sl_r 
    
    E_u = - H - J * sl_s
    E_d =   H + J * sl_s 
    
    return E_u, E_d


def probabilities_spins(i_list, j_list, sl, H, J, T):
    
    Function returning the energies of the states for the spins in given 
    positions in the spin lattice.
    
    Parameters
    ==========
    i_list : Spin position first indices.
    j_list : Spin position second indices.
    sl : Spin lattice.
    
    
    E_u, E_d = energies_spins(i_list, j_list, sl, H, J)
    
    Ei = np.array([E_u, E_d])
    
    Z = np.sum(np.exp(- Ei / T), axis=0)  # Partition function.
    pi = 1 / np.array([Z, Z]) * np.exp(- Ei / T)  # Probability.

    return pi, Z   

import random
import time
from tkinter import *

f = 0.05  # Number of randomly selected spins to flip-test.
N_skip = 10 # Visualize status every N_skip steps. 

window_size = 600

tk = Tk()
tk.geometry(f'{window_size + 20}x{window_size + 20}')
tk.configure(background='#000000')

canvas = Canvas(tk, background='#ECECEC')  # Generate animation window.
tk.attributes('-topmost', 0)
canvas.place(x=10, y=10, height=window_size, width=window_size)

Nspins = np.size(sl)  # Total number of spins in the spin lattice.
Ni, Nj = sl.shape

S = int(np.ceil(Nspins * f))  # Number of randomly selected spins.

step = 0

def stop_loop(event):
    global running
    running = False
tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.
running = True  # Flag to control the loop.
while running:
    ns = random.sample(range(Nspins), S)

    i_list = list(map(lambda x: x % Ni, ns)) 
    j_list = list(map(lambda x: x // Ni, ns)) 

    pi, Z = probabilities_spins(i_list, j_list, sl, H, J, T)

    rn = np.random.rand(S)
    for i in range(S):
        if rn[i] > pi[0, i]:
            sl[i_list[i], j_list[i]] = -1
        else:
            sl[i_list[i], j_list[i]] = 1

    # Update animation frame.
    if step % N_skip == 0:        
        canvas.delete('all')
        spins = []
        for i in range(Ni):
            for j in range(Nj):
                spin_color = '#FFFFFF' if sl[i,j] == 1 else '#000000'
                spins.append(
                    canvas.create_rectangle(
                        j / Nj * window_size, 
                        i / Ni * window_size,
                        (j + 1) / Nj * window_size, 
                        (i + 1) / Ni * window_size,
                        outline='', 
                        fill=spin_color,
                    )
                )
        
        tk.title(f'Iteration {step}')
        tk.update_idletasks()
        tk.update()
        time.sleep(0.1)  # Increase to slow down the simulation.

    step += 1

tk.update_idletasks()
tk.update()
tk.mainloop()  # Release animation handle (close window to finish).




#Where do I set the N1 and N2?
dhalf=3 #half the distance between the plates 3,5,7,10
s1=1
s2=-1

N1=N/2-dhalf
N2=N/2+dhalf
for row in N:
    sl[row][N1]=s1
    sl[row][N2]=s2






N=2 #What is this supposed to be ?
N_oth=101 #?
Eb = 2 #energy barrier, what is this supposed to be
xi = np.arange(N) 

Ei = np.array([0,Eb,0]) #Energies for the different states? Change since there should be more states?

pi, Z=probability_distribution(Ei)

#Plot probability distribution
plt.bar(xi, Ei, color='#FF8000', width=0.4, edgecolor='#A04000')
plt.title('Energy spectrum')
plt.xlabel('states')
plt.ylabel('Energy (kBT)')
plt.show()

plt.bar(xi, pi, color='c', width=0.4, edgecolor='k')
plt.title('Probability distribution')
plt.xlabel('states')
plt.ylabel('Probability')
plt.show()

def next_state_2(x,pi):
    if x not in (-1, 1):
        raise ValueError('x must be -1 or 1')
    
    p = np.random.rand()

    if p < pi[0]:
        x_next=-1
    else:
        x_next=1
    
    return x_next

N_steps=6000
x = np.zeros(N_steps); x[0]=0

"""






















import numpy as np

N = 100 #Size of the spin lattice
H = 0 #External field
J = 1 #spin-spin coupling
T = 2.3 #temperature

sl = 2 * np.random.randint(2, size=(N,N))-1

N_up = np.sum(sl + 1) / 2
N_down = N * N - N_up

print(f"Spin lattice created: N_up={N_up} N_down={N_down}")

def neighbouring_spins(i_list, j_list, sl):
    #returns position of the neighbouting spin of a list of spins
    #identified by their position in the spin lattice
    #i_list: spin position first indices, j_list= spin position second indicies, sl: spin lattie
    Ni, Nj = sl.shape # shape of the spin lattice, do I really need this?

    #position neighbours right
    i_r = i_list
    j_r = list(map(lambda x: (x+1) % Nj, j_list)) #Why the %

    #Position neighbours left
    i_l = i_list
    j_l = list(map(lambda x:(x - 1) % Nj, j_list))

    #Position neightbours up
    i_u = list(map(lambda x: (x - 1) % Ni, i_list))
    j_u = j_list

    #Position neighbours down
    i_d = list(map(lambda x:(x + 1) % Ni, i_list))
    j_d = j_list

    #Spin values
    print(sl.shape)
    print(f"i_u is {i_u} and j_u is {j_u}")
    sl_u = sl[i_u, j_u]
    sl_d = sl[i_d, j_d]
    sl_l = sl[i_l, j_l]
    sl_r = sl[i_r, j_r]

    return sl_u, sl_d, sl_l, sl_r

def energies_spins(i_list, j_list, sl, H, J):
    """
    Returns the energies of the states for spin in given position

    Parameters
    ==========
    i_list : Spin position first indices
    j_list : Spin position second indices
    sl : spin lattice
    """

    sl_u, sl_d, sl_l, sl_r = neighbouring_spins(i_list, j_list, sl)

    sl_s = sl_u + sl_d + sl_l + sl_r 

    E_u = - H - J * sl_s
    E_d = H + J * sl_s

    return E_u, E_d

def probabilities_spins(i_list, j_list, sl, H, J, T):
    """
    Energies of the states for the spins in a given position

    Parameters
    ==========
    i_list : spin position first indices
    j_list : spin position second indices
    sl : spin lattice
    """

    E_u, E_d = energies_spins(i_list, j_list, sl, H, J)

    Ei = np.array([E_u, E_d])

    Z = np.sum(np.exp(- Ei / T), axis=0) 
    pi = 1 / np.array([Z, Z]) * np.exp(- Ei / T)

    return pi, Z

#Each time step, we randomly choose S spins
#For each spin, we calculate the posisble energies of the configurations with spin up and down
#Based on these, we randomly draw the status of each spin in the next time step
import random
import time
#from tkinter import *

N_steps = 6000
f = 0.05 #Number of randomly selected spins to flip-test
#N_skip = 10 #Visualise status every N_skip steps 

#window_size = 600

#tk = Tk()
#tk.geometry(f'{window_size+20}x{window_size + 20}')
#tk.configure(background='#000000')

#canvas = Canvas(tk, background='#ECECEC')
#tk.attributes('-topmost', 0)
#canvas.place(x=10, y=10, height = window_size, width= window_size)

N_spins = np.size(sl) #tot numb of spins in spin lattice
Ni, Nj = sl.shape

S = int(np.ceil(N_spins * f)) #num of randomly selected spins

step = 0

#def stop_loop(event):
#    global running
#    running = False
#tk.bind("<Escape>", stop_loop) #Bind esccape key to stop loop
#running = True #Flag to control loop

def run(d_half):
    for i in range(N_steps): 
        sum=0
        for i in range(N-1):
            sum_loc=0
            for j in range(N-1):
                i_list = [i+1, i-1, i, i]
                j_list = [j, j, j+1, j-1]
                s_u, s_d, s_l, s_r = neighbouring_spins(i_list, j_list, sl)
                around_sl = s_u + s_d + s_l + s_r
                sum_loc += sl[i][j] * around_sl
            sum += sum_loc
        
        e_tot = -(J/(2 * N**2)) * sum
                
        ns = random.sample(range(N_spins), S)

        i_list = list(map(lambda x: x % Ni, ns))
        j_list = list(map(lambda x: x // Ni, ns))

        pi, Z = probabilities_spins(i_list, j_list, sl, H, J, T)

        rn = np.random.rand(S)
        for i in range(S):
            if rn[i] > pi[0,i]:
                sl[i_list[i], j_list[i]] = -1
            else:
                sl[i_list[i], j_list[i]] = 1
        
    return sl, e_tot
import matplotlib.pyplot as plt

d_half_list=[3, 5, 7, 10]
e_val_list=[]
plot_val_list=[]
for d_half in d_half_list:
    plot_val, e_val = run(d_half)
    e_val_list.append(e_val)
    plot_val_list.append(plot_val)
    plt.plot(plot_val)
    plt.show()
    


    
    #Update animation frame
    #if step % N_skip == 0:
    #    canvas.delete('all')
    #    spins = []
    #    for i in range(Ni):
    #        for j in range(Nj):
    #            spin_color = '#FFFFFF' if sl[i,j] == 1 else '#000000'
    #            spins.append(
    #                canvas.create_rectangle(
    #                    j / Nj * window_size, 
    #                    i / Ni * window_size,
    #                    (j + 1) / Nj * window_size, 
    #                    (i + 1) / Ni * window_size,
    #                    outline='', 
    #                    fill=spin_color,
    #                )
    #            )
    #    
    #    tk.title(f'Iteration {step}')
    #    tk.update_idletasks()
    #    tk.update()
    #    time.sleep(0.1)  # Increase to slow down the simulation.

    #step += 1

#tk.update_idletasks()
#tk.update()
#tk.mainloop() #Release animation handle (close window to finish) 




"""




H=0
J = 1
kB = 1
T = 2.3

#Initalise spins randomly with rqual probability for being +1 and -1:
sl = 2 * np.random.randint(2, size=(N, N)) - 1

#N_up = np.sum(sl + 1) / 2
#N_down = N * N - N_up

d_half_list=[3,5,7,10]



def for_different_d_half(d_half):
    N1=N/2-d_half
    N2=N/2+d_half
    s1=1
    s2=-1
    for row in range(N):
        sl[row][N1]=s1
        sl[row][N2]=s2
    
    for step in num_steps:
        for spin in sl:
            ns = random.sample(range(Nspins), S)

            i_list = list(map(lambda x: x % N, ns)) 
            j_list = list(map(lambda x: x // N, ns)) 
            pi, Z = probabilities_spins(i_list, j_list, sl, H, J, T) #return energies of states for a spin, i_list: spin position first indices, j_list: spin position second indices, sl: spin lattice
            #Z is partition function
            next_state_3(spin, pi)   #generates the next state given x: current state and pi: probability distribution


final_state=[]
e_tot=[]
for d_half in d_half_list:
    loc_final_state, loc_e_tot=for_different_d_half(d_half)
    final_state.append(loc_final_state)
    e_tot.append(loc_e_tot)
    


"""