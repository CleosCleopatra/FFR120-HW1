



import random
import numpy as np

random.seed(5)
np.random.seed(5)

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
    #print(sl.shape)
    #print(f"i_u is {i_u} and j_u is {j_u}")
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

N_steps = 6000 #Swirch back to 6000 ?
f = 0.05 #Number of randomly selected spins to flip-test

N_spins = np.size(sl) #tot numb of spins in spin lattice
Ni, Nj = sl.shape

S = int(np.ceil(N_spins * f)) #num of randomly selected spins

step = 0

s1 = 1
s2 = 1
def run(d_half):
    sl_loc = sl.copy()
    N1 = int(N/2 - d_half)
    N2 = int(N / 2 + d_half)
    sl_loc[N1, :] = s1
    sl_loc[N2, :] = s2

    e_in_end=0
    e_vals = []  
    
    for i in range(N_steps): 
        print(i)
        i_list = list(range(N))
        j_list = list(range(N))

        s_u = np.roll(sl_loc, -1, axis=0)
        s_d = np.roll(sl_loc, 1, axis=0)
        s_l = np.roll(sl_loc, -1, axis=1)
        s_r = np.roll(sl_loc, 1, axis=1)
        #s_u, s_d, s_l, s_r = neighbouring_spins(i_list, j_list, sl_loc)
        #summan = [s_u[i][j]+s_d[i][j]+s_l[i][j]+s_r[i][j] for i in range(N) for j in range(N)]
        sum_val = np.sum(sl_loc * (s_u + s_d + s_l + s_r))
        #for i in range(N):
        #    sum_loc=0
        #    i_list = [i]
        #    j_list = [j in range(N)]
        #    s_u, s_d, s_l, s_r = neighbouring_spins(i_list, j_list, sl_loc)
        #    sum = (int )
        #    for j in range(N):
        #        i_list = [i]
        #        j_list = [j]
        #        s_u, s_d, s_l, s_r = neighbouring_spins(i_list, j_list, sl_loc)
        #        around_sl = int(s_u) + int(s_d) + int(s_l) + int(s_r)
        #        sum_loc += sl_loc[i][j] * around_sl
        #        #print(f"sum_loc is {sum_loc} and sum is {sum}")
        #    sum += sum_loc
        
        
        
        e_tot = -(J/(2 * N**2)) * sum_val
        #print(f"e_tot is {e_tot} and e_vals is {e_vals}, sum is {sum}")
        e_vals.append(e_tot)
                
        ns = random.sample(range(N_spins), S)

        i_list = list(map(lambda x: x % N, ns)) #column????? remove
        j_list = list(map(lambda x: x // N, ns)) #row????? remove

        pi, Z = probabilities_spins(i_list, j_list, sl_loc, H, J, T)

        rn = np.random.rand(S)
        for j in range(S):
            if i_list[j] == N1 or i_list[j] == N2:
                continue
            elif rn[j] > pi[0,j]:
                sl_loc[i_list[j], j_list[j]] = -1 #Is it supposed to be -1 here?
            else:
                sl_loc[i_list[j], j_list[j]] = 1
        if i>=N_steps-(N_steps//10):
            e_in_end += e_tot
    
    e_in_end /= (N_steps//10)
        
    return sl_loc, e_vals, e_in_end
import matplotlib.pyplot as plt

d_half_list=[3, 5, 7, 10]
e_val_list=[]
plot_val_list=[]

def check_fixed_rows(sl_loc, N1, N2, s1=1, s2=1):
    row1_ok=np.all(sl_loc[N1, :]==s1)
    row2_ok=np.all(sl_loc[N2, :] ==  s2)

    print(f"row N1={N1} correct? {row1_ok}")
    print(f"row N2= {N2} correct? {row2_ok}")

for idx, d_half in enumerate(d_half_list):
    plot_val, e_vals, e_in_end = run(d_half)
    N1 = int(N/2 - d_half)
    N2 = int(N/2 + d_half)
    check_fixed_rows(plot_val, N1, N2)

    e_val_list.append(e_vals)
    plot_val_list.append(plot_val)
    #print(f"Here i is {i}")
    plt.figure(1)
    plt.subplot(2,2,idx+1)
    plt.imshow(plot_val, cmap = 'Greys', vmin=-1, vmax=1)
    plt.axhline(y=N1, color='red', linewidth=0.5)
    plt.axhline(y=N2, color='red', linewidth=0.5)

    plt.figure(2)
    plt.subplot(2,2, idx+1)
    #print(e_val)
    #print(N)
    #print(f"len of e_val is {len(e_val)} and N is {N_steps}")
    plt.plot(range(N_steps), e_vals)

    plt.text(
        0.95, 0.95,
        f'e_eq = {e_in_end: .3f}',
        transform=plt.gca().transAxes,
        ha='right', va='top'
    )

    
plt.show()


