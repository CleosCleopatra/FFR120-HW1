
import numpy as np
np.random.seed(5)

#Game of life

def neighbours_Moore(status): 
    """
    Function to return the number of neighbours for each cell in status
    
    Parameters
    ==========
    status : current status
    """
    #10 December to 7e April

    # Initialize the neighbor count array
    n_nn = (
        np.roll(status, 1, axis=0) +  # Up.
        np.roll(status, -1, axis=0) +  # Down.
        np.roll(status, 1, axis=1) +  # Left.
        np.roll(status, -1, axis=1) +  # Right.
        np.roll(np.roll(status, 1, axis=0), 1, axis=1) +  # Up-Left.
        np.roll(np.roll(status, 1, axis=0), -1, axis=1) +  # Up-Right
        np.roll(np.roll(status, -1, axis=0), 1, axis=1) +  # Down-Left
        np.roll(np.roll(status, -1, axis=0), -1, axis=1)  # Down-Right
    )

    return n_nn

#Function to apply 2-d rule
def apply_rule_2d(rule_2d, status):
    """
    Function to apply a 2-d rule on a status. Return the next status.
    
    Parameters
    ==========
    rule_2d : Array with size [2, 9]. Describe the CA rule.
    status : Current status.
    """
    
    Ni, Nj = status.shape  # Dimensions of 2-D lattice of the CA.
    next_status = np.zeros([Ni, Nj]) 
    
    # Find the number of neighbors.
    n_nn = neighbours_Moore(status) 
    for i in range(Ni):
        for j in range(Nj):
            next_status[i, j] = rule_2d[int(status[i, j]), int(n_nn[i, j])]
        
    return next_status

#Visualisation
N = 100
#gol = np.zeros([N, N])
p_list = [0.45, 0.48, 0.50, 0.52, 0.55]

#Oscillator: The toad
#gol[N // 2 - 3:N // 2 + 3, 
#    N // 2 - 3:N // 2 + 3] = [
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 1, 1, 1, 0],
#        [0, 1, 1, 1, 0, 0],
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0]
#    ]

# Glider.
#gol[N // 2 - 3:N // 2 + 3, 
#    N // 2 - 3:N // 2 + 3] = [
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1],
#        [0, 0, 0, 1, 0, 0],
#        [0, 0, 0, 0, 1, 0]
#    ]

#Random initial state
#gol = np.random.randint(2, size=[N, N])

rule_2d = np.zeros([2, 9])

#Game of Lifes rules
#rule_2d[0, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0] #New born from empty cell
#rule_2d[1, :] = [0, 0, 1, 1, 0, 0, 0, 0, 0] #Survival of living cell

# Majority Rules.
rule_2d[0, :] = [0, 0, 0, 0, 0, 0, 1, 1, 1]
rule_2d[1, :] = [0, 0, 0, 0, 0, 1, 1, 1, 1]


import matplotlib.pyplot as plt

num_1_votes_list = []
gol_list = []

fig, axs = plt.subplots(1, len(p_list))

for idx, p in enumerate(p_list):
    gol = np.random.rand(N, N) < p
    gol = gol.astype(int)
    #print(gol)
    Ni, Nj = gol.shape #Sets the variable desicription the shape
    running = True
    print(p)
    rounds_since_imporvement = 0
    old_num_1_votes = np.count_nonzero(gol==1)

    local_imprt = []
    while rounds_since_imporvement < 10: 
        gol = apply_rule_2d(rule_2d, gol)
    
        num_1_votes = np.count_nonzero(gol == 1)
        #print(f"rounds since imporvement {rounds_since_imporvement},  nuM_1_votes {num_1_votes}")
        if num_1_votes != old_num_1_votes:
            #print("here")
            rounds_since_imporvement = 0
        else: 
            rounds_since_imporvement +=1
        local_imprt.append(num_1_votes)
        old_num_1_votes = num_1_votes
    #print(num_1_votes)

    gol_list.append(gol)

    num_1_votes_list.append(num_1_votes)

    axs[idx].imshow(gol, cmap = 'binary')
    axs[idx].set_title(f'p={p}, V1={num_1_votes}', fontsize=5)

print(num_1_votes_list)
plt.tight_layout()
plt.show()

plt.plot(p_list, num_1_votes_list)
plt.show()