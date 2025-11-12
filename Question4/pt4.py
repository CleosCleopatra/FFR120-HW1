#Function to implement a 1-d cellular automation
import numpy as np
"""
def apply_rule_1d(n_rule, status):
    
    Function to apply a rule on a status. Return the next status-
    
    Parameters
    ==========
    n_rule: Code of the rule
    status: Current status
    

    N = len(status) #Dimensions of 1-d lattice of the CA
    next_status = np.zeros(N) #Same dimesnion lattice for next status

    rules = np.zeros(8) + n_rule; 
    q = n_rule
    for i in range(8):
        rules[i] = q % 2
        q = (q - q % 2) // 2
    
    #Use peridoic boundary conditions
    for i in range(N):
        c = status[(i - 1) % N] * 4 + status[i] * 2 + status[(i+1) % N] * 1 
        next_status[i] = rules[int(c)]
    
    return next_status
"""
"""
#Simulation
from IPython.display import clear_output
from matplotlib import pyplot as plt

n_rule = 30 # Try: 184, 90, 30, 110, 165.
N = 200  # Lattice dimension.
T = 100  # Iterations.


# Status with all zeros and a single 1 in the middle.
status = np.zeros(N)  # initialize status
status[N // 2] = 1

# Random status.
# status = np.random.randint(2, size=N)

sequence = np.zeros([T, N])
sequence[0, :] = status
for t in range(T):
    
    plt.figure(figsize=(10, 5))
    plt.imshow(sequence)
    plt.xlabel('space')
    plt.ylabel('time')
    plt.show()
    
    if input("Press Enter to continue to the next image...") == 'x':
        break
    elif t == T - 1:
        break
    else:
        clear_output(wait=True)  # Clear previous output
    
    new_status = apply_rule_1d(n_rule, status)
    sequence[t + 1, :] = new_status
    status = new_status

"""


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
gol = np.zeros([N, N])
p_list = [0.45, 0.48, 0.50, 0.52, 0.55]

#Oscillator: The toad
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ]

# Glider.
gol[N // 2 - 3:N // 2 + 3, 
    N // 2 - 3:N // 2 + 3] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ]

#Random initial state
gol = np.random.randint(2, size=[N, N])

rule_2d = np.zeros([2, 9])

#Game of Lifes rules
#rule_2d[0, :] = [0, 0, 0, 1, 0, 0, 0, 0, 0] #New born from empty cell
#rule_2d[1, :] = [0, 0, 1, 1, 0, 0, 0, 0, 0] #Survival of living cell

# Majority Rules.
rule_2d[0, :] = [0, 0, 0, 0, 0, 1, 1, 1, 1]
rule_2d[1, :] = [0, 0, 0, 0, 1, 1, 1, 1, 1]

Ni, Nj = gol.shape #Sets the variable desicription the shape

num_1_votes_list = []

for p in p_list:
    running = True
    print(p)
    rounds_since_imporvement = 0
    old_num_1_votes = np.count_nonzero(gol==1)

    local_imprt = []
    while rounds_since_imporvement < 5: 
        gol = apply_rule_2d(rule_2d, gol)
    
        num_1_votes = np.count_nonzero(gol == 1)
        print(f"rounds since imporvement {rounds_since_imporvement},  nuM_1_votes {num_1_votes}")
        if num_1_votes != old_num_1_votes:
            print("here")
            rounds_since_imporvement = 0
        else: 
            rounds_since_imporvement +=1
        local_imprt.append(num_1_votes)
        old_num_1_votes = num_1_votes
    print(num_1_votes)

    

    num_1_votes_list.append(num_1_votes)

print(num_1_votes_list)