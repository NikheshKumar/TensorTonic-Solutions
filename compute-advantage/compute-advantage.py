import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    arr = np.zeros(len(states))
    g = 0
    
    for i in range(len(states)-1,-1,-1):
        g = gamma*g + rewards[i] 
        a = g - V[states[i]] 
        arr[i] = a

    return arr
