import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    q_values = np.asarray(q_values, float)
    
    if rng is None:
        rng = np.random.default_rng()

    if rng.random() < epsilon:
        a = rng.integers(0, len(q_values))
    else:
        a = np.argmax(q_values)
        

    return int(a)
        

    
    
