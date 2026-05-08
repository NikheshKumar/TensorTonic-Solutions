def discounted_returns(rewards, gamma):
    """
    Returns: list of G_t values, one per timestep, each rounded to 4 decimals
    """
    import numpy as np 

    rewards = np.asarray(rewards, np.float64)
    T = len(rewards)

    if T == 0:
        return []

    G = [0.0]*T
    running = 0.0

    for t in reversed(range(T)):
        running = float(rewards[t]) + gamma*running
        G[t] = round(running,4)

    return G
