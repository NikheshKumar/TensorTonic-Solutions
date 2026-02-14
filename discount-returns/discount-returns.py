def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # Write code here
    import numpy as np 

    rewards = np.asarray(rewards)

    T = len(rewards)

    G = np.zeros(T)

    G[T-1] = rewards[T-1]

    for i in range(T-2, -1, -1):
        G[i] = rewards[i] + gamma*G[i+1]

    return G.tolist()    



    