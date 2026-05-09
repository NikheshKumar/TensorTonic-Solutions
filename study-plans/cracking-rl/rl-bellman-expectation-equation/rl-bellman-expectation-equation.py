def bellman_expectation_backup(P, R, policy, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    import numpy as np 

    P = np.asarray(P, np.float64)
    R = np.asarray(R, np.float64)
    V = np.asarray(V, np.float64)
    
    S = len(V)
    A = len(policy[0])
    
    V_new = np.zeros(S, np.float64)
    
    for s in range(S):
        running = 0.0
        for a in range(A):
            for sp in range(S):
                running += policy[s][a] * P[s][a][sp] * ( R[s][a][sp] + gamma*V[sp])
                V_new[s] = round(running, 4)


    return V_new