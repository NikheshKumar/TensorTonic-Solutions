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
    
    Q = np.sum(P * (R + gamma * V), axis=2)
    V_new = np.sum(policy * Q, axis=1)


    return [round(v_new,4) for v_new in V_new]