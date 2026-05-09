def bellman_optimality_backup(P, R, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    import numpy as np 

    P = np.asarray(P)
    R = np.asarray(R)
    V = np.asarray(V)

    S = len(V)

    Q = np.sum(P*(R + gamma * V), axis=2)

    V_new = np.max(Q, axis=1)

    return [round(v,4) for v in V_new]
    

    
