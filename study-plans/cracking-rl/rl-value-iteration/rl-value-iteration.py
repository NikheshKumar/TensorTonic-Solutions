def value_iteration(P, R, gamma, tol=1e-6, max_iters=1000):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    import numpy as np 


    P = np.asarray(P)
    R = np.asarray(R)

    S = len(P)
    A = len(P[0])
    
    V = np.zeros((S,), np.float64)

    for i in range(max_iters):
        
        Q = np.sum(P * (R + gamma * V), axis=2)
        V_new = np.max(Q, axis=1)
        
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        
        V = V_new
            

    Q_final = np.sum(P * (R + gamma * V), axis=2)
    policy = np.argmax(Q_final, axis=1)

    V_final = [round(v,4) for v in V]
    policy = policy.astype(int).tolist()

    return V_final, policy
        
         
        
