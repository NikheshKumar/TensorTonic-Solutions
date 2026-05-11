def policy_iteration(P, R, gamma, eval_tol=1e-8, max_iters=200):
    """
    Returns: tuple (V, policy) where V is a list of S floats rounded to 4 decimals and policy is a list of S integer action indices
    """
    import numpy as np

    P = np.asarray(P, np.float64)
    R = np.asarray(R, np.float64)

    S = len(P)
    A = len(P[0])
    policy = np.zeros((S,), int)

    V = np.zeros((S,), np.float64)

    for i in range(max_iters):

        while True:

            P_new = P[np.arange(S), policy]
            R_new = R[np.arange(S), policy]
    
            V_new = np.sum(P_new * (R_new + gamma * V), axis=1)
    
            if np.max(abs(V_new-V)) < eval_tol:
                V = V_new
                break
    
            V = V_new
        

        Q = np.sum(P * (R + gamma * V), axis=2)
        new_policy = np.argmax(Q, axis=1)

        if (new_policy==policy).all():
            break

        policy = new_policy

    V_final = [round(v,4) for v in V]
    policy = policy.tolist()

    return V_final, policy
        