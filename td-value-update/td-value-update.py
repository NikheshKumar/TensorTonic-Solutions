import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    V = np.asarray(V, float)

    delta = r + gamma*V[s_next] - V[s]
    V_new = V.copy()
    V_new[s] += alpha * delta

    return V_new.tolist()