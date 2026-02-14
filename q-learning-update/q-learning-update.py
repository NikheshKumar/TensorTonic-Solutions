import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Write code here
    Q = np.asarray(Q, float)

    Q_new = Q.copy()

    td = r + gamma* np.max( Q[s_next] )
    Q_new[s,a] = Q[s][a] + alpha*( td - Q[s][a] )

    return Q_new