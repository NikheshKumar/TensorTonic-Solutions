def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Write code here
    import numpy as np 

    q_table = np.asarray(q_table, dtype=float)
    Q = q_table.copy()

    td = reward + gamma*q_table[next_state][next_action] - q_table[state][action]

    Q[state][action] += alpha*td

    return Q.tolist()