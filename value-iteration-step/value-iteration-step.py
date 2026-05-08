def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)
    transitions = np.asarray(transitions)
    rewards = np.asarray(rewards)

    q = rewards + gamma * np.dot(transitions, values) 
    ans = np.max(q, axis=1)

    return ans.tolist()