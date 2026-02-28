def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    import numpy as np 

    values = np.asarray(values)
    transitions = np.asarray(transitions)
    rewards = np.asarray(rewards)
    ans = []

    for s in range(len(transitions)):
      max_q = -np.inf
      q = rewards[s] + gamma*np.dot(transitions[s], values)
      max_q = np.max(q)
      ans.append(max_q)
    

    return ans