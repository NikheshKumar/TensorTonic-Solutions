import numpy as np
from collections import defaultdict

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # Write code here
    episodes = np.asarray(episodes)
    s = np.zeros(n_states)
    c = np.zeros(n_states)

    for epi in episodes:
      visited = set()
      g = 0 

      ret = np.zeros(len(epi))
      for i in range(len(epi) - 1, -1, -1):
          g = epi[i][1] + gamma * g
          ret[i] = g

      for i in range(len(epi)):
        if epi[i][0] not in visited:
          s[epi[i][0]] += ret[i]
          c[epi[i][0]] += 1
          visited.add(epi[i][0])
          
      
    res = np.divide(s, c, where=c!=0)

    return res.tolist()
