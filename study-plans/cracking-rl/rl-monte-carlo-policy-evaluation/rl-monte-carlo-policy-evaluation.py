def mc_policy_evaluation(episodes: list, gamma: float, num_states: int) -> list:
    """
    Returns first-visit state values rounded to four decimals; unvisited states are 0.0.
    """
    
    counts = [0] * num_states
    sums = [0] * num_states

    for ep in episodes:

        # compute G
        T = len(ep)
        G = [0.0]*T
        g = 0.0
        for t in reversed(range(T)):
            state, reward = ep[t]
            g = reward + gamma * g
            G[t] = g

        # compute first visit only values and sum them up
        unique = set()
        for t, (st, re) in enumerate(ep):
            if st in unique:
                continue 
            else:
                unique.add(st)
                sums[st] += G[t]
                counts[st] += 1


    V = [sums[i] / counts[i] if counts[i]>0.0 else 0.0 for i in range(num_states)]

    return [round(float(v), 4) for v in V]
            
        
        

    
    