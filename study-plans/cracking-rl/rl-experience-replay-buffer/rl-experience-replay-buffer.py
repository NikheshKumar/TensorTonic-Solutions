def replay_buffer_sample(capacity: int, transitions: list, sample_indices: list) -> tuple:
    """
    Returns (states, actions, rewards, next_states, dones), each a list in sample order.
    """
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    buffer = [None] * capacity
    for j, tr in enumerate(transitions):
        buffer[j % capacity] = tr

    for i in sample_indices:
        s,a,r,s_next,d = buffer[i]

        states.append(s)
        actions.append(a)
        rewards.append(float(r))
        next_states.append(s_next)
        dones.append(d)


    return states, actions, rewards, next_states, dones