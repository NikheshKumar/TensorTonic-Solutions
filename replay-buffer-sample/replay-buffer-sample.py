def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    # Write code here

    import numpy as np 

    rs = np.random.RandomState(seed)

    indices = rs.choice(len(buffer), batch_size, replace=False)

    ans = [buffer[i] for i in indices]
    

    return ans  





