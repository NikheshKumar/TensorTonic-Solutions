def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Write code here
    import numpy as np 

    term = np.pi * current_step / total_steps

    lr = min_lr + 0.5*(base_lr - min_lr)*(1 + np.cos(term))

    return lr

    