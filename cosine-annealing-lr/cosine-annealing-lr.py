def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Write code here

    import numpy as np 

    k = 1 + np.cos(np.pi*current_step / total_steps)

    lr = min_lr + (base_lr - min_lr) * (k) / 2

    return lr