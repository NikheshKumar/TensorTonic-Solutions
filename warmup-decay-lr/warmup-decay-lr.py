def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    """
    Compute the learning rate at a given step using warmup + linear decay.
    """
    # Write code here

    import numpy as np 

    if current_step < warmup_steps:
        lr = base_lr * current_step / warmup_steps
        return lr

    if current_step >= warmup_steps:
        num = total_steps - current_step
        den = total_steps - warmup_steps
        lr = base_lr * (num) / (den)
        return lr