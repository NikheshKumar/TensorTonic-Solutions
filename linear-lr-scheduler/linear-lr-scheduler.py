def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    lr = float(initial_lr)

    if step >= total_steps:
            lr = float(final_lr)
            return lr 

    if warmup_steps > 0 and step < warmup_steps:
            lr = float(step*final_lr/warmup_steps)
            return lr

    if warmup_steps <= step <= total_steps:
            lr = final_lr + (initial_lr - final_lr) * ( (total_steps - step )/(total_steps - warmup_steps ) )
            lr = float(lr)
            return lr

         