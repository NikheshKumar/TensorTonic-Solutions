import math

def warmup_cosine_schedule(base_lr, warmup_steps, total_steps):
    """
    Returns: list of learning rates
    """
    learning_rates = []

    for i in range(total_steps):
        if i < warmup_steps:
            lr = base_lr * (i+1) / warmup_steps
        else:
            num = i - warmup_steps
            den = total_steps - warmup_steps
            lr = 0.5 * base_lr * (1+math.cos(math.pi*num/den))  

        learning_rates.append(lr)

    return learning_rates