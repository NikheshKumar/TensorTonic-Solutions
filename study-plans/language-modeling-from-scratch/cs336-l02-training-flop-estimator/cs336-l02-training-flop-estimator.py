def flop_estimator(matmuls, attention_flops=0):
    """
    Returns: dictionary containing exact forward, backward, and total FLOP counts
    """
    import math
    f_forward = sum(2 * math.prod(row) for row in matmuls) + attention_flops

    f_backward = 2 * f_forward

    f_total = f_forward + f_backward

    return {"forward_flops":f_forward,"backward_flops":f_backward,"total_flops":f_total}
