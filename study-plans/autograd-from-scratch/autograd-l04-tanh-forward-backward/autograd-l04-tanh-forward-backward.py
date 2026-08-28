import torch

def tanh_forward_backward(x, upstream_gradient):
    """
    Returns: tanh output and its upstream-scaled input gradient
    """
    y = torch.tanh(x)

    grad = upstream_gradient * (1-y**2)

    return y, grad
