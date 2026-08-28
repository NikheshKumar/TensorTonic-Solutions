import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    a = torch.dot(inputs, weights) + bias

    output = torch.tanh(a)

    delta = upstream_gradient * (1- output**2)

    grad_Lx = delta * weights

    grad_Lw = delta * inputs

    grad_Lb = delta

    return output, grad_Lx, grad_Lw, grad_Lb
