import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    if not nonlinear:
        res = weight_matrix @ inputs + biases

    if nonlinear:
        a = weight_matrix @ inputs + biases
        res = torch.tanh(a)

    return res

    
