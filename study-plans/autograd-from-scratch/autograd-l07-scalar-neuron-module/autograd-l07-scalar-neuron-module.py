import torch

def scalar_neuron_module(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns a scalar tensor preserving the input dtype and device.
    """
    res = bias + torch.sum(weights * inputs)
    
    if nonlinear:
        res = torch.tanh(res)

    return res

        
