import torch

def smebu_update(expert_bias, expert_load, target_load, momentum, lr, max_bias, eps=1e-8):
    """
    Returns: torch.Tensor of shape (num_experts,)
    """
    # YOUR CODE HERE
    imbalance = expert_load - target_load 

    gradient = imbalance / (torch.sum(expert_load, dim=-1) + eps)

    bias_new = momentum * expert_bias - lr * gradient

    bias_new = max_bias * torch.tanh(bias_new / max_bias)

    return bias_new