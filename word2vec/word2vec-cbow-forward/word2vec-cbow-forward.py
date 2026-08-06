import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the CBOW cross-entropy loss for predicting target_id from the averaged context.
    """
    # YOUR CODE HERE
    h = torch.mean(W_in[context_ids], dim=0)

    z = W_out @ h

    L = -F.log_softmax(z, dim=-1)

    return L[target_id]
