import torch

def sigmoid_router(x, gate_weight, top_k):
    """
    Returns: tuple (indices, weights) both of shape (batch, seq, top_k)
    """
    # YOUR CODE HERE
    scores = torch.sigmoid(x @ gate_weight.T)

    weights, indices = torch.topk(scores, top_k, dim=-1)

    weights = weights / torch.sum(weights, dim=-1, keepdim=True)

    return indices, weights