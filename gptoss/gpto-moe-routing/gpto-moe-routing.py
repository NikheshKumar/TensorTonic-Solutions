import torch

def moe_route(x: torch.Tensor, gate_W: torch.Tensor, gate_b: torch.Tensor, top_k: int):
    """Returns: tuple (top_indices: LongTensor (n_tokens, top_k), top_weights: FloatTensor (n_tokens, top_k))."""
    # YOUR CODE HERE
    gate = x @ gate_W + gate_b

    values, indices = torch.topk(gate, top_k, dim=-1)

    weights = torch.softmax(values, dim=-1)

    return (indices.to(torch.int64), weights)
