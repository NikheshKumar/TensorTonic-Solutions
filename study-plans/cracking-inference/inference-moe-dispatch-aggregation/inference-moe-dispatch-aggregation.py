import torch

def sparse_moe_forward(
    token_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_in: torch.Tensor,
    w_out: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (num_tokens, d_model)
    """
    values, indices = torch.sort(router_logits, dim=-1, stable=True, descending=True)

    values = values[...,:top_k]
    indices = indices[...,:top_k]

    weights = torch.softmax(values, dim=-1)
    
    num_tokens, d_model = token_states.shape

    idx_flat = indices.reshape(-1)                       
    x_rep    = token_states.repeat_interleave(top_k, dim=0) 

    ffn = torch.bmm((torch.relu(torch.bmm(x_rep.unsqueeze(1), w_in[idx_flat]).squeeze(1))).unsqueeze(1), w_out[idx_flat])

    ffn = ffn.squeeze(1).reshape(num_tokens, top_k, d_model)
    
    y = torch.sum(weights.unsqueeze(-1) * ffn, dim=1)

    return y
