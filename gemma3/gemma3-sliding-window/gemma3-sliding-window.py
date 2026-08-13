import torch

def sliding_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, heads, seq_len, d_head)
    """
    # YOUR CODE HERE


    B, heads, S_q, d_head = Q.shape
    _, _, S_k, _ = K.shape


    scores = Q @ K.transpose(-2,-1) / (d_head**0.5)

    i = torch.arange(S_q, device=Q.device).unsqueeze(1)
    j = torch.arange(S_k, device=Q.device).unsqueeze(0)

    out_of_bounds_mask = (i-j < 0) | (i-j > window_size)

    scores = scores.masked_fill(out_of_bounds_mask, float('-inf'))
    
    weights = torch.nn.functional.softmax(scores, dim=-1)

    weights = torch.nan_to_num(weights, nan=0.0)

    att = weights @ V

    return att

    