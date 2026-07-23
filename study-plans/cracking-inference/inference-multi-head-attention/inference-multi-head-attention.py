import torch

def multi_head_attention(
    hidden_states: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: output tensor of shape (batch, seq, d_model)
    """
    B, S_q, d_model = hidden_states.shape
    d_k = d_model // num_heads
    
    q = (hidden_states @ w_q).reshape(B, S_q, num_heads, d_k).transpose(1, 2)
    k = (hidden_states @ w_k).reshape(B, S_q, num_heads, d_k).transpose(1, 2)
    v = (hidden_states @ w_v).reshape(B, S_q, num_heads, d_k).transpose(1, 2)

    scores = q @ k.transpose(-2,-1) / (d_k**0.5)

    if causal is True:
        mask = torch.triu(torch.ones(S_q, S_q, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    weights = torch.softmax(scores, dim=-1)

    att = weights @ v

    concat_head = att.transpose(1,2).contiguous().reshape(B, S_q, d_model)

    mha = concat_head @ w_o

    return mha
