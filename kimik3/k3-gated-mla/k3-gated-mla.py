import math
import torch

def gated_mla(hidden_states, query_projection, latent_down_projection, key_up_projection, value_up_projection, output_gate_projection, output_projection, num_heads, causal=True):
    """
    Returns: gated attention outputs and the latent key-value cache.
    """
    B, seq_len, d_model = hidden_states.shape
    d_h = d_model // num_heads

    Q = (hidden_states @ query_projection.T).reshape(B, seq_len, num_heads, d_h).transpose(1, 2)
    C = hidden_states @ latent_down_projection.T
    K = (C @ key_up_projection.T).reshape(B, seq_len, num_heads, d_h).transpose(1, 2)
    V = (C @ value_up_projection.T).reshape(B, seq_len, num_heads, d_h).transpose(1, 2)

    
    scores = Q @ K.transpose(-2,-1) / (d_h**0.5)
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device), diagonal=1)
        scores = scores.masked_fill(mask, value=-float("inf"))

    weights = torch.softmax(scores, dim=-1)

    att = weights @ V

    context = att.transpose(1, 2).reshape(B, seq_len, d_model)

    Y = (torch.sigmoid(hidden_states @ output_gate_projection.T) * context) @ output_projection.T

    return Y, C