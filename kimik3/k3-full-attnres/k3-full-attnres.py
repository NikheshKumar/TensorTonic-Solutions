import torch

def full_attention_residual(embedding, previous_outputs, pseudo_query, eps=1e-6):
    """
    Returns: retrieved representations and depth-attention weights.
    """

    embedding = torch.cat((embedding.unsqueeze(0), previous_outputs), dim=0)

    key_rms = embedding * torch.rsqrt(torch.mean(embedding**2, dim=-1, keepdim=True) + eps)

    logits = key_rms @ pseudo_query

    weights = torch.softmax(logits, dim=0)

    retrieved_rep = torch.sum(weights.unsqueeze(-1) * embedding, dim=0)

    return retrieved_rep, weights