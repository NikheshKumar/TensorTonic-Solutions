import torch

def embedding_head(token_states: torch.Tensor, attention_mask: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """
    Returns the unit-norm embeddings with shape (B, d).
    """
    mask = attention_mask.to(token_states.dtype)[:,:, None]
    pooled = torch.sum(token_states * mask, dim=1) / torch.sum(mask, dim=1)

    embedding = pooled @ projection.T

    embedding *= torch.rsqrt(torch.sum(embedding**2, dim=-1, keepdim=True))

    return embedding
