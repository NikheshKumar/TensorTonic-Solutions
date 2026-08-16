import torch

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    uniform_draws: torch.Tensor,
) -> torch.Tensor:
    """
    Returns: sampled token id tensor of shape (batch,), dtype torch.int64
    """

    B, d = logits.shape

    if temperature == 0.0:
        return torch.argmax(logits, dim=-1).to(torch.int64)

    scaled_logits = logits / temperature

    if d > top_k > 0.0 :

        values, i = torch.topk(scaled_logits, top_k, dim=-1)

        logits_new = torch.full_like(scaled_logits,-float("inf"))

        scaled_logits= logits_new.scatter_(-1,i,values)

    
    p = torch.softmax(scaled_logits, dim=-1)
    
    if 0.0 < top_p < 1.0 :

        p_new, i = torch.sort(p,dim=-1,descending=True)

        cum_p = torch.cumsum(p_new,dim=-1)

        more_than_top_p = cum_p > top_p
        more_than_top_p[..., 1:] = more_than_top_p[...,:-1].clone()
        more_than_top_p[..., 0] = False

        p_new = p_new.masked_fill(more_than_top_p,value=0.0)

        q = torch.zeros_like(logits, dtype=p.dtype)
        q = q.scatter_(-1, i, p_new)
        q = q / torch.sum(q, dim=-1, keepdim=True)
        p = q
        
    p = p / torch.sum(p, dim=-1, keepdim=True)

    F = torch.cumsum(p, dim=-1)

    token = torch.searchsorted(F, uniform_draws.unsqueeze(-1), right=True).squeeze(-1)

    token = torch.clamp(token, max=d-1).to(torch.int64)
    
    return token
