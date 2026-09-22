import torch

def glm_route(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing top_indices and top_weights tensors.
    """
    logits = x @ gate_W
    s_raw = torch.sigmoid(logits)

    s_sel = s_raw + e_score_correction_bias
    
    s_groups = s_sel.reshape(x.shape[0], n_group, n_routed_experts // n_group)

    vals, _ = torch.topk(s_groups, 2, dim=-1)

    group_scores = torch.sum(vals, dim=-1)

    group_weights, group_indices = torch.topk(group_scores, topk_group, dim=-1) 

    mask = torch.zeros_like(group_scores)

    mask.scatter_(1, group_indices, 1.0)
    
    expert_mask = (mask.unsqueeze(-1).expand(x.shape[0], n_group, n_routed_experts // n_group).reshape(x.shape[0], n_routed_experts)).bool()

    s_masked = s_sel.masked_fill(~expert_mask, value=-float("inf"))

    _, top_indices = torch.topk(s_masked, num_experts_per_tok, dim=-1)

    w = s_raw.gather(1, top_indices)   

    if norm_topk_prob:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)

    w_out = w * routed_scaling_factor

    return {"top_indices":top_indices, "top_weights":w_out}
        
