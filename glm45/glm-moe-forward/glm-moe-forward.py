import torch
import torch.nn.functional as F

def glm_moe_forward(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    rW_gate: torch.Tensor,
    rW_up: torch.Tensor,
    rW_down: torch.Tensor,
    sW_gate: torch.Tensor,
    sW_up: torch.Tensor,
    sW_down: torch.Tensor,
    n_routed_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
) -> torch.Tensor:
    """
    Returns the float64 MoE output with shape (n_tokens, hidden_size).
    """
    x = x.to(torch.float64)
    gate_W  = gate_W.to(torch.float64)
    rW_gate = rW_gate.to(torch.float64)
    rW_up = rW_up.to(torch.float64)
    rW_down = rW_down.to(torch.float64)
    sW_gate = sW_gate.to(torch.float64)
    sW_up = sW_up.to(torch.float64)
    sW_down = sW_down.to(torch.float64)

    N, H = x.shape
    E, G, K, K_G = n_routed_experts, n_group, num_experts_per_tok, topk_group
    E_hat = E // G
    
    logits = x @ gate_W

    s = torch.sigmoid(logits)
    
    s_groups = s.reshape(N, G, E_hat)
    gs, _ = torch.topk(s_groups, k=min(2, E_hat), dim=-1)
    gs = torch.sum(gs, dim=-1)

    _, tg = torch.topk(gs, K_G, dim=-1)       
    group_mask = torch.zeros((N, G,), dtype=s.dtype, device=s.device)
    group_mask.scatter_(1, tg, 1.0)              
    expert_mask = group_mask.repeat_interleave(E_hat, dim=-1) 

    s_masked = s * expert_mask
    w, idx = torch.topk(s_masked, K, dim=-1)

    if norm_topk_prob:
        w = w / torch.sum(w, dim=-1, keepdim=True)
    w = w * routed_scaling_factor

    expert_output = torch.zeros((N, H,), dtype=x.dtype, device=x.device)

    for i in range(E):
        mask = (idx == i)                              
        w_e = (w * mask).sum(dim=-1)     
        h = (F.silu(x @ rW_gate[i]) * (x @ rW_up[i])) @ rW_down[i]          
        expert_output += w_e.unsqueeze(-1) * h 

    shared_output = (F.silu(x @ sW_gate) * (x @ sW_up)) @ sW_down 

    out = expert_output + shared_output

    return out.to(torch.float64)





    

   