import torch

def sparse_moe_ffn(x: torch.Tensor, router_weight: torch.Tensor, expert_bias: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor, shared_W_gate: torch.Tensor, shared_W_up: torch.Tensor, shared_W_down: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Returns shared-plus-routed expert outputs with the same shape as x.
    """
    logits = x @ router_weight.T
    scores = torch.sigmoid(logits)
    biased_scores = scores + expert_bias

    _, indices = torch.topk(biased_scores, top_k, dim=-1)

    selected_scores = torch.gather(scores, dim=-1, index=indices)

    weights = selected_scores / torch.sum(selected_scores, dim=-1, keepdim=True)

    gate = torch.einsum('bsd,efd->bsef', x, W_gate)
    up = torch.einsum('bsd,efd->bsef', x, W_up)
    hidden = torch.nn.functional.silu(gate) * up
    e = torch.einsum('bsef,edf->bsed', hidden, W_down)

    indices = indices.unsqueeze(-1).expand(-1, -1, -1, e.shape[-1])
    selected_indices = torch.gather(e, dim=2, index=indices)
    y_routed = (weights.unsqueeze(-1) * selected_indices).sum(dim=-2)

    y_shared = (torch.nn.functional.silu(x @ shared_W_gate.T) * (x @ shared_W_up.T)) @ shared_W_down.T

    out = y_shared + y_routed

    return out

