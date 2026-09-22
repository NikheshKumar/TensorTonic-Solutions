import torch

def glm_shared_plus_routed(
    x: torch.Tensor,
    routed_W_gate: torch.Tensor,
    routed_W_up: torch.Tensor,
    routed_W_down: torch.Tensor,
    shared_W_gate: torch.Tensor,
    shared_W_up: torch.Tensor,
    shared_W_down: torch.Tensor,
    top_indices: torch.Tensor,
    top_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Returns the float64 shared-plus-routed output with shape (n_tokens, hidden_size).
    """

    shared_exp = (torch.nn.functional.silu(x @ shared_W_gate) * (x @ shared_W_up)) @ shared_W_down

    k = top_indices.shape[1]

    x_expanded =  x.unsqueeze(1).expand(-1, k, -1) 

    routed_gate_out = torch.einsum('tkd,tkda->tka', x_expanded, routed_W_gate[top_indices])
    routed_gate_up = torch.einsum('tkd,tkda->tka', x_expanded, routed_W_up[top_indices])
    routed_hidden = torch.nn.functional.silu(routed_gate_out) * routed_gate_up

    routed_exp = torch.einsum('tka,tkad->tkd', routed_hidden, routed_W_down[top_indices])

    tot_routed_exp = (top_weights.unsqueeze(-1) * routed_exp).sum(dim=1) 
    
    y = shared_exp + tot_routed_exp

    return y