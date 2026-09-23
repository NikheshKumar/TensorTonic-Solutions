import torch

def gpto_moe_forward(
    x: torch.Tensor,
    gate_W: torch.Tensor,
    gate_b: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    W3: torch.Tensor,
    b3: torch.Tensor,
    top_k: int,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """
    Returns sparse MoE outputs with the same shape as x.
    """
    logits = x @ gate_W + gate_b

    vals, idx = torch.topk(logits, top_k, dim=-1)

    top_w = torch.softmax(vals, dim=-1)

    W1_sel = W1[idx]
    b1_sel = b1[idx]
    W2_sel = W2[idx]
    b2_sel = b2[idx]
    W3_sel = W3[idx]
    b3_sel = b3[idx]


    gate = (torch.einsum('nh,nkha->nka', x, W1_sel) + b1_sel).clamp(max=swiglu_limit)   
    up = (torch.einsum('nh,nkha->nka', x, W2_sel) + b2_sel).clamp(min=-swiglu_limit, max=swiglu_limit)   

    alpha = 1.702

    g = gate * torch.sigmoid(alpha * gate) * (up+1.0)

    y = torch.einsum('nka,nkah->nkh', g, W3_sel) + b3_sel

    y = torch.sum(top_w.unsqueeze(-1) * y, dim=1)

    return y

