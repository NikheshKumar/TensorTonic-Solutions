import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    # YOUR CODE HERE
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()
    
    vc = W_in[center_id]
    uo = W_out[pos_id]
    un = W_out[neg_ids]

    so = torch.dot(vc, uo)
    sn = torch.sum(vc * un, dim=-1)

    grad_uo = (torch.sigmoid(so)-1.0)*vc 
    grad_un = torch.sigmoid(sn).unsqueeze(-1)*vc
    grad_vc = (torch.sigmoid(so)-1.0)*uo + torch.sum((torch.sigmoid(sn).unsqueeze(-1))*un, dim=0)

    W_in_updated[center_id] -= lr * grad_vc
    W_out_updated[pos_id] -= lr * grad_uo

    for i, ni in enumerate(neg_ids):
        W_out_updated[ni] -= lr * grad_un[i]


    return (W_in_updated, W_out_updated)
