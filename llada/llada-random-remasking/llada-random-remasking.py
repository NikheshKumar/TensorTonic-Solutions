import torch

def llada_random_remasking(step_logits: torch.Tensor, uniforms: torch.Tensor, mask_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the final response and the full trajectory of response states.
    """
    N, L, V = step_logits.shape
    
    r = torch.full((N + 1, L), mask_id, dtype=torch.long, device=step_logits.device)
    
    for k in range(N):
        t = (N-k)/N 

        s = (N-k-1)/N 

        rho = s/t 
        
        for i in range(L):
            if r[k][i] != mask_id:
                r[k+1][i] = r[k][i]
            elif r[k][i] == mask_id and uniforms[k][i] < rho:
                r[k+1][i] = mask_id
            elif r[k][i] == mask_id and uniforms[k][i] >= rho:
                r[k+1][i] = torch.argmax(step_logits[k][i], dim=-1)

    return (r[N], r)
