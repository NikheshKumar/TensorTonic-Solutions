import torch

def per_layer_embedding(input_embeddings: torch.Tensor, layer_output: torch.Tensor, p_weight: torch.Tensor, g_weight: torch.Tensor, u_weight: torch.Tensor, p_norm_scale: torch.Tensor, u_norm_scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns the layer output with the gated per-layer embedding update added.
    """

    T_in = torch.tensor(input_embeddings, dtype=input_embeddings.dtype) 
    H = torch.tensor(layer_output, dtype=layer_output.dtype)

    def rmsnorm(z, s):
        return torch.rsqrt(torch.mean(z**2, dim=-1, keepdim=True) + eps) * z * s

    def gelu(z):
        return torch.nn.functional.gelu(z)
        

    P = rmsnorm(T_in @ p_weight.T, p_norm_scale)

    G = gelu(H @ g_weight.T)

    U = rmsnorm((G*P) @ u_weight.T, u_norm_scale)

    H_new = H + U

    return H_new


    


    

    
