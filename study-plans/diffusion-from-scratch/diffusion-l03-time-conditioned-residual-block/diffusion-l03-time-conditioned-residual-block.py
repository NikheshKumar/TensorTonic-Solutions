import torch
import torch.nn.functional as F

def time_residual_block(x: torch.Tensor, time_embedding: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the block output of shape (B, C_out, H, W) with the dtype of x.
    """
    out_dtype = x.dtype           

    W_t = params["time_weight"]
    b_t = params["time_bias"]
    
    norm1_weight = params["norm1_weight"]
    norm1_bias = params["norm1_bias"]
    norm2_weight = params["norm2_weight"]
    norm2_bias = params["norm2_bias"]
    
    conv1_weight = params["conv1_weight"]
    conv1_bias = params["conv1_bias"]
    conv2_weight = params["conv2_weight"]
    conv2_bias = params["conv2_bias"]
    
    skip_weight = params.get("skip_weight")
    skip_bias   = params.get("skip_bias")

    def GN(z, W, b):
        return F.group_norm(z, num_groups, W, b, eps=1e-5)

    def skip_connection(z, skip_weight, skip_bias):
        if skip_weight is not None and skip_bias is not None:
            return F.conv2d(z, skip_weight, skip_bias)  
        else:
            return z

    
    tau = (F.silu(time_embedding) @ W_t.T + b_t)[:, :, None, None]

    h = F.conv2d(F.silu(GN(x, norm1_weight, norm1_bias)), conv1_weight, conv1_bias, padding=1) + tau

    out = skip_connection(x, skip_weight, skip_bias) + F.conv2d(F.silu(GN(h, norm2_weight, norm2_bias)), conv2_weight, conv2_bias, padding=1)


    return out.to(dtype=out_dtype)
