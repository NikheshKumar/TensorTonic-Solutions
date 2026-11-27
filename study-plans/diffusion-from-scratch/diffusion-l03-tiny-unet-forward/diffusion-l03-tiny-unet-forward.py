import math
import torch
import torch.nn.functional as F

def tiny_unet_forward(x: torch.Tensor, timesteps: torch.Tensor, params: dict, num_groups: int) -> torch.Tensor:
    """
    Returns the network output of shape (B, C_out, H, W) with the dtype of x.
    """
    out_dtype = x.dtype
    
    def GN(z, W, b):
        return F.group_norm(z, num_groups, W, b, eps=1e-5)

    def skip_connection(z, skip_weight, skip_bias):
        if skip_weight is not None and skip_bias is not None:
            return F.conv2d(z, skip_weight, skip_bias)  
        else:
            return z

    def residual_block(h, e, prefix):

        C_out = params[f"{prefix}_conv1_weight"].shape[0]

        p = F.linear(F.silu(e), params[f"{prefix}_time_weight"], params[f"{prefix}_time_bias"])

        p = p.reshape(B, C_out, 1, 1)

        a = GN(h, params[f"{prefix}_norm1_weight"], params[f"{prefix}_norm1_bias"])
        
        a = F.conv2d(F.silu(a), params[f"{prefix}_conv1_weight"], params[f"{prefix}_conv1_bias"], padding=1)

        a = a + p

        g = GN(a, params[f"{prefix}_norm2_weight"], params[f"{prefix}_norm2_bias"])

        g = F.conv2d(F.silu(g), params[f"{prefix}_conv2_weight"], params[f"{prefix}_conv2_bias"], padding=1)
        

        skip_weight = params.get(f"{prefix}_skip_weight")
        skip_bias = params.get(f"{prefix}_skip_bias")

        z = g + skip_connection(h, skip_weight, skip_bias)
        
        return z
        

    B, C_in, H, W = x.shape 
    T, E = params["time_linear1_weight"].shape
        
    t = timesteps.reshape(B,1).to(out_dtype)

    i = torch.arange(E//2, dtype=out_dtype, device=x.device)
    f = 10000**(-2.0 * i / E)
    f = f.reshape(1,E//2).to(out_dtype)

    angles = t*f

    e = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    
    lin1 = F.linear(e, params["time_linear1_weight"], params["time_linear1_bias"])
    lin2 = F.linear(F.silu(lin1), params["time_linear2_weight"], params["time_linear2_bias"] )

    h1 = F.conv2d(x, params["conv_in_weight"], params["conv_in_bias"], padding=1)

    h1 = residual_block(h1, lin2, "down_res")

    h2 = F.conv2d(h1, params["downsample_weight"], params["downsample_bias"], stride=2, padding=1)

    h3 = residual_block(h2, lin2, "mid_res")

    u = F.interpolate(h3, size=(h1.shape[-2], h1.shape[-1]), mode="nearest")

    u = F.conv2d(u, params["upsample_weight"], params["upsample_bias"], padding=1)

    out = residual_block(torch.cat([u,h1],dim=1), lin2, "up_res")

    out = GN(out, params["out_norm_weight"], params["out_norm_bias"])

    out = F.conv2d(F.silu(out), params["conv_out_weight"], params["conv_out_bias"], padding=1)

    return out.to(out_dtype)