import torch
from typing import Tuple

def per_channel_int8_quantize(
    x: torch.Tensor,
    channel_axis: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (quantized int8 tensor, per-channel scale tensor, dequantized float tensor)
    """
        
    dims = [d for d in range(x.dim()) if d!=channel_axis]

    eps = 1e-8

    abs_max = torch.amax(torch.abs(x), dim=dims, keepdim=True)

    scale = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max / 127.0)

    q = torch.tensor(torch.clip(torch.round(x/scale), -127.0, 127.0)).to(torch.int8)

    x_dq = q.to(torch.float32) * scale

    return (q, scale, x_dq)
