import torch
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(x_ptr, sumsq_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offset < n

    x = tl.load(x_ptr + offset, mask=m)
    y = tl.sum(x*x, axis=0)
    tl.atomic_add(sumsq_ptr, y)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch l2_norm_kernel and finalize the square root."""
    n = x.numel()
    sumsq_buf = torch.zeros(1, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    l2_norm_kernel[grid](x, sumsq_buf, n, BLOCK_SIZE=BLOCK_SIZE)
    out.copy_(torch.sqrt(sumsq_buf))