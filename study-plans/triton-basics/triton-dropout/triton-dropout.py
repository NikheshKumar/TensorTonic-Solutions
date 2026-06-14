import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    # Write code here
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bound_mask = offset < n

    
    x = tl.load(x_ptr + offset, mask=bound_mask)
    m = tl.load(mask_ptr + offset, mask=bound_mask)
    y = x * m * 1.0/(1.0-p)
    tl.store(out_ptr + offset, y, mask=bound_mask)

def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )