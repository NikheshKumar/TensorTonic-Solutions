import torch

def glm_swiglu(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a float64 tensor with shape (n_tokens, hidden_size).
    """
    gate = torch.nn.functional.silu(x @ W_gate)

    up = x @ W_up

    y = (gate * up) @ W_down

    return y