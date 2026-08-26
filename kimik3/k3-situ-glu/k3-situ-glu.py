import torch

def situ_glu(input_tensor, gate_projection, up_projection, gate_cap=4.0, up_cap=25.0):
    """
    Returns: the bounded element-wise gated activation.
    """
    gate = input_tensor @ gate_projection.transpose(0,1)

    up = input_tensor @ up_projection.transpose(0,1)

    act1 = gate_cap * torch.tanh(gate/gate_cap) * torch.sigmoid(gate)
    act2 = up_cap * torch.tanh(up/up_cap)

    act = act1 * act2
    return act