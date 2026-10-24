import torch
import torch.nn.functional as F

def compact_policy_value_network(
    x: torch.Tensor,
    trunk_weight: torch.Tensor,
    trunk_scale: torch.Tensor,
    trunk_shift: torch.Tensor,
    residual_parameters: list[dict],
    head_parameters: dict,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: float32 tensors (policy_logits, values), shaped (B, N*N+1) and (B,).
    """
    def bn(z, gamma, delta, eps):
        mu  = z.mean(dim=(0, 2, 3), keepdim=True)
        var = z.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        gamma = gamma.reshape(1, -1, 1, 1)
        delta = delta.reshape(1, -1, 1, 1)
        return gamma * (z - mu) * torch.rsqrt(var + eps) + delta

    policy_conv_weight = head_parameters["policy_conv_weight"]
    policy_scale = head_parameters["policy_scale"]
    policy_shift = head_parameters["policy_shift"]
    policy_linear_weight = head_parameters["policy_linear_weight"]
    policy_linear_bias = head_parameters["policy_linear_bias"]

    value_conv_weight =head_parameters["value_conv_weight"]
    value_scale = head_parameters["value_scale"]
    value_shift = head_parameters["value_shift"]
    value_hidden_weight = head_parameters["value_hidden_weight"]
    value_hidden_bias = head_parameters["value_hidden_bias"]
    value_output_weight = head_parameters["value_output_weight"]
    value_output_bias = head_parameters["value_output_bias"]


    z = F.conv2d(x, trunk_weight, padding=1)
    h = torch.relu(bn(z, trunk_scale, trunk_shift, epsilon))


    for i in range(len(residual_parameters)):
        conv1_weight, conv2_weight = residual_parameters[i]["conv1_weight"], residual_parameters[i]["conv2_weight"]
        scale1, shift1 = residual_parameters[i]["scale1"], residual_parameters[i]["shift1"]
        scale2, shift2 = residual_parameters[i]["scale2"], residual_parameters[i]["shift2"]
        
        z = F.conv2d(h, conv1_weight, padding=1)
        z = bn(z,scale1, shift1, epsilon)
        z = torch.relu(z)
        z = F.conv2d(z, conv2_weight, padding=1)
        z = bn(z, scale2, shift2, epsilon)
        h = torch.relu(h + z)

    L = F.conv2d(h, policy_conv_weight)
    L = bn(L, policy_scale, policy_shift, epsilon)
    L = torch.relu(L)
    L = L.flatten(1)
    policy_logits = F.linear(L, policy_linear_weight, policy_linear_bias)
    

    v = F.conv2d(h, value_conv_weight)
    v = bn(v, value_scale, value_shift, epsilon)
    v = torch.relu(v)
    v = v.flatten(1)
    v = F.linear(v, value_hidden_weight, value_hidden_bias)
    v = torch.relu(v)
    v = torch.tanh(F.linear(v, value_output_weight, value_output_bias))
    values = v.squeeze(-1)

    return policy_logits.to(torch.float32), values.to(torch.float32)
        
        
        


    

    

    
    
    
        
