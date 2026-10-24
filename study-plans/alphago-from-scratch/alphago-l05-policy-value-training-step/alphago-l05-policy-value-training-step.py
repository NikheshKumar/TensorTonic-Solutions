import torch
import torch.nn.functional as F

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns detached float32 tensors: loss, policy weight/bias, value weight/bias.
    """
    X = features.to(torch.float32)
    policy_weight = policy_weight.to(torch.float32)
    policy_bias = policy_bias.to(torch.float32)
    value_weight = value_weight.to(torch.float32)
    value_bias = value_bias.to(torch.float32)
    target_policies = target_policies.to(torch.float32)
    target_values = target_values.to(torch.float32)

    pw = policy_weight.to(torch.float32).clone().requires_grad_(True)            
    pb = policy_bias.to(torch.float32).clone().requires_grad_(True)   
    vw = value_weight.to(torch.float32).reshape(1, -1).clone().requires_grad_(True)
    vb = value_bias.to(torch.float32).reshape(1).clone().requires_grad_(True)
    
    B, D = X.shape
    
    p = F.log_softmax(F.linear(X, pw, pb), dim=-1)
    v = torch.tanh(F.linear(X, vw, vb)).squeeze(-1)

    l = -torch.sum(target_policies * p, dim=-1).mean()
    l += torch.mean((v-target_values)**2)
    l += l2_coefficient * (vw.pow(2).sum() + pw.pow(2).sum() + vb.pow(2).sum() + pb.pow(2).sum())

    l.backward()
    with torch.no_grad():
        policy_weight_new = (pw - learning_rate * pw.grad).detach().to(torch.float32)
        policy_bias_new = (pb - learning_rate * pb.grad).detach().to(torch.float32)
        value_weight_new = (vw - learning_rate * vw.grad).detach().reshape(value_weight.shape).to(torch.float32)
        value_bias_new = (vb - learning_rate * vb.grad).detach().reshape(value_bias.shape).to(torch.float32)


    return l.detach(), policy_weight_new, policy_bias_new, value_weight_new, value_bias_new

    

    
