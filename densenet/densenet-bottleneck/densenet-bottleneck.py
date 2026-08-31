import torch
import torch.nn.functional as F

def bottleneck_layer(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, conv1_weight,
                     bn2_gamma, bn2_beta, bn2_mean, bn2_var, conv2_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W) after the two-stage bottleneck composite.
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)
    
    bn1_gamma = torch.tensor(bn1_gamma, dtype=torch.float64)
    bn1_beta = torch.tensor(bn1_beta, dtype=torch.float64)
    
    bn2_gamma = torch.tensor(bn2_gamma, dtype=torch.float64)
    bn2_beta = torch.tensor(bn2_beta, dtype=torch.float64)
    
    bn1_mean = torch.tensor(bn1_mean, dtype=torch.float64)
    bn2_mean = torch.tensor(bn2_mean, dtype=torch.float64)
    
    bn1_var = torch.tensor(bn1_var, dtype=torch.float64)
    bn2_var = torch.tensor(bn2_var, dtype=torch.float64)
    
    conv1_weight = torch.tensor(conv1_weight, dtype=torch.float64)
    conv2_weight = torch.tensor(conv2_weight, dtype=torch.float64)
    
    N, C, H, W = x.shape
    
    def BN(x, gamma, beta, m, var, eps):
        return gamma[None,:,None, None] * (x-m[None,:,None, None]) * torch.rsqrt(var[None,:,None, None] + eps) + beta[None,:,None, None]

    eps = 1e-5

    y1 = F.relu(BN(x, bn1_gamma, bn1_beta, bn1_mean, bn1_var, eps))

    y1 = F.conv2d(y1, weight = conv1_weight, padding=0, bias=None, stride=1)

    y2 = F.relu(BN(y1, bn2_gamma, bn2_beta, bn2_mean, bn2_var, eps))

    y2 = F.conv2d(y2, weight=conv2_weight, padding=1, bias=None, stride=1)

    return y2

    
