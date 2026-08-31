import torch
import torch.nn.functional as F

def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, growth_rate, H, W): BN, ReLU, then a 3x3 same-padding convolution.
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)
    bn_gamma = torch.tensor(bn_gamma, dtype=torch.float64)
    bn_beta = torch.tensor(bn_beta, dtype=torch.float64)
    bn_mean = torch.tensor(bn_mean, dtype=torch.float64)
    bn_var = torch.tensor(bn_var, dtype=torch.float64)
    conv_weight = torch.tensor(conv_weight, dtype=torch.float64)
    
    N, C, H, W = x.shape

    def BN(x, bn_gamma, bn_beta, bn_mean, bn_var, eps):
        return bn_gamma[None,:,None,None] * (x - bn_mean[None,:,None,None]) * torch.rsqrt(bn_var[None,:,None,None] + eps) + bn_beta[None,:,None,None]


    H = torch.nn.functional.relu(BN(x, bn_gamma, bn_beta, bn_mean, bn_var, eps=1e-5))

    H = torch.nn.functional.conv2d(H, weight = conv_weight, padding=1, stride=1, bias=None)
    
    return H
