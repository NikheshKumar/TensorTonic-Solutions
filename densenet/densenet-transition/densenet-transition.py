import torch
import torch.nn.functional as F

def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, out_channels, H//2, W//2) after BN-ReLU-1x1Conv then 2x2 average pooling.
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)
    bn_beta = torch.tensor(bn_beta, dtype=torch.float64)
    bn_gamma = torch.tensor(bn_gamma, dtype=torch.float64)
    bn_mean = torch.tensor(bn_mean, dtype=torch.float64)
    bn_var = torch.tensor(bn_var, dtype=torch.float64)
    conv_weight = torch.tensor(conv_weight, dtype=torch.float64)

    def BN(x, gamma, beta, m, var, eps):
        return gamma.view(1,-1,1,1) * (x - m.view(1,-1,1,1)) * torch.rsqrt(var.view(1,-1,1,1) + eps) + beta.view(1,-1,1,1)

    eps = 1e-5
    y = F.relu(BN(x, bn_gamma, bn_beta, bn_mean, bn_var, eps))

    H = F.conv2d(y, weight = conv_weight, padding=0, stride=1, bias=None)

    H = F.avg_pool2d(H, kernel_size=(2,2), stride=2)


    return H
