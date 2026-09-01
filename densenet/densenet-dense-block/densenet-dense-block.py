import torch

def dense_block(x, layers, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, C + L*growth_rate, H, W).
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)
    
    def BN(x, gamma, beta, m, var, eps):
        return gamma.view(1,-1,1,1) * (x-m.view(1,-1,1,1)) * torch.rsqrt(var.view(1,-1,1,1) + eps) + beta.view(1,-1,1,1)


    layer_output = [x]

    n_layers = len(layers)
    
    for l in range(n_layers):
        
        gamma = torch.tensor(layers[l]["bn_gamma"], dtype=torch.float64)
        beta = torch.tensor(layers[l]["bn_beta"], dtype=torch.float64)
        m = torch.tensor(layers[l]["bn_mean"], dtype=torch.float64)
        var = torch.tensor(layers[l]["bn_var"], dtype=torch.float64)
        conv_weight = torch.tensor(layers[l]["conv_weight"], dtype=torch.float64)

        x_in = torch.cat(layer_output, dim=1)
    
        H = BN(x_in, gamma, beta, m, var, eps)
        H = torch.nn.functional.relu(H)
        H = torch.nn.functional.conv2d(H, weight = conv_weight, stride=1, padding=1, bias=None)
        layer_output.append(H)
        

    x_l = torch.cat(layer_output, dim=1)

    return x_l
