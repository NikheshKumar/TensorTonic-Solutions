import torch
import torch.nn.functional as F


def BN(x, gamma, beta, mean, var, eps):
    return gamma.view(1,-1,1,1) * (x-mean.view(1,-1,1,1)) * torch.rsqrt(var.view(1,-1,1,1) + eps) + beta.view(1,-1,1,1)
        
def composite_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-3x3Conv (padding 1, no bias) producing growth_rate channels.
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)
    bn_gamma = torch.tensor(bn_gamma, dtype=torch.float64)
    bn_beta = torch.tensor(bn_beta, dtype=torch.float64)
    bn_mean = torch.tensor(bn_mean, dtype=torch.float64)
    bn_var = torch.tensor(bn_var, dtype=torch.float64)
    conv_weight = torch.tensor(conv_weight, dtype=torch.float64)

    H = F.relu(BN(x, bn_gamma, bn_beta, bn_mean, bn_var, eps))
    H = F.conv2d(H, conv_weight, stride=1, padding=1, bias=None)

    return H


def dense_block(x, layers, eps):
    """
    Returns torch.Tensor: concat of x and every composite-layer output (channels grow by growth_rate per layer).
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)

    layer_output = [x]

    n_layers = len(layers)

    for l in range(n_layers):

        bn_gamma = torch.tensor(layers[l]["bn_gamma"], dtype=torch.float64)
        bn_beta = torch.tensor(layers[l]["bn_beta"], dtype=torch.float64)
        bn_mean = torch.tensor(layers[l]["bn_mean"], dtype=torch.float64)
        bn_var = torch.tensor(layers[l]["bn_var"], dtype=torch.float64)
        conv_weight = torch.tensor(layers[l]["conv_weight"], dtype=torch.float64)

        x_in = torch.cat(layer_output, dim=1)

        H = composite_layer(x_in, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps)
        layer_output.append(H)

    x_l = torch.cat(layer_output, dim=1)

    return x_l
        


def transition_layer(x, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps):
    """
    Returns torch.Tensor: BN-ReLU-1x1Conv then 2x2 average pool with stride 2 (channels compressed, H and W halved).
    """
    # YOUR CODE HERE
    x = torch.tensor(x, dtype=torch.float64)
    bn_gamma = torch.tensor(bn_gamma, dtype=torch.float64)
    bn_beta = torch.tensor(bn_beta, dtype=torch.float64)
    bn_mean = torch.tensor(bn_mean, dtype=torch.float64)
    bn_var = torch.tensor(bn_var,dtype=torch.float64)
    conv_weight = torch.tensor(conv_weight, dtype=torch.float64)
    

    y = F.relu(BN(x, bn_gamma, bn_beta, bn_mean, bn_var, eps))
    H = F.conv2d(y, conv_weight, padding=0, stride=1, bias=None)
    H = F.avg_pool2d(H, kernel_size=(2,2), stride=2)

    return H


def densenet_forward(x, weights, growth_rate, eps=1e-5):
    """
    Returns torch.Tensor of shape (N, num_classes) with class logits.
    """
    # YOUR CODE HERE

    x = torch.tensor(x, dtype=torch.float64)

    stem_conv = torch.tensor(weights["stem_conv"], dtype=torch.float64)

    y = F.conv2d(x, stem_conv, padding=1, stride=1, bias=None)

    n_blocks = len(weights["blocks"])
    blocks = weights["blocks"]
    fc_weight = weights["fc_weight"]
    fc_bias = weights["fc_bias"]
    

    for i in range(n_blocks):
        y = dense_block(y, blocks[i], eps)
        if i < n_blocks - 1:
            transitions = weights["transitions"]
            bn_gamma = torch.tensor(transitions[i]["bn_gamma"], dtype=torch.float64)
            bn_beta = torch.tensor(transitions[i]["bn_beta"], dtype=torch.float64)
            bn_mean = torch.tensor(transitions[i]["bn_mean"], dtype=torch.float64)
            bn_var = torch.tensor(transitions[i]["bn_var"], dtype=torch.float64)
            conv_weight = torch.tensor(transitions[i]["conv_weight"], dtype=torch.float64)
            y = transition_layer(y, bn_gamma, bn_beta, bn_mean, bn_var, conv_weight, eps)


    final_bn_gamma = torch.tensor(weights["final_bn_gamma"], dtype=torch.float64)
    final_bn_beta = torch.tensor(weights["final_bn_beta"], dtype=torch.float64)
    final_bn_mean = torch.tensor(weights["final_bn_mean"], dtype=torch.float64)
    final_bn_var = torch.tensor(weights["final_bn_var"], dtype=torch.float64)
    
    final_y = F.relu(BN(y, final_bn_gamma, final_bn_beta, final_bn_mean, final_bn_var, eps))

    output = torch.mean(final_y, dim=(2,3))

    output = output @ fc_weight.T + fc_bias

    return output

    

    
    
    
