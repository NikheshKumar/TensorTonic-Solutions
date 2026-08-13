import torch

def gemma3_block(x, W_q, W_k, W_v, W_o, gamma_attn, gamma_qk_q, gamma_qk_k,
                 cos_freq, sin_freq, gamma_ffn, W_gate, W_up, W_down,
                 layer_idx, local_ratio, window_size, h_q, h_kv, eps=1e-6):
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    # YOUR CODE HERE
    def gemma_attn(x):

        B, seq_len, d_model = x.shape
        d_head = d_model // h_q
    
        def apply_rms_norm(x, gamma, eps):
            x_new = x * gamma * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
            return x_new

            
        def apply_rope(z, cos_freq, sin_freq):
            z_rotated = z.clone()
            z_even = z[...,0::2]
            z_odd = z[...,1::2]

            cos_freq = cos_freq.unsqueeze(0).unsqueeze(0)
            sin_freq = sin_freq.unsqueeze(0).unsqueeze(0)

            z_rotated[...,0::2] = z_even * cos_freq - z_odd * sin_freq
            z_rotated[...,1::2] = z_even * sin_freq + z_odd * cos_freq

            return z_rotated

        x_norm = apply_rms_norm(x, gamma_attn, eps)
        Q = (x_norm @ W_q.T).view(B, seq_len, h_q, d_head).transpose(1, 2)
        K = (x_norm @ W_k.T).view(B, seq_len, h_kv, d_head).transpose(1, 2)
        V = (x_norm @ W_v.T).view(B, seq_len, h_kv, d_head).transpose(1, 2)


        Q_norm = apply_rms_norm(Q, gamma_qk_q.view(1, 1, 1, d_head), eps)
        K_norm = apply_rms_norm(K, gamma_qk_k.view(1, 1, 1, d_head), eps)

        Q_rotated = apply_rope(Q_norm, cos_freq, sin_freq)
        K_rotated = apply_rope(K_norm, cos_freq, sin_freq)

        
        n_rep = h_q // h_kv
        K_rotated = K_rotated.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)

        scores = Q_rotated @ K_rotated.transpose(-2,-1) / (d_head**0.5)

        i = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j = torch.arange(seq_len, device=x.device).unsqueeze(0)

        if (layer_idx + 1) % (local_ratio + 1) != 0:
            
            out_of_bounds_mask = (i-j<0) | (i - j > window_size)

        else:

            out_of_bounds_mask = (i-j<0)
            

        scores = scores.masked_fill(out_of_bounds_mask, value=float('-inf'))

        weights = torch.nn.functional.softmax(scores, dim=-1)

        att = weights @ V

        context = att.transpose(1, 2).contiguous().view(B, seq_len, d_model)

        out = x + (context @ W_o.T)

        return out
        


    h = gemma_attn(x)

    h_norm = h * gamma_ffn * torch.rsqrt(torch.mean(h**2, dim=-1, keepdim=True) + eps)

    gate = (h_norm @ W_gate.T) * torch.sigmoid(h_norm @ W_gate.T)
    
    up = h_norm @ W_up.T
    
    ffn = (gate * up) @ W_down.T
    
    output = h + ffn
    
    return output

    