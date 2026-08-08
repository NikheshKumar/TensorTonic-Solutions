import torch
import torch.nn.functional as F
import math

def llama_block(x, rms_w1, rms_w2, W_q, W_k, W_v, W_o, n_heads, n_kv_heads, W_gate, W_up, W_down, freqs_cos, freqs_sin, eps=1e-6):
    """
    Returns: dict with key "output" containing tensor (batch, seq_len, d_model) as nested list, rounded to 4 decimals.
    """
    # YOUR CODE HERE
    x = torch.as_tensor(x, dtype=torch.float32)
    
    rms_w1 = torch.as_tensor(rms_w1, dtype=torch.float32)
    rms_w2 = torch.as_tensor(rms_w2, dtype=torch.float32)
    
    w1 = torch.tensor(rms_w1, dtype=torch.float32)
    w2 = torch.tensor(rms_w2, dtype=torch.float32)
    
    W_q = torch.tensor(W_q, dtype=torch.float32)
    W_k = torch.tensor(W_k, dtype=torch.float32)
    W_v = torch.tensor(W_v, dtype=torch.float32)
    W_o = torch.tensor(W_o, dtype=torch.float32)
    W_gate = torch.tensor(W_gate, dtype=torch.float32)
    W_up = torch.tensor(W_up, dtype=torch.float32)
    W_down = torch.tensor(W_down, dtype=torch.float32)
    
    freqs_cos = torch.tensor(freqs_cos, dtype=torch.float32)
    freqs_sin = torch.tensor(freqs_sin, dtype=torch.float32)
    
    B, seq_len, d_model = x.shape
    d_head = d_model // n_heads
    
    x_new = x * rms_w1 / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

    Q = (x_new @ W_q.T).view(B, seq_len, n_heads, d_head).transpose(1, 2)
    K = (x_new @ W_k.T).view(B, seq_len, n_kv_heads, d_head).transpose(1, 2)
    V = (x_new @ W_v.T).view(B, seq_len, n_kv_heads, d_head).transpose(1, 2)

    def rope(z, freqs_cos, freqs_sin):
        z_rotated = z.clone()
        z_even, z_odd = z[...,0::2], z[...,1::2]
        z_rotated[...,0::2] = z_even * freqs_cos.unsqueeze(0).unsqueeze(0) - z_odd * freqs_sin.unsqueeze(0).unsqueeze(0)
        z_rotated[...,1::2] = z_even * freqs_sin.unsqueeze(0).unsqueeze(0) + z_odd * freqs_cos.unsqueeze(0).unsqueeze(0)
        return z_rotated

    Q_rot = rope(Q, freqs_cos, freqs_sin)
    K_rot = rope(K, freqs_cos, freqs_sin)

    num_queries = n_heads // n_kv_heads
    K_rot = K_rot.repeat_interleave(num_queries, dim=1)
    V = V.repeat_interleave(num_queries, dim=1)

    scores = (Q_rot @ K_rot.transpose(-2, -1)) / math.sqrt(d_head)
    att = F.softmax(scores, dim=-1)
    context = (att @ V).transpose(1, 2).contiguous().view(B, seq_len, d_model)
    gqa = context @ W_o.T

    h = x + gqa 

    h_new = h * rms_w2 / torch.sqrt(torch.mean(h**2, dim=-1, keepdim=True) + eps)

    ffn = (F.silu(h_new @ W_gate.T) *(h_new @ W_up.T)) @ W_down.T

    output = h + ffn

    return {"output": torch.round(output, decimals=4).tolist()}