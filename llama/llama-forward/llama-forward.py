import torch
import torch.nn.functional as F
import math

def llama_forward(token_ids, W_embed, blocks, rms_final, W_head, freqs_cos, freqs_sin, eps=1e-6):
    """
    Returns: logits tensor (batch, seq_len, vocab_size) from Llama 3 forward pass.
    """


    def apply_rope(z, freqs_cos, freqs_sin):

        z_rotated = z.clone()
        z_even = z[...,0::2]
        z_odd = z[...,1::2]

        freqs_cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
        freqs_sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

        z_rotated[...,0::2] = z_even * freqs_cos - z_odd * freqs_sin
        z_rotated[...,1::2] = z_even * freqs_sin + z_odd * freqs_cos

        return z_rotated

    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    W_embed = torch.as_tensor(W_embed, dtype=torch.float32)
    rms_final = torch.as_tensor(rms_final, dtype=torch.float32)
    W_head = torch.as_tensor(W_head, dtype=torch.float32)

    h = W_embed[token_ids]
    B, seq_len, d_model = h.shape
        

    for block in blocks :

        n_head = block['n_heads']
        n_kv = block['n_kv_heads']
        d_head = d_model // n_head
        n_rep = n_head // n_kv
        

        h1 =  h*block['rms_w1']* torch.rsqrt(torch.mean(h**2, dim=-1, keepdim=True) + eps)

        Q = (h1 @ block['W_q'].T).view(B, seq_len, n_head, d_head).permute(0, 2, 1, 3)
        K = (h1 @ block['W_k'].T).view(B, seq_len, n_kv, d_head).permute(0, 2, 1, 3)
        V = (h1 @ block['W_v'].T).view(B, seq_len, n_kv, d_head).permute(0, 2, 1, 3)
        
        Q_rotated = apply_rope(Q, freqs_cos, freqs_sin)
        K_rotated = apply_rope(K, freqs_cos, freqs_sin)

        K_rotated = K_rotated.repeat_interleave(n_rep, dim=1).reshape(B, n_kv*n_rep, seq_len, d_head)
        V = V.repeat_interleave(n_rep, dim=1).reshape(B, n_kv*n_rep, seq_len, d_head)
        
        scores = (Q_rotated @ K_rotated.transpose(-2, -1)) / math.sqrt(d_head)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=h.device),    diagonal=1)
        scores = scores.masked_fill(mask, value=-float('inf'))
        
        att = F.softmax(scores, dim=-1)
        context = (att @ V).permute(0, 2, 1, 3).reshape(B, seq_len, d_model)
        gqa = context @ block['W_o'].T
    
        h = h + gqa
        
        h2 = h*block['rms_w2']* torch.rsqrt(torch.mean(h**2, dim=-1, keepdim=True) + eps)
    
        h = h + (F.silu(h2 @ block['W_gate'].T) * (h2 @ block['W_up'].T)) @ block['W_down'].T


    h_final = h * rms_final * torch.rsqrt(torch.mean(h**2, dim=-1, keepdim=True) + eps)

    logits = h_final @ W_head.T

    return logits
        