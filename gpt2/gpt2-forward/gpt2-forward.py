import torch
import torch.nn.functional as F
import math

def gpt2_forward(token_ids, wte, wpe, layers, gamma_f, beta_f, W_lm):
    """
    Returns: list of lists (seq_len, vocab_size) logits from GPT-2 forward pass.
    """

    token_ids = torch.tensor(token_ids, dtype=torch.long)
    wte = torch.tensor(wte, dtype=torch.float64)
    wpe = torch.tensor(wpe, dtype=torch.float64)
    gamma_f = torch.tensor(gamma_f, dtype=torch.float64)
    beta_f = torch.tensor(beta_f, dtype=torch.float64)
    W_lm= torch.tensor(W_lm, dtype=torch.float64)

    
    seq_len = len(token_ids)
    eps = 1e-5
    d_model = wte.shape[-1]
    n_heads = 2
    d_head = d_model // n_heads
    
    pos = torch.arange(seq_len, dtype=torch.long)
    x = wte.index_select(0, token_ids) + wpe.index_select(0, pos)

    def layernorm(z, gamma, beta, eps):
        m = torch.mean(z, dim=-1, keepdim=True)
        var = torch.var(z, dim=-1, keepdim=True, unbiased=False)
        return gamma * (z-m) * torch.rsqrt(var + eps) + beta

    def gelu(z):
        e = torch.tanh(math.sqrt(2.0/math.pi) * (z +0.044715 * z.pow(3)))
        return z * 0.5 * (1.0 + e)
    

    for l in layers:

        eps = 1e-5

        W_q = torch.tensor(l["W_q"], dtype=torch.float64)
        W_k = torch.tensor(l["W_k"], dtype=torch.float64)
        W_v = torch.tensor(l["W_v"], dtype=torch.float64)
        W_o = torch.tensor(l["W_o"], dtype=torch.float64)
        
        gamma1 = torch.tensor(l["gamma1"], dtype=torch.float64)
        beta1 = torch.tensor(l["beta1"], dtype=torch.float64)
        gamma2 = torch.tensor(l["gamma2"], dtype=torch.float64)
        beta2 = torch.tensor(l["beta2"], dtype=torch.float64)

        
        W1 = torch.tensor(l["W1"], dtype=torch.float64)
        W2 = torch.tensor(l["W2"], dtype=torch.float64)
        b1 = torch.tensor(l["b1"], dtype=torch.float64)
        b2 = torch.tensor(l["b2"], dtype=torch.float64)
        

        x_new = layernorm(x, gamma1, beta1, eps)

        Q = (x_new @ W_q.T).reshape(seq_len, n_heads, d_head).permute(1,0,2)
        K = (x_new @ W_k.T).reshape(seq_len, n_heads, d_head).permute(1,0,2)
        V = (x_new @ W_v.T).reshape(seq_len, n_heads, d_head).permute(1,0,2)

        scores = Q @ K.transpose(-2,-1) / math.sqrt(d_head)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[:seq_len, :seq_len], value=-float("inf"))

        weights = torch.softmax(scores, dim=-1)
        att = weights @ V

        h = x + att.permute(1, 0, 2).contiguous().reshape(seq_len, d_model) @ W_o.T
        h_ln = layernorm(h, gamma2, beta2, eps)
        h_hidden = gelu(h_ln @ W1.T + b1)
        h_out = h_hidden @ W2.T + b2
        x = h + h_out


    h_final = layernorm(x, gamma_f, beta_f, eps)

    logits = h_final @ W_lm.T

    return logits.tolist()