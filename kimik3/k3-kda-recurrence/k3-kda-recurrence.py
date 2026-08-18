import torch

def kda_recurrence(query, key, value, decay_logits, write_strength, output_gate_logits, output_projection, initial_state, g_min=-5.0, eps=1e-6):
    """
    Returns: sequence outputs and the final recurrent state.
    """
    B, T, H, d_k = query.shape
    _, _, _, d_v = value.shape


    S = initial_state.clone()
    output_seqs = []

    alpha = torch.exp(g_min * torch.sigmoid(decay_logits))

    for t in range(T):

        q_t = query[:, t]
        k_t = key[:, t]
        v_t = value[:, t]
        alpha_t = alpha[:, t]
        beta_t = write_strength[:, t]

        S_decay = alpha_t[..., None] * S

        S_mod = torch.einsum("bhd,bhdv->bhv", k_t, S_decay)

        S = S_decay + beta_t[..., None] * k_t[..., None] * ((v_t - S_mod)[..., None, :])

        o_t = torch.einsum("bhd,bhdv->bhv",q_t, S)

        o_t_rms_val = torch.rsqrt(torch.mean(o_t**2, dim=-1, keepdim=True) + eps) 

        o_t = o_t * o_t_rms_val

        gate_t = torch.sigmoid(output_gate_logits[:, t])

        o_t = (o_t * gate_t).reshape(B, -1)

        o_t = o_t @ output_projection.transpose(0, 1)

        output_seqs.append(o_t)

    output_seqs = torch.stack(output_seqs, dim=1)
    
    return (output_seqs, S)

        
        