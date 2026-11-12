import numpy as np

def multi_head_attention(Q: list, K: list, V: list, W_q: list, W_k: list, W_v: list, W_o: list, num_heads: int, mask: list | None = None) -> dict:
    """
    Returns a dictionary of lists: output, attention_weights.
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W_q = np.asarray(W_q, dtype=np.float64)
    W_k = np.asarray(W_k, dtype=np.float64)
    W_v = np.asarray(W_v, dtype=np.float64)
    W_o = np.asarray(W_o, dtype=np.float64)

    T, D = Q.shape
    H = num_heads
    d_k = D // H

    q = (Q @ W_q.T).reshape(T, H, d_k).transpose(1, 0, 2)
    k = (K @ W_k.T).reshape(T, H, d_k).transpose(1, 0, 2)
    v = (V @ W_v.T).reshape(T, H, d_k).transpose(1, 0, 2)

    def _softmax(z):
        m = np.max(z, axis=-1, keepdims=True)
        e = np.exp(z-m)
        d = np.sum(e, axis=-1, keepdims=True) 
        safe_denom = np.where(d> 0, d, 1.0)
        res = e / safe_denom
        return np.where(d>0, res, 0.0)
    

    scores = q @ k.transpose(0,2,1) / (d_k**0.5)

    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        scores = np.where(~mask_arr[None, :, :], -np.inf, scores)
        
    weights = _softmax(scores)
    att = weights @ v

    output = att.transpose(1, 0, 2).reshape(T, H * d_k) 
    output = output @ W_o.T

    def convert_to_lists(z):
        return np.round(z, 4).tolist()

    return {"output":convert_to_lists(output), "attention_weights":convert_to_lists(weights)}



