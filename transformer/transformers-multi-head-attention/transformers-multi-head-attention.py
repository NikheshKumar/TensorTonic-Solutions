import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    import numpy as np 
    import math 
    
    batch_size, seq_length, d_model = Q.shape
    d_k = d_model // num_heads

    Q_proj = (Q @ W_q).reshape(batch_size, seq_length, num_heads, d_k).transpose(0,2,1,3)
    K_proj = (K @ W_k).reshape(batch_size, seq_length, num_heads, d_k).transpose(0,2,1,3)
    V_proj = (V @ W_v).reshape(batch_size, seq_length, num_heads, d_k).transpose(0,2,1,3)

    scores = Q_proj @ K_proj.transpose(0, 1, 3, 2)
    scaled_scores = scores / math.sqrt(d_k)

    weights = softmax(scaled_scores, axis=-1)

    context = (weights @ V_proj).transpose(0, 2, 1, 3).reshape(batch_size, seq_length, d_model)

    mha = context @ W_o

    return mha

    

 