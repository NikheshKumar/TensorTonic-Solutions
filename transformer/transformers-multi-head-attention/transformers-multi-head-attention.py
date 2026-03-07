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
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v


    q = (q.reshape(batch_size, seq_len, num_heads, d_k)).transpose(0, 2, 1, 3)
    k = (k.reshape(batch_size, seq_len, num_heads, d_k)).transpose(0, 2, 1, 3)
    v = (v.reshape(batch_size, seq_len, num_heads, d_k)).transpose(0, 2, 1, 3)
    
    # Scaled Dot product attentions
  
    scores = q @ k.transpose(0,1,3,2) / np.sqrt(d_k)

    softmax_scores = softmax(scores, axis=1)


    # Concatenation of multi head attention

    c = softmax_scores @ v

    res = c.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model) @ W_o

    return res

  