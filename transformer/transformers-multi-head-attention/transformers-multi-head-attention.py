import numpy as np
import math

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
    B, seq_len, d_model = Q.shape
    d_head = d_model // num_heads

    Q = (Q @ W_q).reshape(B, seq_len, num_heads, d_head).transpose(0,2,1,3)
    K = (K @ W_k).reshape(B, seq_len, num_heads, d_head).transpose(0,2,1,3)
    V = (V @ W_v).reshape(B, seq_len, num_heads, d_head).transpose(0,2,1,3)

    scores = Q @ K.transpose(0,1,3,2) / math.sqrt(d_head)

    weights = softmax(scores)

    context = (weights @ V).transpose(0,2,1,3).reshape(B, seq_len, d_model)

    mha = context @ W_o

    return mha

    