import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Your code here
    m = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    ln = gamma * (x-m) / np.sqrt(var + eps) + beta
    return ln

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    B, seq_len, d_model = Q.shape
    d_head = d_model // num_heads
    
    Q = (Q @ W_q).reshape(B, seq_len, num_heads, d_head).transpose(0,2,1,3) 
    K = (K @ W_k).reshape(B, seq_len, num_heads, d_head).transpose(0,2,1,3)
    V = (V @ W_v).reshape(B, seq_len, num_heads, d_head).transpose(0,2,1,3)

    scores = Q @ K.transpose(0,1,3,2) / (d_head**0.5)
    weights = softmax(scores, axis=-1)
    att = weights @ V

    context = att.transpose(0,2,1,3).reshape(B, seq_len, d_model)

    mha = context @ W_o

    return mha

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    ffn = np.maximum(0, x@W1+b1)@W2 + b2

    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    x_new = layer_norm(x + multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads), gamma1, beta1)

    output = layer_norm(x_new + feed_forward(x_new, W1, b1, W2, b2), gamma2, beta2)

    return output