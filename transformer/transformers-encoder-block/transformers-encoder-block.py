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

    x = np.asarray(x, float)
    gamma = np.asarray(gamma, float)
    beta = np.asarray(beta, float)

    m = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
  
    ln = gamma*(x-m)/np.sqrt(var + eps) + beta
    return ln

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q = np.asarray(Q, float)
    K = np.asarray(K, float)
    V = np.asarray(V, float)

    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  
  
    # projections

    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    q_hat = q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    k_hat = k.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    v_hat = v.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)

    # weights
  
    scores = q_hat @ k_hat.transpose(0,1,3,2) / np.sqrt(d_k)
    softmax_scores = softmax(scores) 

    # concatenation
  
    c = (softmax_scores @ v_hat).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
  
    res = c @ W_o

    return res

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    x = np.asarray(x, float)
    W1 = np.asarray(W1, float)
    b1 = np.asarray(b1, float)
    W2 = np.asarray(W2, float)
    b2 = np.asarray(b2, float)

    relu = np.maximum(0,np.matmul(x,W1)+b1)
    ffn = relu @ W2 + b2

    return ffn

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    Q=K=V=x
  
    mha = multi_head_attention(Q,K,V,W_q,W_k,W_v,W_o,num_heads)
    x_new = layer_norm(x + mha, gamma1, beta1)
    ffn = feed_forward(x_new, W1, b1, W2, b2)
    enc = layer_norm(x_new + ffn, gamma2, beta2)

    return enc
    

    
    