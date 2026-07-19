import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Returns: None
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))
        

    def forward(self, Q, K, V):
        """
        Returns: output tensor
        """
        import math 

        B, S, d_model = Q.shape

        q = (Q @ self.W_q).reshape(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = (K @ self.W_k).reshape(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = (V @ self.W_v).reshape(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)/math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)

        att = weights @ v

        mha = att.transpose(1, 2).contiguous().reshape(B, S, self.d_model) @ self.W_o

        return mha

        

        

        

        
        