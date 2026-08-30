import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Your code here
    e = torch.nn.Embedding(vocab_size, d_model)

    return e

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Your code here
    scaled_emb = embedding(tokens) * math.sqrt(d_model)

    return scaled_emb
    