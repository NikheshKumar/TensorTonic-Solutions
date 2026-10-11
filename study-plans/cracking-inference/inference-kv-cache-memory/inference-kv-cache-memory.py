import torch

def kv_cache_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_query_heads: int,
    gqa_kv_heads: int,
    head_dim: int,
    mla_latent_dim: int,
    mla_rotary_key_dim: int,
    bytes_per_element: int,
) -> torch.Tensor:
    """
    Returns an int64 tensor of byte counts ordered [MHA, MQA, GQA, MLA].
    """
    def f(z):
        return 2 * batch_size * seq_len * num_layers * z * head_dim * bytes_per_element

    M_mha = f(num_query_heads)
    M_mqa = f(1)
    M_gqa = f(gqa_kv_heads)

    M_mla = batch_size * seq_len * num_layers * (mla_latent_dim + mla_rotary_key_dim) * bytes_per_element

    res = [M_mha, M_mqa, M_gqa, M_mla]

    return torch.tensor(res, dtype=torch.long)
    
