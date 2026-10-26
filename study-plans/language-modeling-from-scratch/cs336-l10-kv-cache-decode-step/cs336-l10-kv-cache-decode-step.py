import torch

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: output, new_key_cache, new_value_cache.
    """
    B, Hq, D = query.shape
    _,Hkv, S, _ = key_cache.shape

    G = Hq // Hkv

    if new_key.ndim == 3 and new_value.ndim==3:
        new_key = new_key.unsqueeze(2)
        new_value = new_value.unsqueeze(2)

    new_key_cache = torch.cat([key_cache,new_key], dim=-2)
    new_value_cache = torch.cat([value_cache, new_value], dim=-2)


    q = query.reshape(B, Hkv, G, D)
    k = new_key_cache.unsqueeze(2).expand(B, Hkv, G, (S+1), D).reshape(B, Hq, (S+1), D)
    v = new_value_cache.unsqueeze(2).expand(B, Hkv, G, (S+1), D).reshape(B, Hq, (S+1), D)

    scores = torch.einsum("bhd, bhsd->bhs", query, k)/(D**0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhs, bhsd->bhd", weights, v)


    return {"output": output,
            "new_key_cache": new_key_cache,
            "new_value_cache": new_value_cache}
