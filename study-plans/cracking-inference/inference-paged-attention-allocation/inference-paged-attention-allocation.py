import torch

def allocate_kv_blocks(
    seq_lengths: list,
    block_size: int,
    free_block_ids: list,
) -> tuple:
    """
    Returns (block_table, blocks_used, remaining_free_blocks), integer tensors.
    """
    import math 
    
    blocks_used = [math.ceil(l/block_size) for l in seq_lengths]

    total = sum(blocks_used)
    if total >  len(free_block_ids):
        raise RuntimeError

    width = max(blocks_used, default=0)

    height = len(seq_lengths) 

    block_table = torch.full((height, width), -1, dtype=torch.long)

    curr = 0

    for i, n in enumerate(blocks_used):
        for j in range(n):
            block_table[i,j] = free_block_ids[curr]
            curr += 1

    blocks_used = torch.tensor(blocks_used, dtype=torch.long)
    rem = torch.tensor(free_block_ids[curr:], dtype=torch.long)

    return block_table, blocks_used, rem

    