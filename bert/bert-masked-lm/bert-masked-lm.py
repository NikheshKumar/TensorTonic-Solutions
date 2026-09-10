import numpy as np

def apply_mlm_mask(token_ids: np.ndarray, mask_positions: np.ndarray,
                   replace_probs: np.ndarray, random_tokens: np.ndarray,
                   mask_token_id: int = 103) -> dict:
    """
    Returns masked_ids and labels as int64 arrays in a dictionary.
    """
    token_ids = np.asarray(token_ids, dtype=np.int64)
    mask_positions = np.asarray(mask_positions, dtype=bool)
    replace_probs = np.asarray(replace_probs, dtype=np.float64)
    random_tokens = np.asarray(random_tokens, dtype=np.int64)

    
    labels = np.full_like(token_ids, -100,  dtype=np.int64)
    masked_ids = token_ids.copy()

    labels[mask_positions] = token_ids[mask_positions]

    mask1 = mask_positions & (replace_probs < 0.8)
    mask2 = mask_positions & (replace_probs <0.9) & (replace_probs>=0.8)
    
    masked_ids[mask1] = mask_token_id
    masked_ids[mask2] = random_tokens[mask2]


    return {"masked_ids": masked_ids, "labels": labels}
                
                