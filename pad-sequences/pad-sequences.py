import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs or len(seqs) == 0:
        return np.array([[]])

    N = len(seqs)

    if max_len==None:
      max_len = max(len(s) for s in seqs) if seqs else 0

    output = np.full((N,max_len), pad_value)

    for i, seq in enumerate(seqs):
      if len(seq) == 0:
        continue
      else:
        l = min(len(seq), max_len)
        output[i, :l] = seq[:l]    

    return output

        