import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here

    d = { vocab[i]:i for i in range(len(vocab)) }

    ans = np.zeros(len(vocab), dtype=int)

    for tok in tokens:
        if tok in d:
            ans[d[tok]] += 1

    return np.asarray(ans, int)    

    