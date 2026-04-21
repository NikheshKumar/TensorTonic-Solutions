import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here

    d = { vocab[i]:i for i in range(len(vocab)) }
    
    res = np.zeros((len(vocab),), dtype=int)


    for t in tokens:
        if t in d:
            res[d[t]] += 1


    return res