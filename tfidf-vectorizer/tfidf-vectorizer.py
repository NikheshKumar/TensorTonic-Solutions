import math
from collections import Counter, defaultdict
import numpy as np

def tfidf_vectorizer(documents: list[str]) -> dict:
    """
    Returns a dictionary with tfidf_matrix and vocabulary.
    """
    # Write code here
    tokenized = [doc.lower().split() for doc in documents]
    vocab = sorted({tok for tokens in tokenized for tok in tokens})
    term_to_idx = {t: i for i, t in enumerate(vocab)}
    N, V = len(documents), len(vocab)

    n_t = defaultdict(int)

    for t in tokenized:
        for term in set(t):
            n_t[term] += 1

    idf = {t: math.log(N / n_t[t])  for t in vocab}

    tfidf_matrix = np.zeros((N, V), dtype=np.float64)
    
    for i, tokens in enumerate(tokenized):
        counts = Counter(tokens)
        total = len(tokens)
        for term, count in counts.items():
            tf = count / total 
            tfidf_matrix[i, term_to_idx[term]] = tf * idf[term]

    return {"tfidf_matrix": tfidf_matrix, "vocabulary": vocab}
