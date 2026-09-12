import math
from collections import Counter
import numpy as np

def bm25_score(query_tokens: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """
    Returns a NumPy array with one score per document.
    """
    # Write code here
    N = len(docs)
    
    count_docs = [Counter(doc) for doc in docs]
    dl = np.array([len(doc) for doc in docs], dtype=np.float64)
    avgdl = dl.mean() if N > 0 else 1.0

    scores = np.zeros(N, dtype=np.float64)
    
    
    for term in query_tokens:
        df = sum(1 for tf in count_docs if tf[term] > 0)
        if df == 0:
            continue

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for i, tf in enumerate(count_docs):
            f = tf[term]
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * dl[i] / avgdl)
            scores[i] += idf * (f * (k1 + 1.0)) / denom

    return scores

