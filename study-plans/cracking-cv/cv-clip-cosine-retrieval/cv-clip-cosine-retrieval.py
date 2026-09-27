import numpy as np

def cosine_topk(queries: list, corpus: list, k: int) -> dict:
    """
    Returns a dictionary with indices (M by k ints) and scores (M by k floats).
    """
    queries = np.array(queries, dtype=np.float64)
    corpus = np.array(corpus, dtype=np.float64)

    eps = 1e-12

    q_norm = np.linalg.norm(queries, axis=1, keepdims=True) + eps
    c_norm = np.linalg.norm(corpus, axis=1, keepdims=True) + eps

    Q = queries / q_norm
    C = corpus / c_norm

    s = np.sum(Q[:, None, :] * C[None, :, :], axis=-1)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    indices = np.argsort(-s, kind="stable", axis=1)[:, :k]
    scores = np.take_along_axis(s, indices, axis=1)

    scores = [[round(float(s),4) for s in row] for row in scores]
    indices = [[int(indices[i][j]) for j in range(k)] for i in range(queries.shape[0])]
    

    return {"indices":indices , "scores":scores}
    