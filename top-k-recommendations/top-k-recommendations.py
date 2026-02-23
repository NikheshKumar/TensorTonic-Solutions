def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Write code here
    import numpy as np 

    rated_indices = set(rated_indices)

    res = []

    for i in range(len(scores)):
      if i in rated_indices:
        continue
      else:
        res.append((scores[i], i))

    res.sort(key=lambda x: -x[0])

    topk = res[:k]
    ans = [item[1] for item in topk]
  
    return ans
    