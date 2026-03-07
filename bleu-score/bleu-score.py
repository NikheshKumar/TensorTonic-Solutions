def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    # Write code here
    import numpy as np 

    prec = []
    r = len(reference)
    c = len(candidate)

    if c == 0:
        return 0.0

    for n in range(1, max_n+1):

      cng = [tuple(candidate[i:i+n]) for i in range(c-n+1)]
      rng = [tuple(reference[i:i+n]) for i in range(r-n+1)]

      if cng is None:
        prec.append(0.0)

      uniques = set(cng)
      tot = 0

      for i in uniques:
        count_c = cng.count(i)
        count_r = rng.count(i)
        tot += min(count_c, count_r)

      prec.append( tot / len(cng) )
        
      
    prec = np.asarray(prec, float)
    if np.any(prec == 0):
        return 0.0

    # if c >= r:
    #     bp = 1.0
    # else:
    #     bp = np.exp(1 - r / c) 

    bp = np.exp(np.min([0, 1 - r/c]))

    prec = np.where(prec == 0, 1e-7, prec)
    
    score = bp * np.exp(np.mean(np.log(prec)))

    return score