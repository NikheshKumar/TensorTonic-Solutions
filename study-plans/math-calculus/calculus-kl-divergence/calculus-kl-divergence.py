import numpy as np

def kl_divergence_analysis(P: list, Q: list, mu1: float, sigma1: float, mu2: float, sigma2: float, n: int) -> dict:
    """
    Returns discrete, numerical Gaussian, and analytical Gaussian KL values.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    mask = (P > 0) & (Q > 0)
    discrete_kl = float(np.sum(P[mask] * (np.log(P[mask]) - np.log(Q[mask]))))

    analytical_kl = np.log(sigma2) - np.log(sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2.0 * (sigma2**2)) - 0.5

    if n%2:
        n = n+1
        
    a = min(mu1, mu2) - 5 * max(sigma1, sigma2) 
    b = max(mu1, mu2) + 5 * max(sigma1, sigma2)
    h = (b-a)/n

    x = np.linspace(a,b,n+1)

    log_p = -np.log(sigma1) - 0.5*np.log(2.0*np.pi) - (x - mu1)**2/(2.0*sigma1**2)
    log_q = -np.log(sigma2) - 0.5*np.log(2.0*np.pi) - (x - mu2)**2/(2.0*sigma2**2)
    p = np.exp(log_p)
    q = np.exp(log_q)

    f = np.zeros_like(p, dtype=np.float64)
    mask = (p>0) & (q>0)
    f[mask] = p[mask] * (np.log(p[mask]) - np.log(q[mask]))
    
    i = np.arange(n+1)
    w = np.where(i%2==0, 2.0, 4.0)
    w[0] = w[-1] = 1.0
    
    numerical_kl = float((h / 3.0) * np.sum(w * f))

    
    return {"discrete_kl": discrete_kl, "numerical_kl": numerical_kl, "analytical_kl": analytical_kl}

    