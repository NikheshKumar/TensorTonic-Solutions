import numpy as np

def ddpm_sample(
    model_predict: callable,
    shape: tuple,
    betas: np.ndarray,
    T: int
) -> np.ndarray:
    """
    Generate a sample using DDPM.
    """
    # YOUR CODE HERE
    betas = np.asarray(betas, float)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
  
    x = np.random.standard_normal(size=shape)
  
    for t in reversed(range(T)):
      
      t_batch = np.full((shape[0],), t)
      x = (x - (betas[t]/np.sqrt(1-alpha_bars[t]))*model_predict(x, t_batch) )/np.sqrt(alphas[t])
      
      if t > 1:
            z = np.random.standard_normal(size=shape)
            sigma_t = np.sqrt(betas[t]) 
            x = x + sigma_t * z

    return x
        
      
        
