import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Returns diversity_score and is_collapsed in a dictionary.
    """
    if generated_samples.ndim == 1:
        generated_samples = generated_samples.reshape(-1, 1)

    div_score = np.mean(np.std(generated_samples, axis=0))

    is_collapsed = div_score < threshold

    return {"diversity_score":div_score,"is_collapsed":is_collapsed}