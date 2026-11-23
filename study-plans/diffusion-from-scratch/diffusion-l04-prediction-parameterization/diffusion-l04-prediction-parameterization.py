import torch

def convert_prediction(model_output: torch.Tensor, x_t: torch.Tensor, timesteps: torch.Tensor, alpha_bars: torch.Tensor, prediction_type: str) -> dict:
    """
    Returns a dict with the "x0", "epsilon" and "v" predictions, each with the shape and dtype of x_t.
    """
    if prediction_type not in ("epsilon", "x0", "v"): raise ValueError(f"unknown prediction_type: {prediction_type}")

    alpha_bars_new = alpha_bars[timesteps].view(-1, *[1] * (x_t.dim() - 1))
    a = torch.sqrt(alpha_bars_new)
    b = torch.sqrt(1-alpha_bars_new)
        
    if prediction_type=="epsilon":
        eps = model_output 
        x0 = (x_t - b*eps)/a
        v = a*eps - b*x0
        
    if prediction_type=="x0":
        x0 = model_output
        eps = (x_t - a*x0) / b
        v = a*eps - b*x0
        
    if prediction_type=="v":
        v = model_output 
        x0 = a*x_t - b*v 
        eps = (v + b*x0)/a


    return {"x0":x0, "epsilon":eps, "v":v}
        