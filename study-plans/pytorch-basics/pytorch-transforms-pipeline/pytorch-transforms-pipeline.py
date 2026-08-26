import torch

class TransformPipeline:
    """
    Returns: float32 tensor of shape (C, H, W) from __call__
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).reshape(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).reshape(-1, 1, 1)

    def __call__(self, image):
        
        image = image / 255.0
        image = image.permute(2,0,1)

        image = (image - self.mean) / self.std

        return image
        
