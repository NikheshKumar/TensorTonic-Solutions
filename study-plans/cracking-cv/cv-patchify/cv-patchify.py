import torch

def patchify(images: list, patch_size: int) -> list:
    """
    Returns a float list of shape (B, num_patches, patch_size * patch_size * C).
    """
    images = torch.tensor(images, dtype=torch.float64)

    B, C, H, W = images.shape

    H_new = H//patch_size
    W_new = W//patch_size

    images_new = images.reshape(B, C, H_new, patch_size, W_new, patch_size)
    images_new = images_new.permute(0,2,4,3,5,1)
    images_new = images_new.reshape(B, H_new * W_new, C*patch_size**2)

    return torch.round(images_new, decimals=4).tolist()