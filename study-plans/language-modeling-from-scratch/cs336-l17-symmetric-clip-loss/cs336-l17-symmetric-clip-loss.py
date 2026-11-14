import torch
import torch.nn.functional as F

def symmetric_clip_loss(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: int | float,
) -> dict:
    """
    Returns a dict: loss and two directional accuracies (scalars), similarities (tensor).
    """
    def normalize(x):
        n = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
        return x / torch.clamp(n, min=1e-12)

    S = normalize(image_embeddings) @ normalize(text_embeddings).T / temperature

    B = S.shape[0]
    diag = torch.arange(B, device=S.device)

    log_p1 = F.log_softmax(S, dim=-1)
    log_p2 = F.log_softmax(S, dim=0)
    
    term1 = log_p1[diag, diag]
    term2 = log_p2[diag, diag]

    L = -0.5 * torch.mean(term1 + term2)

    acc1 = (S.argmax(dim=-1) == diag).float().mean()

    acc2 = (S.argmax(dim=0) == diag).float().mean()

    return {"loss":L, "similarities":S, "image_to_text_accuracy":acc1, "text_to_image_accuracy":acc2}
