import torch

def multi_teacher_opd_reward(student_logits: torch.Tensor, teacher_logits: torch.Tensor, domain_indices: torch.Tensor, effort_indices: torch.Tensor, sampled_tokens: torch.Tensor, clip_threshold: float) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing clipped rewards and selected teacher token log probabilities.
    """

    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1) 

    B, seq_len, V = student_logits.shape

    batch_indices = torch.arange(B, dtype=torch.long)

    teacher_log_probs = teacher_log_probs[domain_indices,effort_indices,batch_indices]

    token_indices = sampled_tokens.unsqueeze(-1).long()

    student_token_log_probs = student_log_probs.gather(dim=-1, index=token_indices).squeeze(-1)
    teacher_token_log_probs = teacher_log_probs.gather(dim=-1, index=token_indices).squeeze(-1)

    reward = teacher_token_log_probs - student_token_log_probs
    reward = reward.detach()
    reward = torch.clamp(reward, -clip_threshold, clip_threshold)

    return {
        "rewards": reward,
        "teacher_token_log_probs": teacher_token_log_probs
    }