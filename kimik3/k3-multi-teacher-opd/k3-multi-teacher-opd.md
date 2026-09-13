# Multi-Teacher On-Policy Distillation

Multi-Teacher On-Policy Distillation gives a student model a dense reward for tokens it sampled itself. For each batch item, the correct teacher is chosen by domain and reasoning-effort indices. The reward compares how much probability that teacher and the student assigned to the sampled token.

## Why there are several teachers

Kimi K3 uses specialized policies for different domains and reasoning efforts, then consolidates their capabilities into one student. In this exercise, teacher logits have two selection axes: three domains and three effort levels. Together they represent nine possible teachers.

Every batch item selects exactly one of those nine teachers. Once selected, that teacher supplies logits for all sequence positions of that item. Domain and effort selection happens per batch item, not per token.

The function does not average teachers. Mixing their logits would create a distribution that belongs to none of the specialized policies.

## Compare the sampled token only

Student and teacher tensors contain a score for every vocabulary item, but the rollout already tells us which token was sampled at each position. Convert logits to log probabilities with log-softmax, then gather the log probability at that sampled token identifier.

For a sampled token $y_{b,t}$, the unclipped log-ratio reward is

$$
\log\pi_{\mathrm{teacher}}(y_{b,t})-\log\pi_\theta(y_{b,t})
$$

This value is positive when the selected teacher considered the sampled token more likely than the student did. It is negative when the student assigned more probability than the teacher.

Use log-softmax rather than taking softmax and then logarithm. Log-softmax performs the same mathematics more stably when logits have large magnitudes.

## Stop gradients through the reward

The reward is a training signal, not a differentiable path back into the teacher or student probabilities used to calculate it. Detach the log-probability difference before returning the reward.

The selected teacher log probabilities are returned separately and should preserve their ordinary values. The prompt only requires the reward tensor to have no gradient connection. Detaching the reward after subtraction is the clearest way to satisfy this.

## Clip extreme comparisons

Very large log-ratios can dominate learning. Clamp the detached difference to the inclusive interval from negative threshold to positive threshold:

$$
r_{b,t}=\operatorname{clip}\left(\operatorname{stopgrad}[\log\pi_T-\log\pi_S],-R_{\max},R_{\max}\right)
$$

Clipping happens after computing the log-probability difference. Clipping logits or individual log probabilities would change which distribution is being compared.

## A small probability example

Suppose the selected teacher assigns probability $0.6$ to a sampled token, while the student assigns $0.2$. The log-ratio is

$$
\log(0.6)-\log(0.2)=\log(3)\approx1.099
$$

With a clipping threshold of $1.0$, the returned reward is $1.0$. If teacher and student probabilities are equal, the reward is zero. If the teacher gives the token lower probability, the reward is negative.

The selected teacher log probability in this example is $\log(0.6)$. It is returned alongside the clipped reward, not converted back into a probability.

## Select teachers with paired batch indices

The domain index and effort index for batch item $b$ must be used together with that same batch index. This is paired advanced indexing, not a Cartesian selection of all domains, efforts, and batch items.

After teacher selection, both the chosen teacher logits and student logits have one batch, sequence, and vocabulary axis. Gathering sampled token IDs along the vocabulary axis leaves a batch-by-sequence result.

## Implementation order

- Use each batch item's domain and effort indices to select one teacher logit sequence.
- Apply log-softmax across vocabulary to student logits and selected teacher logits.
- Gather each model's log probability for the supplied sampled token at every position.
- Subtract student token log probability from teacher token log probability.
- Detach this difference and clamp it to the requested interval.
- Return clipped rewards followed by selected teacher token log probabilities.

## Common mistakes to avoid

- **Averaging all nine teachers.** Each batch item uses one selected teacher.
- **Gathering before log-softmax.** A single logit is not a log probability because normalization needs the full vocabulary.
- **Comparing the most likely token.** Compare the supplied sampled token, even when it is not an argmax.
- **Leaving reward attached to the graph.** The log-ratio reward must be stopped from backpropagating.
- **Clipping teacher and student values separately.** Clip only their final difference.
