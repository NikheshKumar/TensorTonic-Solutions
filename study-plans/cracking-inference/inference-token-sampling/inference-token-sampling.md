# Sampling the Next Token

A language model does not directly produce a word. For each item in a batch, it produces one score for every token in its vocabulary. These raw scores are called logits. The sampling step turns those scores into one chosen token.

The highest-scoring token is not always the best choice. Always taking it makes generation deterministic, which can be useful for factual or structured tasks, but it can also make open-ended text repetitive. Sampling allows several plausible continuations while still giving stronger candidates a better chance.

This problem builds that choice as a precise sequence of operations. The order matters because temperature, top-k, and top-p do different jobs, and changing their order changes the resulting distribution.

## From logits to probabilities

Suppose the model assigns logits $[2,1,0]$ to three tokens. Logits are not probabilities: they can be negative, they do not add to one, and a score of 2 does not mean a 200 percent chance.

Softmax converts the logits into probabilities:

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

For $[2,1,0]$, the probabilities are approximately $[0.665,0.245,0.090]$. The first token is most likely, but the other two still have a chance.

In practice, softmax should be evaluated with the usual numerical stabilization, such as subtracting the largest remaining logit before exponentiating. A library softmax already handles this concern.

## Temperature controls sharpness

Temperature changes how strongly the sampler prefers the largest logits. For a positive temperature $\tau$, scale the logits before softmax:

$$
z'_i=\frac{z_i}{\tau}
$$

When $\tau$ is below one, differences between logits become larger. The probability distribution becomes sharper, so the leading tokens receive more of the mass. When $\tau$ is above one, the differences shrink and the distribution becomes flatter.

Consider logits $[2,1,0]$. At temperature $0.5$, they become $[4,2,0]$, making the first token much more dominant. At temperature $2$, they become $[1,0.5,0]$, giving the alternatives more room.

Temperature does not decide which tokens are permitted. It reshapes the relative probabilities of all tokens that survive later filters.

This exercise gives temperature zero a special meaning: choose the greedy argmax. There is no division by zero and no random draw. If two logits are tied, use the deterministic argmax behavior required by the tensor operation, which selects the first token index.

## Top-k keeps a fixed number of candidates

Top-k filtering keeps only the $k$ largest scaled logits and removes every other token before softmax. A removed token has probability zero.

If the vocabulary has probabilities led by $[0.40,0.25,0.15,0.12,0.08]$ and $k=2$, only the first two candidates remain. They are renormalized to approximately $[0.615,0.385]$ because the surviving probabilities must add to one.

The size of the candidate set is fixed even when the model is extremely confident or very uncertain. This is both the strength and limitation of top-k. It is simple, but the same $k$ may be too broad in one context and too narrow in another.

In this problem, $k=0$ disables top-k filtering. A positive $k$ keeps at most that many tokens. Ties at the boundary must be resolved deterministically so repeated inputs do not produce a different candidate set.

## Top-p keeps enough probability mass

Top-p, also called nucleus sampling, chooses a candidate set by probability mass rather than by a fixed count.

First sort the probabilities from largest to smallest. Then keep the smallest prefix whose cumulative probability reaches the threshold $p$. A peaked distribution may need only one or two tokens, while a flatter distribution may need many.

Suppose the sorted probabilities are

$$
[0.42,0.28,0.16,0.09,0.05]
$$

With top-p equal to $0.70$, the first two tokens are kept because their cumulative mass is exactly $0.70$. With top-p equal to $0.80$, the first three are kept because the first two reach only $0.70$ and the third raises the total to $0.86$.

The token that crosses the threshold is included. Removing it would leave less probability mass than requested. The first, highest-probability token is always included, even if the threshold is very small.

A convenient way to express the keep rule for each sorted token is

$$
c_i-p_i<p_{\text{threshold}}
$$

where $c_i$ is cumulative probability through the current token. The quantity $c_i-p_i$ is the mass strictly before that token. If the earlier mass is below the threshold, the current token belongs to the smallest prefix needed to reach it.

Top-p equal to one disables nucleus filtering because the full distribution is allowed.

## Why the operation order matters

The required order is:

1. handle greedy decoding when temperature is zero,
2. divide logits by the positive temperature,
3. apply top-k filtering,
4. compute softmax,
5. apply top-p filtering and renormalize,
6. sample from the final distribution.

Top-k works on scaled logits, while top-p works on probabilities. If top-p were applied before top-k, it could choose a nucleus containing tokens that top-k later removes. The final renormalized distribution would then describe a different procedure.

Renormalization after top-p is essential. If the kept tokens contain 0.82 of the previous mass, their probabilities must be divided by 0.82 before sampling. Otherwise their cumulative distribution would stop below one.

## Deterministic sampling from a uniform draw

Random samplers usually generate a uniform number internally. This problem receives one uniform draw $u$ for each batch item, which makes the result reproducible and testable.

After all filtering and renormalization, keep probabilities in natural token-index order and build their cumulative distribution. For probabilities $[0.10,0.60,0.30]$, the cumulative values are $[0.10,0.70,1.00]$.

Choose the first token whose cumulative probability is strictly greater than $u$:

$$
t=\min\left\{i:\sum_{j=0}^{i}p_j>u\right\}
$$

If $u=0.05$, choose token 0. If $u=0.10$, choose token 1 because the comparison is strict. If $u=0.85$, choose token 2.

This is inverse-CDF sampling. A right-sided search implements the strict greater-than rule at probability boundaries. Clamp the result to the final vocabulary index as a guard against tiny floating-point error in a cumulative sum that ends just below one.

Do not sample in descending-probability order unless you correctly map the chosen entry back to its original token index. Restoring natural token order before the cumulative sum makes the returned index directly identify the vocabulary token.

## Following the batch

The logits have shape $(B,V)$, with one row per batch item and one column per vocabulary token. The uniform draws have shape $(B)$, providing one draw for each row.

Every filtering and normalization decision is independent across rows. One item may keep two tokens under top-p while another keeps twenty. The implementation can still represent both distributions in a dense tensor by assigning probability zero to removed entries.

The result has shape $(B)$ and contains one integer token index for every input row.

## Cost and memory

Temperature scaling, top-k masking, softmax, cumulative sums, and the final search are linear in the vocabulary size for each batch item, apart from selection or sorting.

Top-p needs probabilities in descending order. A full sort gives time complexity $O(BV\log V)$ and uses $O(BV)$ working memory. Top-k may be implemented with a partial selection, but the top-p sort remains the dominant general operation when nucleus filtering is enabled.

## Common mistakes to avoid

- Applying top-p to raw logits has no probability-mass interpretation.
- Removing the first token that crosses the top-p threshold keeps too little mass.
- Forgetting to renormalize after filtering produces an invalid sampling distribution.
- Using a greater-than-or-equal boundary for inverse-CDF sampling disagrees with the required right-sided rule.
- Taking the cumulative sum in sorted order and returning that position as a token ID confuses rank with vocabulary index.
- Letting the uniform draw affect temperature-zero decoding breaks greedy behavior.
- Resolving equal scores inconsistently makes the same input produce unstable candidate sets.

The full process can be understood as narrowing and reshaping a set of choices. Temperature adjusts confidence, top-k imposes a fixed candidate limit, top-p adapts that limit to the probability distribution, and the uniform draw makes one reproducible choice from what remains.
