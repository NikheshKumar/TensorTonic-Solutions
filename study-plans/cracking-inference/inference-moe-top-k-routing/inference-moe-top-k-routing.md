# Top-k Routing in a Mixture of Experts

A standard feed-forward layer sends every token through the same parameters. A Mixture of Experts layer contains several feed-forward networks, called experts, and lets a router choose which experts should process each token.

The router is a small decision mechanism. It does not run the experts itself. It receives one score per expert for each token, selects the strongest few scores, and converts those selected scores into weights.

Top-k routing is called sparse because a token uses only $k$ experts instead of every available expert. The model can contain many expert parameter sets while the computation for one token remains limited.

## A simple routing example

Imagine four experts with router scores for one token:

$$
[1.2,-0.3,2.0,0.7]
$$

With $k=2$, experts 2 and 0 are selected because their scores are 2.0 and 1.2. Experts 1 and 3 will not process this token.

Selection alone is not enough. The two chosen experts should not necessarily contribute equally. Applying softmax to the selected scores gives approximately

$$
[0.690,0.310]
$$

for experts 2 and 0 respectively. The larger router score receives the larger share, and the two weights add to one.

## Router logits are preference scores

The input has shape $(T,E)$, where $T$ is the number of tokens and $E$ is the number of experts. Each row describes one token’s preference over all experts.

The scores are logits, so they can be positive, negative, or zero. Their absolute values do not need to add to anything meaningful. Only their ordering determines selection, and their relative gaps determine the weights after softmax.

Routing happens independently for every token. Expert 2 may be the best choice for one token and the worst choice for another. There is no single expert ranking shared across the batch of tokens.

## Select first, normalize second

This exercise uses a sparse top-k softmax. For token $t$, let $S_t$ be the set of selected expert indices. The routing weight for a selected expert is

$$
w_{t,e}=\frac{e^{r_{t,e}}}{\sum_{j\in S_t}e^{r_{t,j}}}
$$

where $r_{t,e}$ is the router logit. Experts outside $S_t$ receive no route and therefore no weight.

The denominator includes only the selected experts. This is different from applying softmax over all experts and then keeping the top $k$ probabilities. Selection would be the same because softmax preserves ordering, but the retained weights would add to less than one unless they were normalized again.

Normalizing only the selected scores directly produces the required mixture weights. Every returned weight is non-negative, and the weights for each token sum to one.

## What happens when k is one?

When $k=1$, each token is sent to its highest-scoring expert. Softmax over a single number is exactly one, regardless of the logit value.

For example, logits $[-4,-2,-7]$ select expert 1 because $-2$ is the largest score. The returned route is expert 1 with weight 1. Negative logits do not imply negative weights.

This case is a useful test because any weight other than one shows that softmax was applied across experts that should not have been included.

## Deterministic ties

Equal router scores need a consistent rule. This problem requires the lower expert index to win a tie.

Suppose a token has scores

$$
[0.5,1.0,1.0,0.2]
$$

With $k=1$, expert 1 must be selected, not expert 2. With $k=2$, the order of the returned indices must be $[1,2]$.

A stable descending sort gives this behavior when the original expert order is $0,1,2,\ldots$. Stability means equal elements keep their original relative order, so the lower index remains first.

Determinism matters beyond testing. If tied scores change routes across runs, identical inputs may reach different parameters and produce different outputs.

## Returned indices and weights stay aligned

The function returns two tensors, each with shape $(T,k)$:

- selected expert indices in descending score order,
- routing weights in the corresponding order.

The entry at column $j$ of the weight tensor belongs to the expert index at column $j$ of the index tensor. Sorting one without applying the same permutation to the other silently assigns weights to the wrong experts.

Using the earlier four-expert example, a valid returned row is indices $[2,0]$ with weights approximately $[0.690,0.310]$. Returning indices $[0,2]$ with those same weights would reverse their intended influence.

## Routing is not expert execution

This problem ends after choosing experts and weights. It does not gather token vectors, evaluate expert networks, or combine expert outputs.

Keeping that boundary clear makes the function easier to reason about. Routing answers two questions for every token:

1. Which experts should receive it?
2. How much should each selected expert contribute?

Dispatch and aggregation use those answers later. They are covered by the next problem.

## Why sparse routing is useful

If a layer has many experts but each token visits only a small number, the model can expose more specialized parameters without evaluating every expert for every token.

For example, a layer may have eight experts while using $k=2$. Each token activates one quarter of the expert networks. Different tokens can activate different quarters, so the parameters used across a whole sequence may still be diverse.

This exercise focuses on the basic routing calculation. Production MoE systems may also consider expert capacity, load balancing, communication across devices, or what to do when an expert receives too many tokens. None of those policies should be invented here because they would change the required routing result.

## Numerical behavior

Softmax should be computed stably on the selected logits. Subtracting the largest selected logit before exponentiation prevents overflow without changing the weights:

$$
w_{t,e}=\frac{e^{r_{t,e}-m_t}}{\sum_{j\in S_t}e^{r_{t,j}-m_t}}
$$

where $m_t$ is the largest selected score for token $t$.

The problem assumes finite logits. Under that condition, selected weights should also be finite. Their row sums should be one within floating-point tolerance.

## Cost and memory

A stable full sort of all $E$ experts for each of $T$ tokens costs $O(TE\log E)$. Selecting the first $k$ entries and applying softmax then costs $O(Tk)$.

The input occupies $O(TE)$ memory, while the returned indices and weights occupy $O(Tk)$. More specialized top-k algorithms can reduce selection work, but a stable full sort is clear and matches the deterministic tie contract.

## Common mistakes to avoid

- Applying softmax over every expert and returning unnormalized top-k probabilities makes the selected weights sum to less than one.
- Sorting in ascending order selects the least preferred experts.
- Using an unstable tie rule can choose a higher expert index when scores are equal.
- Taking top-k across the token dimension compares different tokens instead of different experts.
- Returning indices and weights in different orders attaches influence to the wrong expert.
- Treating negative logits as invalid confuses raw scores with probabilities.
- Adding capacity limits or load-balancing behavior changes the scope of the exercise.

The essential idea is compact: rank experts separately for each token, keep the strongest $k$ with a deterministic tie rule, and normalize only those selected scores into mixture weights.
