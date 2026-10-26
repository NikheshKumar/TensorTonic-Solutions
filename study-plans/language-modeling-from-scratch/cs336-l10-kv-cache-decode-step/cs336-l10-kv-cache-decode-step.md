# Decode Attention with a KV Cache

A language model generates text one token at a time, and each new token attends to every token before it. Recomputing the keys and values of the whole prefix at every step would repeat almost all of the work. The KV cache stores them instead, so each decode step computes keys and values for the new token only and appends them. The heart of this problem is running one decode step of grouped-query attention against such a cache and returning the extended cache.

## Why cache keys and values

In causal attention, the key and value for a past token never change once they are computed, because later tokens cannot affect earlier positions. Only the newest token brings a new query, key and value.

Generation therefore runs in two phases. Prefill processes the whole prompt at once and fills the cache. Decode then produces one token per step. Each step computes the new token's query, key and value, appends the key and value to the cache, and attends from the new query to every cached position. Without the cache, step $t$ would cost work proportional to $t$ tokens of projections. With it, the projections cost one token's worth, and only the attention itself reads the whole prefix.

## One new token against the whole cache

Let $B$ be the batch size, $H_q$ the number of query heads, $H_{kv}$ the number of key-value heads, $D$ the head width and $S$ the number of cached positions. There is one new token per sequence, so the new query has no sequence axis. The new key and value go at position $S$, after every cached position, so the extended caches hold $S + 1$ positions with the old entries in their original order. A cache can start empty, with $S = 0$, in which case the new token attends only to itself.

## Grouped heads read a shared cache

Grouped-query attention keeps fewer key-value heads than query heads and lets several query heads share each one, which shrinks the cache. Each key-value head serves a contiguous group of $G = H_q / H_{kv}$ query heads, and query head $hG + g$ is the $g$-th member of key-value head $h$'s group. With 4 query heads and 2 key-value heads, $G = 2$, so query heads 0 and 1 read key-value head 0, and query heads 2 and 3 read key-value head 1.

Copying each key-value head once per query head would rebuild, during every step, the memory traffic that sharing was meant to remove. The queries in a group are therefore arranged to read the one shared head directly.

## Attention over the extended cache

For batch item $b$, key-value head $h$ and group member $g$, let $q_{b,h,g}$ be query head $hG + g$, let $s$ run over the $S + 1$ positions of the extended cache, and let $k_{b,h,s}$ and $v_{b,h,s}$ be the key and value stored there. The weights over all positions are

$$
a_{b,h,g,s} = \operatorname{softmax}_s\left( \frac{q_{b,h,g}^{\mathsf T} k_{b,h,s}}{\sqrt{D}} \right)
$$

and the output is the weighted sum of values

$$
y_{b,h,g} = \sum_{s=0}^{S} a_{b,h,g,s} \, v_{b,h,s}
$$

No position is masked, because every cached position is already in the past and the new token may attend to itself. Low-precision formats such as bfloat16 round exponentials coarsely, so the attention weights are computed in float32 even when the cache is stored at lower precision.

## A one-head example

Take one sequence, one head of width $D = 2$, and a cache holding one position with key $[1, 0]$ and value $[2, 0]$. The new token has query $[2, 1]$, key $[0, 1]$ and value $[0, 4]$.

After appending, the cache holds keys $[1, 0]$ and $[0, 1]$. The scaled scores are

$$
\frac{[2, 1] \cdot [1, 0]}{\sqrt{2}} = 1.414214, \qquad \frac{[2, 1] \cdot [0, 1]}{\sqrt{2}} = 0.707107
$$

The softmax gives weights of about $0.669762$ and $0.330238$, so the output is

$$
0.669762 \times [2, 0] + 0.330238 \times [0, 4] \approx [1.339523, \ 1.320954]
$$

computed with the unrounded weights. The new token contributes through its own key and value in the same step it was added.

## Why decoding is memory-bound

Each decode step does little arithmetic per byte, because one query per sequence is compared against the entire cache and the model weights are read once to produce a single token. The cost is dominated by reading the weights and the cache from memory, so the cache's size, which grows with sequence length, batch size and the number of key-value heads, directly limits generation speed. Grouped-query attention and lower-precision caches both exist to shrink it.

## Common mistakes to avoid

- **Leaving the new token out of attention.** The new key and value are appended before the softmax, so the token attends to itself.
- **Prepending instead of appending.** The new entries go at position $S$, after all cached positions.
- **Masking the newest position.** A causal mask built for a full sequence can hide the new token from itself, but every position in the extended cache is visible.
- **Mapping query heads to the wrong group.** Query head $hG + g$ uses key-value head $h$.
- **Repeating the cache for each query head.** Copying multiplies the cache's memory by $G$ during the step and defeats the point of grouped-query attention.
- **Failing on an empty cache.** With no cached positions, the new token's own value is the output.
