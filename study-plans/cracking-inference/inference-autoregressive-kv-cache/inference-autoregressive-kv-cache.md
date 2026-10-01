# Autoregressive Decoding with a KV Cache

An autoregressive model generates one token at a time. After producing a token, it adds that token to the sequence and uses the longer sequence to produce the next one. This repeated use of the growing history creates an obvious question: should the model calculate the same information for earlier tokens again at every step?

A key-value cache avoids that repetition inside attention. Once a token’s key and value have been calculated, they are stored and reused. At the next step, the cache grows by one key and one value, while all earlier entries stay unchanged.

This problem simulates that process from beginning to end for a supplied sequence. It returns the attention output at every position and the final caches that were built along the way.

## Why keys and values can be reused

In self-attention, every token produces a query, a key, and a value. Their roles are different:

- the current query asks what information is relevant,
- cached keys are compared with that query,
- cached values provide the information that is combined.

For a fixed earlier token, its key and value do not change merely because a later token arrives. They were computed from that earlier token’s representation at the same layer, so the decoder can keep them.

The current query is different. It is used only to calculate the output for the current position. Once that output is complete, the next position will have its own query. This is why the cache stores keys and values rather than queries.

## Building the cache one step at a time

Let $k_t$ and $v_t$ be the new key and value at position $t$. After they are appended, the caches contain every entry from position zero through position $t$:

$$
K_{0:t}=[k_0,k_1,\ldots,k_t]
$$

$$
V_{0:t}=[v_0,v_1,\ldots,v_t]
$$

The query at that position attends to these caches:

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{0:t}^{\mathsf T}}{\sqrt{d_k}}\right)V_{0:t}
$$

The new key and value must be appended before calculating $o_t$. A token is allowed to attend to itself, so position $t$ needs access to $k_t$ and $v_t$ as well as all earlier entries.

## A three-token example

Consider positions 0, 1, and 2.

At position 0, append $k_0$ and $v_0$. The cache contains one entry, so $q_0$ attends only to position 0.

At position 1, preserve the first entries and append $k_1$ and $v_1$. The new query $q_1$ attends to positions 0 and 1.

At position 2, append $k_2$ and $v_2$. The query $q_2$ attends to positions 0, 1, and 2.

The cache lengths therefore grow as one, two, and three. Earlier outputs are not recalculated after a later cache entry is added.

This is causal attention without an explicit triangular mask. Each query can only see the cache that exists at its own step, so future positions are absent by construction.

## Why the result matches dense causal attention

Dense causal attention processes all positions together and masks scores above the causal diagonal. Its row for position $t$ retains keys zero through $t$ and removes every later key.

Cached decoding processes that same row using a cache whose length is exactly $t+1$. Both methods use the same query, the same allowed keys, the same values, and the same scaled softmax. They must therefore produce the same $o_t$ within floating-point tolerance.

The methods organize the work differently. Dense causal attention exposes the complete causal score matrix, while cached decoding exposes one valid row at each step.

Agreement with dense causal attention is the strongest correctness check for the returned output sequence.

## The cache is append-only

Append-only means two things:

- a new entry is placed after all existing entries,
- no existing key or value is recomputed, reordered, or modified.

After the last position, the key cache must equal the supplied key sequence in its original order. The value cache must similarly equal the complete supplied value sequence.

This final equality is easy to test and catches several subtle errors. A cache may have the right length while containing duplicated, reversed, or overwritten entries.

In this exercise, concatenating one-position slices is a clear way to model growth. A production cache often reserves storage and writes into the next slot to avoid repeated memory copies, but that storage optimization does not change the append-only meaning.

## Following the tensor shapes

Queries and keys have shape $(B,S,d_k)$. Values have shape $(B,S,d_v)$, where the value width may differ from the key width.

At step $t$:

- the current query has shape $(B,1,d_k)$,
- the key cache has shape $(B,t+1,d_k)$,
- the value cache has shape $(B,t+1,d_v)$,
- the attention scores have shape $(B,1,t+1)$,
- the step output has shape $(B,1,d_v)$.

The single-position outputs are collected along the sequence dimension. The final output has shape $(B,S,d_v)$.

Batch items share the same sequence length in these tensors, but their attention calculations remain independent. A cache entry from one batch item must never be mixed with another.

## Scaling and softmax still matter

The query-key dot products are divided by $\sqrt{d_k}$ before softmax. This is the same scaling used by ordinary dot-product attention. Omitting it changes the probabilities and prevents agreement with the dense reference.

Softmax is applied across the cached-position dimension. Each current query receives one probability for every available cache entry, and those probabilities sum to one.

The weighted sum uses the value cache, so its last dimension is $d_v$. The attention score width depends on cache length, while the output width depends on the values.

## Computation and memory

At step $t$, the current query compares with $t+1$ cached keys. Across an entire sequence, the attention comparisons still total

$$
1+2+\cdots+S=O(S^2)
$$

for each batch item, apart from feature dimensions. KV caching does not remove the need for a new query to inspect its history.

Its benefit is avoiding repeated calculation of earlier keys and values. Each supplied key and value is appended once and reused thereafter.

The final caches use $O(BS(d_k+d_v))$ memory. The output uses $O(BSd_v)$ memory. The conceptual cache grows linearly with sequence length even though the total attention work across all positions is quadratic.

The simple repeated-concatenation simulation may copy existing cache data as it grows. That copying cost belongs to the educational representation, not to the logical KV-cache algorithm.

## Common mistakes to avoid

- Calculating attention before appending the current key and value incorrectly prevents self-attention.
- Rebuilding the complete key and value history at every step defeats the purpose of the cache.
- Running every earlier query again after each append recomputes outputs that are already final.
- Applying softmax over the feature dimension normalizes the wrong axis.
- Assuming $d_v=d_k$ can produce incorrect output allocation or matrix multiplication.
- Returning only the final token’s output loses the outputs required for earlier positions.
- Modifying or reordering cached entries breaks the final-cache contract.

The main idea is simple: calculate the current row of causal attention using a history that grows by one immutable key-value pair per step. The growing history produces the same outputs as dense causal attention while making the reuse of earlier keys and values explicit.
