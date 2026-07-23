# <span style="font-size: 20px;">Implement Multi-Query Attention (MQA)</span>

<span style="font-size: 14px;">Multi-query attention keeps many query heads but shares one key head and one value head, reducing autoregressive cache traffic while retaining multiple query subspaces.</span>

---

## <span style="font-size: 16px;">The tradeoff relative to standard multi-head attention</span>

<span style="font-size: 14px;">Standard multi-head attention gives every query head an independent key head and an independent value head. This means that at inference time, every head's keys and values are distinct tensors that must be cached and read separately during autoregressive generation, so the KV-cache memory and the memory bandwidth needed to read it back scale linearly with the number of heads.</span>

<span style="font-size: 14px;">Multi-query attention removes this scaling entirely by collapsing all key and value heads into one: every query head still has its own learned projection and therefore its own notion of what to attend to, but all of them compare against, and aggregate from, the exact same key and value tensor.</span>

<span style="font-size: 14px;">This is the most aggressive point on the query-to-key/value sharing spectrum: standard multi-head attention shares nothing, multi-query attention shares everything, and grouped-query attention, covered next, sits at every point in between.</span>

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K^\top}{\sqrt{d_k}} + M\right) V
$$

$$
h = 0, \ldots, num\_query\_heads - 1
$$

---

## <span style="font-size: 16px;">Why there is no key/value head dimension at all</span>

<span style="font-size: 14px;">In standard multi-head attention, $K$ and $V$ are projected to the full $d_{model}$ width and then split into $num\_heads$ heads of size $d_k$, exactly mirroring how $Q$ is split. In multi-query attention, $W_k$ and $W_v$ project directly to a single $d_k$-wide tensor, with no larger dimension to split in the first place. This is not merely an optimization applied after the fact; it is a different parameterization from the start, with far fewer key/value parameters.</span>

<span style="font-size: 14px;">The absence of a head dimension on $K$ and $V$ is what makes sharing exact rather than approximate: there is only one key tensor and one value tensor, and every query head is compared against literally the same data, not against separately learned copies that happen to be initialized identically.</span>

---

## <span style="font-size: 16px;">Why broadcasting, not repetition, is the natural implementation</span>

<span style="font-size: 14px;">Because every query head uses the identical key and value tensor, there is no need to construct $num\_query\_heads$ separate copies of $K$ and $V$ before the attention computation.</span>

<span style="font-size: 14px;">Giving $K$ and $V$ a size-1 head dimension and relying on standard batched-matrix-multiply broadcasting produces mathematically the same result as explicitly repeating the key/value tensor $num\_query\_heads$ times, since a broadcast dimension of size 1 is treated as if it were repeated to match, without allocating new memory for the repetition.</span>

<span style="font-size: 14px;">This is a meaningful distinction from grouped-query attention's general implementation, which must materialize an expanded key/value tensor with repeat_interleave because different query heads there can map to different, non-identical key/value heads.</span>

<span style="font-size: 14px;">In multi-query attention, every query head maps to the same one, so the broadcast approach is both correct and strictly cheaper, and understanding this distinction clarifies why multi-query attention is frequently implemented as its own direct code path rather than always being routed through a general grouped-query attention implementation with the key/value head count hardcoded to 1.</span>

---

## <span style="font-size: 16px;">The representational cost being traded away</span>

<span style="font-size: 14px;">Because every query head compares against the same keys and produces its weighted sum from the same values, the only way different query heads can produce different outputs is through their own query projection, which determines which positions each head weights most heavily; the values being aggregated are identical across heads.</span>

<span style="font-size: 14px;">This is a genuine representational restriction relative to standard multi-head attention, where each head could also learn to extract different information from the same position through an independent value projection.</span>

<span style="font-size: 14px;">In practice this restriction is often acceptable, since a large fraction of what multiple heads contribute comes from attending to different positions rather than extracting different features from the same position, but it is not free, and is the reason grouped-query attention exists as a middle ground: recovering some of this lost flexibility by giving groups of query heads their own shared key/value head, rather than forcing every query head in the entire layer to share one.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">Multi-query attention was introduced specifically to address memory bandwidth during autoregressive decoding, not to improve model quality or reduce compute in the way FlashAttention or quantization do. During generation, each new token requires reading the entire KV cache for every head to compute attention scores against all previous positions.</span>

<span style="font-size: 14px;">With standard multi-head attention, this read volume scales with the number of heads; with multi-query attention, exactly one key/value pair per token needs to be stored and read, independent of how many query heads the model uses.</span>

<span style="font-size: 14px;">On hardware where the decoding step is memory-bandwidth-bound, which is the common case for large models generating one token at a time, this reduction in KV-cache read volume directly translates into faster decoding and the ability to serve larger batch sizes within the same memory budget, even though the total floating-point operation count for the attention computation itself is essentially unchanged.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">The query projection costs $O(B \cdot S \cdot d_{model}^2)$, unchanged from standard multi-head attention. The key and value projections cost only $O(B \cdot S \cdot d_{model} \cdot d_k)$, a factor of $num\_query\_heads$ cheaper than projecting to the full $d_{model}$ width, since the output is a single $d_k$-wide tensor rather than $num\_query\_heads$ of them.</span>

<span style="font-size: 14px;">The attention computation itself, scores, masking, softmax, and the weighted sum, costs $O(B \cdot num\_query\_heads \cdot S^2 \cdot d_k)$, identical to standard multi-head attention, since broadcasting affects memory and projection cost, not how many per-head score matrices must be computed.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">The key and value projection weights, and the tensors they produce, are smaller by a factor of $num\_query\_heads$ compared to standard multi-head attention.</span>

<span style="font-size: 14px;">This is the entire mechanism behind multi-query attention's inference benefit: the KV cache stores one key and one value vector per token, not $num\_query\_heads$ of them, which is the maximum possible reduction achievable by sharing key/value heads across query heads, more aggressive than any grouped-query attention configuration with more than one key/value head.</span>

<span style="font-size: 14px;">The attention score and weight tensors during computation remain $(B, num\_query\_heads, S, S)$, the same size as standard multi-head attention, since the memory savings are entirely on the key/value side, not in the attention matrices themselves.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Giving $K$ or $V$ a head dimension and splitting them the same way as $Q$, which silently turns the implementation back into standard multi-head attention with an unused, mismatched head count.</span>

* <span style="font-size: 14px;">Materializing a repeated copy of $K$ or $V$ for every query head instead of relying on broadcasting, which is not incorrect but defeats the memory-efficiency motivation the architecture exists for.</span>

* <span style="font-size: 14px;">Sizing $W_k$ or $W_v$ to $d_{model}$ instead of $d_k$, which produces a key/value tensor that cannot broadcast correctly against the per-head query tensor.</span>

* <span style="font-size: 14px;">Forgetting that num_query_heads=1 should exactly match single-head attention, which is the most sensitive check for a subtly wrong broadcasting or shape implementation.</span>

* <span style="font-size: 14px;">Assuming multi-query attention is only a memory optimization with no representational consequence, and being surprised when it measurably changes model quality relative to standard multi-head attention on some tasks.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$X$ denotes hidden states, $h_q$ is the number of query heads, $d_k=d_{model}/h_q$ is one head width, $Q_h$ is query head $h$, and the unindexed $K$ and $V$ are the single shared key and value tensors.</span>

<span style="font-size: 14px;">Queries have shape $(B,h_q,S,d_k)$, shared keys and values have shape $(B,1,S,d_k)$, scores have shape $(B,h_q,S,S)$, and the result has shape $(B,S,d_{model})$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: hidden_states.shape = (1, 2, 4), num_query_heads = 1, causal = False</span>

<span style="font-size: 14px;">Output: tensor of shape (1, 2, 4), e.g. row 0 is [0.3085, -0.4199, -0.3009, 0.3859]</span>

<span style="font-size: 14px;">Explanation: With num_query_heads = 1, there is only one query head and one key/value pair, which is exactly single-head attention.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">The paper One Write-Head Is All You Need proposed MQA to reduce memory bandwidth during incremental decoding. The query side remains multi-headed, while all query heads read a single cached key stream and value stream.</span>

<span style="font-size: 14px;">This changes the serving bottleneck more than the prefill arithmetic. Decode repeatedly reads the cache for one new query, so reducing KV bytes can raise token throughput even when the number of query-key comparisons stays similar.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">A size-one head axis on keys and values broadcasts the same KV states across every query head.</span>

<span style="font-size: 14px;">Each query head still receives its own score matrix, so query-side specialization is preserved.</span>

<span style="font-size: 14px;">Concatenating query-head outputs and applying the output projection restores the model-width contract.</span>

<span style="font-size: 14px;">With $h_q=1$, broadcasting is a no-op and the computation reduces to ordinary single-head attention.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">Validation should compare the result with an explicit reference that repeats the shared KV tensors, then confirm the vectorized implementation produces identical values without storing those repetitions.</span>

<span style="font-size: 14px;">The cache-saving claim can be checked directly by counting stored elements per token: MQA uses $2d_k$ values per layer, independent of the number of query heads.</span>

---

## <span style="font-size: 16px;">Behavioral contract</span>

<span style="font-size: 14px;">Accept hidden states, query, key, value, and output projection matrices, a positive query-head count, and a causal flag.</span>

<span style="font-size: 14px;">Return a tensor with the same batch size, sequence length, and model width as the hidden states.</span>

<span style="font-size: 14px;">Every query head must use the same key tensor and the same value tensor.</span>

<span style="font-size: 14px;">When causal mode is enabled, output position $i$ must not depend on any later input position.</span>

<span style="font-size: 14px;">The one-query-head case must match single-head scaled dot-product attention with the supplied projections.</span>

<span style="font-size: 14px;">The output must match the mathematical MQA definition without materializing distinct learned key or value heads.</span>

<span style="font-size: 14px;">The result must be finite for every valid input.</span>

---