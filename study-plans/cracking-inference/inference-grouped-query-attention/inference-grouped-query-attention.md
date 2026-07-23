# <span style="font-size: 20px;">Implement Grouped-Query Attention (GQA)</span>

<span style="font-size: 14px;">Grouped-query attention assigns several query heads to each key and value head, creating a tunable compromise between MHA quality and MQA cache efficiency.</span>

---

## <span style="font-size: 16px;">The tradeoff GQA sits inside</span>

<span style="font-size: 14px;">Standard multi-head attention gives every query head its own independent key and value head: full representational flexibility, but every head's keys and values must be cached separately during autoregressive generation, so KV-cache memory scales linearly with the number of heads.</span>

<span style="font-size: 14px;">Multi-query attention takes the opposite extreme: every query head shares one single key/value head, cutting KV-cache memory by a factor of $num\_query\_heads$, at the cost of every head being forced to look at exactly the same keys and values, which can measurably hurt model quality when the heads genuinely needed to attend differently.</span>

<span style="font-size: 14px;">Grouped-query attention is the interpolation between these two extremes: choose some number of key/value heads strictly between $1$ and $num\_query\_heads$, and let groups of query heads share each one. This lets a model recover most of multi-query attention's memory savings while keeping enough independent key/value heads that different groups of query heads can still specialize in different notions of relevance.</span>

$$
\text{head}_h = \text{softmax}\!\left(\frac{Q_h K_{g(h)}^\top}{\sqrt{d_k}} + M\right) V_{g(h)}
$$

$$
g(h) = \left\lfloor \frac{h}{\text{group\_size}} \right\rfloor
$$

---

## <span style="font-size: 16px;">Why the mapping must be exact and consecutive</span>

<span style="font-size: 14px;">group_size = num_query_heads / num_kv_heads must be an integer, since every query head needs exactly one key/value head, and a fractional group size would leave some query heads without a well-defined assignment. Given an integer group size, the mapping groups query heads into consecutive blocks: heads $0, \ldots, \text{group\_size}-1$ map to key/value head $0$, heads $\text{group\_size}, \ldots, 2 \cdot \text{group\_size}-1$ map to key/value head $1$, and so on.</span>

<span style="font-size: 14px;">This specific consecutive grouping, rather than some other assignment such as interleaving, is the convention used throughout production GQA implementations, and it is what makes repeat_interleave the correct expansion primitive: it repeats each key/value head in place before advancing to the next one, reproducing exactly this consecutive block structure.</span>

<span style="font-size: 14px;">A different tensor operation, ordinary repeat (tiling the entire sequence of key/value heads as a whole, multiple times), produces a different and incompatible mapping: it would assign query head $1$ to key/value head $1$ (if $num\_kv\_heads > 1$) whenever $group\_size > 1$, rather than assigning it back to key/value head $0$. Both operations produce a tensor of the same final shape, so this is exactly the kind of error that executes without raising any exception while silently computing an architecturally different, incorrect result.</span>

---

## <span style="font-size: 16px;">Why validating divisibility matters beyond correctness</span>

<span style="font-size: 14px;">Rejecting invalid (num_query_heads, num_kv_heads) combinations before computing anything is not only about producing a clean error message. Continuing past an invalid combination with, for example, a fractional group_size truncated by integer division would silently drop some query heads from any key/value head entirely, or in the worst case, misassign query heads asymmetrically depending on how the truncation happens to fall, producing a model that runs without error but attends incorrectly, in a way that is extremely difficult to detect through normal testing since the output shapes are still correct.</span>

---

## <span style="font-size: 16px;">The two boundary equivalences as a correctness anchor</span>

<span style="font-size: 14px;">Because grouped-query attention is a strict generalization of both standard multi-head attention and multi-query attention, its two boundary cases are the most reliable way to check a GQA implementation for correctness against a simpler, already-understood reference. Setting num_kv_heads = num_query_heads gives group_size = 1, under which the expansion step is a no-op and every query head has its own key/value head, exactly reproducing standard multi-head attention with no approximation.</span>

<span style="font-size: 14px;">Setting num_kv_heads = 1 gives group_size = num_query_heads, under which every query head is expanded from the same single key/value head, exactly reproducing multi-query attention. Any GQA implementation that fails either equivalence has a bug in either the head-splitting, the expansion, or the projection-shape handling, since these two cases exercise the full mapping logic at both of its extremes.</span>

---

## <span style="font-size: 16px;">Why the key/value projection shapes differ from query</span>

<span style="font-size: 14px;">$W_q$ projects to $num\_query\_heads \cdot d_k = d_{model}$ features, matching the full model dimension, since every query head needs its own slice. $W_k$ and $W_v$ project to only $num\_kv\_heads \cdot d_k$ features, a strictly smaller output dimension whenever $num\_kv\_heads < num\_query\_heads$.</span>

<span style="font-size: 14px;">This asymmetry is the direct source of GQA's parameter and compute savings on the key/value side: fewer output features means smaller weight matrices, smaller projected tensors, and, most importantly for inference, a smaller per-token key/value cache footprint, since the cache stores the post-projection, pre-expansion key and value tensors rather than the expanded versions.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">During autoregressive decoding, the KV cache must store one key and one value vector per cached token per key/value head, and this cache is read back at every subsequent generation step. With standard multi-head attention, this cost scales with $num\_query\_heads$; with grouped-query attention, it scales with $num\_kv\_heads$, which can be many times smaller.</span>

<span style="font-size: 14px;">This is why production LLM architectures that must serve long contexts or large batch sizes overwhelmingly choose GQA over standard multi-head attention: it is one of the few architectural choices that directly reduces the memory bandwidth cost of serving, which is frequently the dominant bottleneck during token-by-token decoding, while remaining close in output quality to standard multi-head attention when $num\_kv\_heads$ is chosen to not be too small.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">The query projection is $O(B \cdot S \cdot d_{model}^2)$, unchanged from standard multi-head attention. The key/value projections cost $O(B \cdot S \cdot d_{model} \cdot num\_kv\_heads \cdot d_k)$, smaller by a factor of $\text{group\_size}$ compared to projecting to $num\_query\_heads \cdot d_k$ features. The expansion itself, repeat_interleave, only duplicates existing values and costs $O(B \cdot num\_query\_heads \cdot S \cdot d_k)$, negligible next to the projections.</span>

<span style="font-size: 14px;">The attention computation after expansion costs $O(B \cdot num\_query\_heads \cdot S^2 \cdot d_k)$, identical to standard multi-head attention, since expansion does not reduce how many query-head attention computations must be performed.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">The key/value weight matrices and their projected outputs are smaller by exactly $\text{group\_size}$ compared to standard multi-head attention. During inference, this is the parameter directly responsible for KV-cache size: caching post-projection, pre-expansion keys and values means the cache holds $num\_kv\_heads$ worth of vectors per token, not $num\_query\_heads$ worth.</span>

<span style="font-size: 14px;">The expanded score and weight tensors used during the attention computation itself remain $(B, num\_query\_heads, S, S)$, the same size as standard multi-head attention, since expansion happens before scoring; grouped-query attention reduces cache and projection memory, not the size of the attention matrices themselves.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Using repeat instead of repeat_interleave for key/value expansion, which produces a valid shape but an incorrect query-head-to-key/value-head mapping whenever group_size > 1.</span>

* <span style="font-size: 14px;">Computing group_size with true division instead of integer division, or failing to validate divisibility first, which can silently truncate or misassign heads instead of raising a clear error.</span>

* <span style="font-size: 14px;">Sizing $W_k$ or $W_v$ to $num\_query\_heads \cdot d_k$ features instead of $num\_kv\_heads \cdot d_k$, which defeats the entire memory-saving purpose of the architecture while still producing correct-looking shapes after an incorrect expansion.</span>

* <span style="font-size: 14px;">Forgetting to test the two boundary equivalences, which are the most sensitive checks for a subtly wrong head-mapping implementation.</span>

* <span style="font-size: 14px;">Caching the expanded, rather than the pre-expansion, key and value tensors in a real serving system, which silently reintroduces the full multi-head-attention memory cost this architecture exists to avoid.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$h_q$ is the number of query heads, $h_{kv}$ is the number of key and value heads, $g=h_q/h_{kv}$ is the group size, and query head $h$ uses KV head $lfloor h/gfloor$.</span>

<span style="font-size: 14px;">Queries have shape $(B,h_q,S,d_k)$, compact keys and values have shape $(B,h_{kv},S,d_k)$, expanded KV tensors have shape $(B,h_q,S,d_k)$, and outputs have shape $(B,S,d_{model})$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: hidden_states.shape = (1, 2, 4), num_query_heads = 2, num_kv_heads = 2, causal = False</span>

<span style="font-size: 14px;">Output: tensor of shape (1, 2, 4), e.g. row 0 is [0.3467, -0.4255, -0.2473, 0.3479]</span>

<span style="font-size: 14px;">Explanation: With num_kv_heads = num_query_heads, this must equal standard multi-head attention.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">The GQA paper formalized an intermediate architecture between MHA and MQA and showed how models can use fewer KV heads without collapsing all query heads onto one shared representation.</span>

<span style="font-size: 14px;">Modern inference stacks expose KV-head count as an architectural quantity because it directly determines cache capacity, cache-read bandwidth, and the number of concurrent sequences that fit on a device.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">Repeating each KV head by the group size creates the required consecutive query-to-KV mapping.</span>

<span style="font-size: 14px;">Divisibility validation prevents an incomplete group and makes both boundary equivalences exact.</span>

<span style="font-size: 14px;">All expanded heads share the same causal visibility rule while retaining independent query projections.</span>

<span style="font-size: 14px;">The MHA and MQA boundaries are not approximations; they are exact special cases of the same head mapping.</span>

---

## <span style="font-size: 16px;">Validation strategy</span>

<span style="font-size: 14px;">The strongest checks use the two exact boundaries, plus an intermediate case where several consecutive query heads must map to the same KV head. A nonconsecutive mapping should fail value comparison even though every tensor shape remains valid.</span>

<span style="font-size: 14px;">A causal test should vary future tokens and confirm that outputs at earlier positions remain unchanged for every group.</span>

---