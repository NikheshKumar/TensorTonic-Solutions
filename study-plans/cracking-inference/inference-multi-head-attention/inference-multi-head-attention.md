# <span style="font-size: 20px;">Implement Multi-Head Attention (MHA)</span>

<span style="font-size: 14px;">Multi-head attention lets a transformer compare the same sequence through several learned query, key, and value subspaces while preserving the model width at the output.</span>

---

## <span style="font-size: 16px;">From one attention function to multiple heads</span>

<span style="font-size: 14px;">Scaled dot-product attention, the previous problem, computes a single weighted average per query, using one shared $d_k$-dimensional subspace to decide relevance. Multi-head attention runs that same computation $h$ times in parallel, each time projecting the input into a different, independently learned $d_k$-dimensional subspace, then combines the results.</span>

<span style="font-size: 14px;">The motivation is representational, not just computational: a single attention head can only express one notion of "relevance" between positions, since its similarity scores come from one fixed projection. Multiple heads let the model attend to different positions for different reasons simultaneously, for example one head tracking syntactic adjacency while another tracks longer-range semantic association, without either head's signal diluting the other's.</span>

$$
\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}} + M\right) V_i
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W_o
$$

---

## <span style="font-size: 16px;">Where the heads come from</span>

<span style="font-size: 14px;">$Q = X W_q$, $K = X W_k$, and $V = X W_v$ are each full $(batch, seq, d_{model})$ tensors, computed with one shared projection matrix per query, key, and value, not $h$ separate matrices. The split into heads happens afterward, by reinterpreting the $d_{model}$ feature dimension as $h$ groups of $d_k = d_{model} / h$ contiguous features.</span>

<span style="font-size: 14px;">This is a deliberate parameterization choice: a single $(d_{model}, d_{model})$ matrix per projection is equivalent in expressive power to $h$ independent $(d_{model}, d_k)$ matrices, one per head, since the columns of the shared matrix simply concatenate what the per-head matrices would have computed separately.</span>

<span style="font-size: 14px;">Implementations universally use the single shared-matrix form because it is one matrix multiply instead of $h$ smaller ones, which is significantly more efficient on real hardware even though the arithmetic is identical.</span>

---

## <span style="font-size: 16px;">Why splitting requires a specific reshape order</span>

<span style="font-size: 14px;">After projection, $Q$ has shape $(batch, seq, d_{model})$. Reinterpreting the last axis as $(num\_heads, d_k)$ only produces the correct per-head slices if the reshape is applied directly to that axis, before any dimension reordering: view(batch, seq, num_heads, d_k) groups the $d_{model}$ features into num_heads contiguous chunks of size $d_k$ each, in the same order the weight matrix's output columns were arranged.</span>

<span style="font-size: 14px;">Transposing to $(batch, num\_heads, seq, d_k)$ afterward only moves the head dimension next to batch for broadcasting purposes; it does not change which features belong to which head. Doing the transpose before the reshape, or reshaping a tensor that is not contiguous in the expected layout, silently produces a tensor with the right shape but scrambled features, which will not raise an error but will compute a mathematically different, incorrect result.</span>

---

## <span style="font-size: 16px;">Per-head attention and batched heads</span>

<span style="font-size: 14px;">Once split, every head's $Q_i, K_i, V_i$ has shape $(batch, seq, d_k)$, and the attention computation applied to each head is exactly scaled dot-product attention as already established: scores scaled by $1/\sqrt{d_k}$, an optional additive mask, a numerically stable softmax, and a weighted sum of values.</span>

<span style="font-size: 14px;">Because PyTorch's batched matrix multiply operates over all leading dimensions simultaneously, stacking the head dimension alongside batch means every head's attention is computed with the same operations used for a single head, without an explicit loop over heads.</span>

<span style="font-size: 14px;">The score tensor becomes $(batch, num\_heads, seq, seq)$, one full attention matrix per head per batch element, and the causal mask, when used, broadcasts identically across both the batch and head dimensions since the masking pattern (which positions may attend to which) does not depend on which head or which batch element is being processed.</span>

---

## <span style="font-size: 16px;">Concatenation and the output projection</span>

<span style="font-size: 14px;">After per-head attention, each head produces a $(batch, seq, d_k)$ output. Concatenating the heads means placing each head's output back into its original slice of the $d_{model}$ feature dimension, which requires transposing the head dimension back next to seq and then reshaping.</span>

<span style="font-size: 14px;">The transpose alone produces a tensor whose memory layout no longer matches a simple reshape, since transposing swaps strides without moving data; a reshape applied directly to a transposed tensor can fail outright or silently reinterpret memory incorrectly, which is why implementations explicitly make the tensor contiguous before the final reshape.</span>

<span style="font-size: 14px;">The concatenated $(batch, seq, d_{model})$ tensor is then passed through $W_o$, a final learned mixing matrix. This step matters beyond just restoring the original shape: without $W_o$, the model's only way to combine information across heads would be whatever downstream layer receives the output, and that layer would receive the heads' outputs in a fixed, un-mixed arrangement. $W_o$ lets the model learn how to weight and combine each head's contribution, rather than treating the heads as independent parallel channels forever.</span>

---

## <span style="font-size: 16px;">Multi-head attention is the general case, not a variant</span>

<span style="font-size: 14px;">The reason single-head attention (num_heads=1) reduces exactly to plain scaled dot-product attention is not a coincidence of implementation, but a structural fact: with one head, $d_k = d_{model}$, so the reshape into heads and the later concatenation are both shape-preserving no-ops, and the entire computation collapses to a single projection, a single attention computation over the full feature dimension, and a single output projection.</span>

<span style="font-size: 14px;">Every later attention variant in this study plan, multi-query attention, grouped-query attention, and multi-head latent attention, is best understood as a modification to how many independent key and value projections exist per query head, not as a departure from this same head-splitting and per-head-attention structure.</span>

<span style="font-size: 14px;">Multi-query attention is the case where every query head shares one key and value head; grouped-query attention interpolates between the two extremes; this problem is the baseline where every query head owns an independent key and value head.</span>

---

## <span style="font-size: 16px;">Inference motivation</span>

<span style="font-size: 14px;">At inference time, every one of these $h$ heads requires its own key and value cache entries during autoregressive generation, since each head's keys and values come from a different projection of the same hidden state and are not interchangeable across heads.</span>

<span style="font-size: 14px;">This is the direct reason KV-cache memory scales with the number of heads, not just with sequence length or model dimension, and it is exactly the cost that multi-query and grouped-query attention are designed to reduce by sharing key and value projections across multiple query heads while keeping the query-side head count, and therefore the representational benefit described above, unchanged.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">The two projection steps, computing $Q, K, V$ and applying $W_o$, each cost $O(batch \cdot seq \cdot d_{model}^2)$, independent of how many heads the features are split into, since splitting is a reshape with no arithmetic cost. The per-head attention cost is $O(batch \cdot h \cdot seq^2 \cdot d_k)$, which simplifies to $O(batch \cdot seq^2 \cdot d_{model})$ because $h \cdot d_k = d_{model}$ by construction.</span>

<span style="font-size: 14px;">For long sequences where $seq \gg d_{model}$, the quadratic-in-sequence-length attention term dominates over the projection terms, exactly as in single-head attention; the number of heads changes how that cost is distributed across parallel per-head computations, not the total asymptotic cost.</span>

---

## <span style="font-size: 16px;">Memory behavior</span>

<span style="font-size: 14px;">The dominant additional memory cost relative to single-head attention is the score and weight tensors, now $(batch, h, seq, seq)$ instead of $(batch, seq, seq)$: splitting into more heads with a correspondingly smaller $d_k$ keeps the total parameter count in $Q, K, V, W_o$ unchanged, but multiplies the number of separate $seq \times seq$ attention matrices that must be held in memory simultaneously by $h$.</span>

---

## <span style="font-size: 16px;">Common failure modes</span>

* <span style="font-size: 14px;">Splitting heads with the wrong axis order, for example transposing before reshaping instead of after, which produces a valid shape but scrambles which features belong to which head.</span>

* <span style="font-size: 14px;">Forgetting .contiguous() before the final reshape after attention, which can raise a runtime error or, worse on some backends, silently produce an incorrectly laid out tensor.</span>

* <span style="font-size: 14px;">Using a different scaling factor than $\sqrt{d_k}$, such as $\sqrt{d_{model}}$, which is only correct when num_heads=1.</span>

* <span style="font-size: 14px;">Applying the output projection $W_o$ before concatenating heads, or applying it per head instead of once on the concatenated tensor, which changes the model's expressive power and does not match the standard formulation.</span>

* <span style="font-size: 14px;">Building the causal mask with the wrong shape so that it fails to broadcast across the head dimension, or broadcasts against the wrong pair of seq axes.</span>

---

## <span style="font-size: 16px;">Contract and notation</span>

<span style="font-size: 14px;">$X$ denotes the hidden states, $W_q$, $W_k$, $W_v$, and $W_o$ denote the four projection matrices, $h$ is the number of heads, $d_{model}$ is the model width, $d_k=d_{model}/h$ is one head width, and $M$ is the optional causal mask.</span>

<span style="font-size: 14px;">Hidden states and the final result have shape $(B,S,d_{model})$; split projections have shape $(B,h,S,d_k)$; scores have shape $(B,h,S,S)$.</span>

---

## <span style="font-size: 16px;">Worked example</span>

<span style="font-size: 14px;">Input: hidden_states = [[[1.0, 0.0], [0.0, 1.0]]], w_q = w_k = w_v = w_o = identity(2), num_heads = 1, causal = False</span>

<span style="font-size: 14px;">Output: tensor([[[0.6698, 0.3302], [0.3302, 0.6698]]])</span>

<span style="font-size: 14px;">Explanation: With num_heads = 1, this is exactly single-head scaled dot-product attention over the full 2-dimensional space.</span>

---

## <span style="font-size: 16px;">Paper and system context</span>

<span style="font-size: 14px;">The Transformer paper introduced multi-head attention as the core mechanism used by both its encoder and decoder. Separate learned subspaces allow attention patterns to specialize while one output projection mixes their results.</span>

<span style="font-size: 14px;">During inference, the head count does not change the leading arithmetic order when the model width is fixed, but it does determine the number of independent key and value heads that must be cached. That cache cost motivates MQA, GQA, and MLA later in this plan.</span>

---

## <span style="font-size: 16px;">Correctness invariants</span>

<span style="font-size: 14px;">The head reshape keeps each contiguous projected feature group together, so every head receives the intended query, key, and value coordinates.</span>

<span style="font-size: 14px;">The causal mask is applied to every batch item and head before normalization, which prevents future positions from contributing.</span>

<span style="font-size: 14px;">Restoring sequence-major layout before the output projection concatenates heads in their original feature order.</span>

<span style="font-size: 14px;">The divisibility guarantee makes the head reshape exact, and the one-head case provides a direct equivalence check against ordinary attention.</span>

---