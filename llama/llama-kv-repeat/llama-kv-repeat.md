# <span style="font-size: 20px;">KV Head Repeat</span>

<span style="font-size: 14px;">KV Head Repeat is the expansion operation at the heart of Grouped-Query Attention (GQA). When the number of key/value heads is smaller than the number of query heads, each KV head must be repeated to match the query head count so that standard attention can proceed. This operation takes a tensor of shape $(B, N_{kv}, S, D)$ and produces $(B, N_{kv} \times n_{rep}, S, D)$. It is used in LLaMA 2 and LLaMA 3 to enable GQA's memory savings while keeping the attention computation unchanged.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">In standard Multi-Head Attention, each query head has its own dedicated key head and value head. In Grouped-Query Attention, multiple query heads share a single key-value pair, so the model stores fewer KV heads than query heads. Before the dot-product attention step, the KV heads must be expanded (repeated) along the head dimension so that every query head has a corresponding key and value head to attend against.</span>

<span style="font-size: 14px;">KV Head Repeat performs exactly this expansion. Given a key or value tensor with $N_{kv}$ heads, it replicates each head $n_{rep}$ times to produce a tensor with $N_{kv} \times n_{rep} = N_q$ heads. The resulting tensor has the same shape as if each query head had its own unique KV projection, and attention proceeds with the standard formula.</span>

<span style="font-size: 14px;">The operation is purely structural -- it applies no learned weights, nonlinearities, or normalization. When $n_{rep} = 1$ (meaning $N_{kv} = N_q$), the tensor is returned unchanged because no expansion is needed. This corresponds to standard MHA where every query head already has its own KV head.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $N_q$ be the number of query heads and $N_{kv}$ be the number of key/value heads. The repetition factor is:</span>

$$
n_{rep} = \frac{N_q}{N_{kv}}
$$

<span style="font-size: 14px;">This must be a positive integer. The input KV tensor has shape:</span>

$$
x \in \mathbb{R}^{B \times N_{kv} \times S \times D}
$$

<span style="font-size: 14px;">After the repeat operation, the output has shape:</span>

$$
\text{output} \in \mathbb{R}^{B \times (N_{kv} \times n_{rep}) \times S \times D} = \mathbb{R}^{B \times N_q \times S \times D}
$$

<span style="font-size: 14px;">The expansion maps each KV head index $g \in \{0, 1, \dots, N_{kv}-1\}$ to $n_{rep}$ consecutive output head indices. Output head $h$ corresponds to input KV head $g = \lfloor h / n_{rep} \rfloor$:</span>

$$
\text{output}[:, h, :, :] = x[:, \lfloor h / n_{rep} \rfloor, :, :]
$$

<span style="font-size: 14px;">This means query heads $0$ through $n_{rep}-1$ all attend against KV head $0$, query heads $n_{rep}$ through $2 \cdot n_{rep}-1$ attend against KV head $1$, and so on. The edge case is straightforward: if $n_{rep} = 1$, output $= x$.</span>

---

## <span style="font-size: 16px;">Why Repeat Instead of Separate Projections</span>

<span style="font-size: 14px;">A natural question is: why not give every query head its own separate key and value projection? Sharing KV heads provides substantial savings in both parameters and memory, while the quality loss is minimal.</span>

### <span style="font-size: 14px;">Parameter and Cache Savings</span>

<span style="font-size: 14px;">In MHA, the key and value projections each have shape $(d_{model}, N_q \times D)$. In GQA, these shrink to $(d_{model}, N_{kv} \times D)$, reducing KV projection parameters by a factor of $n_{rep}$. More importantly, during autoregressive inference the KV cache stores only $N_{kv}$ heads per layer instead of $N_q$, cutting cache memory by the same factor. The repeat operation reconstructs the full head count on-the-fly from the compact cache.</span>

### <span style="font-size: 14px;">Why Quality Is Preserved</span>

<span style="font-size: 14px;">Ainslie et al. (2023) showed that GQA models achieve quality very close to MHA. The key insight is that different query heads within a group still compute different attention patterns even though they share keys and values. Each query head has its own unique $W_Q$ projection, so the queries differ. Different queries attending against the same keys produce different attention weights, which when applied to the same values still yield different outputs.</span>

---

## <span style="font-size: 16px;">The Expansion Operation</span>

<span style="font-size: 14px;">The repeat operation is implemented as three tensor manipulations: unsqueeze, expand, and reshape.</span>

### <span style="font-size: 14px;">Step 1: Unsqueeze</span>

<span style="font-size: 14px;">Insert a new dimension of size 1 after the head dimension (dimension 1). This prepares a slot for the repetition:</span>

$$
(B, N_{kv}, S, D) \xrightarrow{\text{unsqueeze(2)}} (B, N_{kv}, 1, S, D)
$$

<span style="font-size: 14px;">The unsqueeze is at position 2 (after the $N_{kv}$ dimension), not at position 1. The new dimension of size 1 will be expanded to $n_{rep}$.</span>

### <span style="font-size: 14px;">Step 2: Expand</span>

<span style="font-size: 14px;">Use broadcasting to logically replicate the tensor along the new dimension without allocating new memory:</span>

$$
(B, N_{kv}, 1, S, D) \xrightarrow{\text{expand}} (B, N_{kv}, n_{rep}, S, D)
$$

<span style="font-size: 14px;">The expand operation returns a view of the original data, not a copy. The repeated slices share the same underlying memory. Internally, the stride for the expanded dimension is set to 0, meaning stepping along that axis does not advance the data pointer. This makes the expand operation $O(1)$ in memory.</span>

### <span style="font-size: 14px;">Step 3: Reshape</span>

<span style="font-size: 14px;">Merge the $N_{kv}$ and $n_{rep}$ dimensions into a single head dimension:</span>

$$
(B, N_{kv}, n_{rep}, S, D) \xrightarrow{\text{reshape}} (B, N_{kv} \times n_{rep}, S, D)
$$

<span style="font-size: 14px;">The reshape collapses dimensions 1 and 2 into one. The ordering ensures that the first $n_{rep}$ heads in the output all come from KV head 0, the next $n_{rep}$ from KV head 1, and so on. This contiguous grouping aligns with how query heads are assigned to KV groups.</span>

### <span style="font-size: 14px;">How Broadcasting Works Here</span>

<span style="font-size: 14px;">When a dimension has size 1, PyTorch's expand (and NumPy's broadcast_to) can stretch it to any size without copying data. The subsequent reshape may trigger a copy if the expanded tensor is not contiguous. In practice, calling `.contiguous()` or using `.repeat()` instead of `.expand()` ensures a physical copy when needed, for example if the result will be modified in-place.</span>

---

## <span style="font-size: 16px;">GQA Context: The Attention Sharing Spectrum</span>

<span style="font-size: 14px;">Grouped-Query Attention sits on a spectrum between two extremes. Understanding this spectrum clarifies when $n_{rep}$ takes different values.</span>

### <span style="font-size: 14px;">Multi-Head Attention (MHA)</span>

<span style="font-size: 14px;">$N_{kv} = N_q$, so $n_{rep} = 1$. Every query head has a unique KV head. No repetition is needed. This is used in the original Transformer (Vaswani et al., 2017), GPT-2, GPT-3, and LLaMA 1.</span>

### <span style="font-size: 14px;">Multi-Query Attention (MQA)</span>

<span style="font-size: 14px;">$N_{kv} = 1$, so $n_{rep} = N_q$. A single KV head is shared across all query heads. Maximum memory savings but can degrade quality for larger models. Introduced by Shazeer (2019) and used in PaLM and Falcon.</span>

### <span style="font-size: 14px;">Grouped-Query Attention (GQA)</span>

<span style="font-size: 14px;">$1 < N_{kv} < N_q$, so $1 < n_{rep} < N_q$. The query heads are divided into $N_{kv}$ groups, each sharing one KV head. GQA provides most of MQA's memory savings while retaining most of MHA's quality.</span>

* <span style="font-size: 14px;">**MHA:** $n_{rep} = 1$, maximum quality, maximum KV cache size</span>
* <span style="font-size: 14px;">**GQA:** $1 < n_{rep} < N_q$, near-MHA quality, reduced KV cache</span>
* <span style="font-size: 14px;">**MQA:** $n_{rep} = N_q$, some quality loss, minimum KV cache size</span>

---

## <span style="font-size: 16px;">Paper Context</span>

### <span style="font-size: 14px;">Ainslie et al. (2023) -- GQA Paper</span>

<span style="font-size: 14px;">This paper formally introduced Grouped-Query Attention. The key contribution was showing that an existing MHA model can be "uptrained" into a GQA model by mean-pooling adjacent KV heads into groups and continuing training for a small fraction of the original compute. GQA-8 (8 KV groups) achieved quality comparable to MHA on summarization and translation benchmarks while matching MQA's inference speed. This work directly motivated the repeat operation: at inference time, the reduced KV heads must be expanded back to the full query head count.</span>

### <span style="font-size: 14px;">LLaMA 2 (Touvron et al., 2023)</span>

<span style="font-size: 14px;">LLaMA 2 was one of the first major open-weight model families to adopt GQA. The 7B model uses standard MHA ($N_q = 32$, $N_{kv} = 32$, $n_{rep} = 1$), while the 70B model uses GQA with $N_q = 64$ and $N_{kv} = 8$, giving $n_{rep} = 8$. GQA was applied only to the larger model where the KV cache becomes a critical bottleneck.</span>

### <span style="font-size: 14px;">LLaMA 3 (Meta, 2024)</span>

<span style="font-size: 14px;">LLaMA 3 adopted GQA across all model sizes. The 8B model uses $N_q = 32$ and $N_{kv} = 8$ ($n_{rep} = 4$). The 70B model uses $N_q = 64$ and $N_{kv} = 8$ ($n_{rep} = 8$). The repeat operation is invoked in every attention layer of every LLaMA 3 model. In Meta's reference implementation, the function is called `repeat_kv` and is invoked twice per attention layer -- once for keys and once for values.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider $B = 1$, $N_{kv} = 2$, $S = 3$, $D = 2$, and $n_{rep} = 4$. This means 2 KV heads and $2 \times 4 = 8$ query heads.</span>

### <span style="font-size: 14px;">Input Tensor (shape: 1, 2, 3, 2)</span>

<span style="font-size: 14px;">KV head 0 (3 positions, 2 features each):</span>

$$
\text{head}_0 = \begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 4.0 \\ 5.0 & 6.0 \end{bmatrix}
$$

<span style="font-size: 14px;">KV head 1 (3 positions, 2 features each):</span>

$$
\text{head}_1 = \begin{bmatrix} 7.0 & 8.0 \\ 9.0 & 10.0 \\ 11.0 & 12.0 \end{bmatrix}
$$

### <span style="font-size: 14px;">Step-by-Step Expansion</span>

<span style="font-size: 14px;">**Unsqueeze:** $(1, 2, 3, 2) \to (1, 2, 1, 3, 2)$. A size-1 axis is inserted at position 2.</span>

<span style="font-size: 14px;">**Expand:** $(1, 2, 1, 3, 2) \to (1, 2, 4, 3, 2)$. Each KV head is logically replicated 4 times.</span>

<span style="font-size: 14px;">**Reshape:** $(1, 2, 4, 3, 2) \to (1, 8, 3, 2)$. Dimensions 1 and 2 are flattened into a single head dimension of size 8.</span>

### <span style="font-size: 14px;">Output Tensor (shape: 1, 8, 3, 2)</span>

* <span style="font-size: 14px;">**Head 0** (from KV head 0): $[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]$</span>
* <span style="font-size: 14px;">**Head 1** (from KV head 0): $[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]$</span>
* <span style="font-size: 14px;">**Head 2** (from KV head 0): $[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]$</span>
* <span style="font-size: 14px;">**Head 3** (from KV head 0): $[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]$</span>
* <span style="font-size: 14px;">**Head 4** (from KV head 1): $[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]$</span>
* <span style="font-size: 14px;">**Head 5** (from KV head 1): $[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]$</span>
* <span style="font-size: 14px;">**Head 6** (from KV head 1): $[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]$</span>
* <span style="font-size: 14px;">**Head 7** (from KV head 1): $[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]$</span>

<span style="font-size: 14px;">Heads 0-3 are identical copies of KV head 0, and heads 4-7 are identical copies of KV head 1. Despite sharing K and V, each query head produces a different attention output because each has a unique $W_Q$ projection.</span>

---

## <span style="font-size: 16px;">KV Cache Savings</span>

<span style="font-size: 14px;">The practical motivation for GQA is KV cache memory reduction during inference. Here is a concrete comparison using LLaMA 3 8B dimensions: $d_{model} = 4096$, $N_q = 32$, $D = 128$, 32 layers, sequence length $S = 8192$, FP16 precision (2 bytes per value).</span>

### <span style="font-size: 14px;">MHA: $N_{kv} = 32$</span>

<span style="font-size: 14px;">Per token across all layers: $2 \times 32 \times 128 \times 32 = 262{,}144$ values $= 0.5$ MB. For $S = 8192$: $\approx 4$ GB per batch element.</span>

### <span style="font-size: 14px;">GQA: $N_{kv} = 8$ (LLaMA 3 8B)</span>

<span style="font-size: 14px;">Per token across all layers: $2 \times 8 \times 128 \times 32 = 65{,}536$ values $= 0.125$ MB. For $S = 8192$: $\approx 1$ GB per batch element. A $4\times$ reduction over MHA.</span>

### <span style="font-size: 14px;">MQA: $N_{kv} = 1$</span>

<span style="font-size: 14px;">Per token across all layers: $2 \times 1 \times 128 \times 32 = 8{,}192$ values $= 0.016$ MB. For $S = 8192$: $\approx 128$ MB per batch element. A $32\times$ reduction over MHA.</span>

<span style="font-size: 14px;">GQA with $N_{kv} = 8$ captures $75\%$ of the cache reduction MQA provides (4 GB down to 1 GB vs. MQA's 128 MB) while preserving nearly all of MHA's quality. The repeat operation bridges the gap at runtime: the model stores the compact representation and expands it on-the-fly when computing attention.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Repeating Along the Wrong Dimension</span>

<span style="font-size: 14px;">The unsqueeze must be placed at dimension 2 (between head and sequence dimensions). Unsqueezing at dimension 1 would produce shape $(B, n_{rep}, N_{kv}, S, D)$, which interleaves incorrectly when reshaped. Unsqueezing at dimension 3 would replicate along the sequence axis, producing $n_{rep} \times S$ sequence positions instead of $n_{rep} \times N_{kv}$ heads.</span>

### <span style="font-size: 14px;">Forgetting the $n_{rep} = 1$ Edge Case</span>

<span style="font-size: 14px;">When $n_{rep} = 1$, the function must return the input unchanged. Skipping this check wastes computation and, more subtly, the reshape step may produce a non-contiguous tensor even when the input was contiguous. The $n_{rep} = 1$ case is not hypothetical -- LLaMA 2 7B uses this configuration.</span>

### <span style="font-size: 14px;">Creating Copies vs. Views</span>

<span style="font-size: 14px;">The expand operation returns a view (shared memory), while repeat returns a copy (new memory). For the forward pass, expand followed by reshape is correct and memory-efficient. However, if any downstream operation modifies the expanded tensor in-place, it will corrupt the original data because expanded slices share memory. Standard attention only reads the expanded K/V, so this is safe. Custom implementations that modify K or V in-place must use `.contiguous()` or `.repeat()` to force a physical copy.</span>

### <span style="font-size: 14px;">Expanding Q Instead of K/V</span>

<span style="font-size: 14px;">GQA reduces the number of K/V heads, not query heads. The repeat operation must be applied to K and V, never to Q. Queries always have the full head count $N_q$. Applying repeat to Q would produce $N_q \times n_{rep}$ query heads against $N_{kv}$ KV heads, and the attention dot product would fail with a shape mismatch.</span>

### <span style="font-size: 14px;">Dimension Mismatch After Repeat</span>

<span style="font-size: 14px;">After repeat, the expanded K/V tensor must have exactly $N_q$ heads. A common bug is computing $n_{rep}$ with truncating integer division: if $N_q = 32$ and $N_{kv} = 6$, then $n_{rep} = 32 / 6 = 5$ (truncated), producing $6 \times 5 = 30$ heads instead of 32. GQA requires that $N_q$ is evenly divisible by $N_{kv}$. A robust implementation should assert $N_q \mod N_{kv} = 0$.</span>

### <span style="font-size: 14px;">Wrong Reshape Order</span>

<span style="font-size: 14px;">The 5D tensor after expand has shape $(B, N_{kv}, n_{rep}, S, D)$. If you accidentally transpose dimensions 1 and 2 before reshaping, the result has shape $(B, n_{rep}, N_{kv}, S, D)$, which when flattened interleaves KV heads instead of grouping them. Head 0 would come from KV head 0, head 1 from KV head 1, head 2 from KV head 0 again. This breaks the query-to-KV-group assignment and produces incorrect attention outputs.</span>

---