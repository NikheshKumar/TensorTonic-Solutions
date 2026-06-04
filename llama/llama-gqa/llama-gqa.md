# <span style="font-size: 20px;">Grouped Query Attention</span>

<span style="font-size: 14px;">Grouped Query Attention (GQA) is a memory-efficient variant of multi-head attention where multiple query heads share the same key and value head. Introduced by Ainslie et al. (2023) and adopted as a core component of LLaMA 2 and LLaMA 3, GQA dramatically reduces the KV cache memory footprint during inference while retaining nearly all of the quality of standard multi-head attention.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In standard **Multi-Head Attention (MHA)**, every query head has its own dedicated key head and value head. If the model uses $n_{\text{heads}} = 32$ query heads, it also maintains 32 key heads and 32 value heads. During autoregressive inference, the key and value tensors for every past token must be stored in the **KV cache**, which grows linearly with sequence length and the number of KV heads.</span>

<span style="font-size: 14px;">GQA reduces the number of KV heads to $n_{\text{kv\_heads}} < n_{\text{heads}}$. Multiple query heads are grouped together, and each group shares one key head and one value head. The number of query heads per group is the **repeat factor**:</span>

$$
n_{\text{rep}} = \frac{n_{\text{heads}}}{n_{\text{kv\_heads}}}
$$

<span style="font-size: 14px;">LLaMA 3 8B uses $n_{\text{heads}} = 32$ and $n_{\text{kv\_heads}} = 8$, giving $n_{\text{rep}} = 4$. Each KV head is shared by 4 query heads. The attention computation is identical to standard scaled dot-product attention; the only difference is that K and V are repeated along the head dimension before the dot product so every query head has a matching K and V.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 -- Linear projections.** The input $x \in \mathbb{R}^{B \times S \times d}$ is projected into queries, keys, and values:</span>

$$
Q = x \, W_Q^T, \quad K = x \, W_K^T, \quad V = x \, W_V^T
$$

<span style="font-size: 14px;">where $W_Q \in \mathbb{R}^{(n_{\text{heads}} \cdot d_{\text{head}}) \times d}$, and $W_K, W_V \in \mathbb{R}^{(n_{\text{kv\_heads}} \cdot d_{\text{head}}) \times d}$. Note that Q has a larger projection dimension than K and V because there are more query heads than KV heads.</span>

<span style="font-size: 14px;">**Step 2 -- Reshape into multi-head form.** Q is reshaped to $(B, n_{\text{heads}}, S, d_{\text{head}})$ and K, V are reshaped to $(B, n_{\text{kv\_heads}}, S, d_{\text{head}})$:</span>

$$
Q \rightarrow (B, n_{\text{heads}}, S, d_{\text{head}}), \quad K \rightarrow (B, n_{\text{kv\_heads}}, S, d_{\text{head}}), \quad V \rightarrow (B, n_{\text{kv\_heads}}, S, d_{\text{head}})
$$

<span style="font-size: 14px;">**Step 3 -- Repeat KV heads.** Each KV head is repeated $n_{\text{rep}}$ times along the head dimension so K and V match the query head count:</span>

$$
K_{\text{exp}} = \text{repeat}(K, n_{\text{rep}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times S \times d_{\text{head}}}
$$

$$
V_{\text{exp}} = \text{repeat}(V, n_{\text{rep}}) \in \mathbb{R}^{B \times n_{\text{heads}} \times S \times d_{\text{head}}}
$$

<span style="font-size: 14px;">**Step 4 -- Scaled dot-product attention.** With all tensors now at $(B, n_{\text{heads}}, S, d_{\text{head}})$, standard attention proceeds:</span>

$$
\text{scores} = \frac{Q \cdot K_{\text{exp}}^T}{\sqrt{d_{\text{head}}}}, \quad \text{weights} = \text{softmax}(\text{scores}), \quad \text{context} = \text{weights} \cdot V_{\text{exp}}
$$

<span style="font-size: 14px;">**Step 5 -- Concatenate and output projection.** The heads are concatenated and projected through $W_O$:</span>

$$
\text{output} = \text{concat}(\text{head}_1, \ldots, \text{head}_{n_{\text{heads}}}) \cdot W_O^T
$$

<span style="font-size: 14px;">where $W_O \in \mathbb{R}^{d \times (n_{\text{heads}} \cdot d_{\text{head}})}$. The concat reshapes from $(B, n_{\text{heads}}, S, d_{\text{head}})$ to $(B, S, n_{\text{heads}} \cdot d_{\text{head}})$, and $W_O$ projects back to $(B, S, d)$.</span>

---

## <span style="font-size: 16px;">The GQA Spectrum</span>

<span style="font-size: 14px;">GQA is a generalization that places Multi-Head Attention (MHA) and Multi-Query Attention (MQA) at opposite ends of a spectrum, with the number of KV heads as the tuning knob:</span>

* <span style="font-size: 14px;">**MHA ($n_{\text{kv\_heads}} = n_{\text{heads}}$):** Every query head has its own KV head. No sharing. Maximum capacity but largest KV cache. Used by the original Transformer, GPT-2, BERT.</span>
* <span style="font-size: 14px;">**MQA ($n_{\text{kv\_heads}} = 1$):** All query heads share one key head and one value head. Minimal KV cache but can degrade quality. Introduced by Shazeer (2019), used in PaLM and Falcon.</span>
* <span style="font-size: 14px;">**GQA ($1 < n_{\text{kv\_heads}} < n_{\text{heads}}$):** Query heads are divided into groups, each sharing one KV head. More capacity than MQA, far less memory than MHA. Introduced by Ainslie et al. (2023).</span>

<span style="font-size: 14px;">The key finding is that with an intermediate number of KV groups, quality stays close to MHA while memory savings approach MQA. This is why LLaMA 2 70B was the first major model to adopt GQA, and LLaMA 3 extended it to all sizes.</span>

<span style="font-size: 14px;">Concrete configurations in the LLaMA family:</span>

* <span style="font-size: 14px;">**LLaMA 2 7B:** $n_{\text{heads}} = 32$, $n_{\text{kv\_heads}} = 32$ (standard MHA, no grouping)</span>
* <span style="font-size: 14px;">**LLaMA 2 70B:** $n_{\text{heads}} = 64$, $n_{\text{kv\_heads}} = 8$, $n_{\text{rep}} = 8$</span>
* <span style="font-size: 14px;">**LLaMA 3 8B:** $n_{\text{heads}} = 32$, $n_{\text{kv\_heads}} = 8$, $n_{\text{rep}} = 4$</span>
* <span style="font-size: 14px;">**LLaMA 3 70B:** $n_{\text{heads}} = 64$, $n_{\text{kv\_heads}} = 8$, $n_{\text{rep}} = 8$</span>

---

## <span style="font-size: 16px;">Why Share KV Heads</span>

<span style="font-size: 14px;">The motivation for GQA is almost entirely about **inference efficiency**. During autoregressive generation, the key and value tensors for all previous tokens must be cached so they can be reused. The cache size per layer is:</span>

$$
\text{KV cache per layer} = 2 \times n_{\text{kv\_heads}} \times S \times d_{\text{head}} \times \text{bytes}
$$

<span style="font-size: 14px;">The factor of 2 accounts for both K and V. With MHA, $n_{\text{kv\_heads}} = n_{\text{heads}}$, so the cache scales with the full head count. With GQA, fewer KV heads means a proportionally smaller cache.</span>

<span style="font-size: 14px;">Why does this work without significant quality loss? Key and value representations across heads are often highly correlated. Different query heads extract different aspects from the context, but the underlying KV structure does not need to be as diverse. Sharing KV heads forces groups of query heads to attend over the same key-value space, which acts as benign regularization.</span>

<span style="font-size: 14px;">Model parameters also decrease with GQA. The K and V projection matrices shrink from $\mathbb{R}^{(n_{\text{heads}} \cdot d_{\text{head}}) \times d}$ to $\mathbb{R}^{(n_{\text{kv\_heads}} \cdot d_{\text{head}}) \times d}$. For LLaMA 3 8B ($d = 4096$, $n_{\text{heads}} = 32$, $n_{\text{kv\_heads}} = 8$, $d_{\text{head}} = 128$): each of $W_K$ and $W_V$ shrinks from $4096 \times 4096$ to $1024 \times 4096$, saving about 25M parameters per matrix per layer.</span>

---

## <span style="font-size: 16px;">The KV Repeat Step</span>

<span style="font-size: 14px;">The repeat step is the mechanical heart of GQA. After reshaping, K is $(B, n_{\text{kv\_heads}}, S, d_{\text{head}})$ but Q is $(B, n_{\text{heads}}, S, d_{\text{head}})$. The head dimensions do not match, so K and V must be expanded before the batched matrix multiply.</span>

<span style="font-size: 14px;">Each KV head is repeated $n_{\text{rep}}$ times along the head axis. If $n_{\text{kv\_heads}} = 8$ and $n_{\text{rep}} = 4$, KV head 0 is copied to positions 0-3, KV head 1 to positions 4-7, and so on, producing shape $(B, 32, S, d_{\text{head}})$.</span>

<span style="font-size: 14px;">In PyTorch, the standard implementation uses `unsqueeze` followed by `expand` and `reshape`:</span>

* <span style="font-size: 14px;">**Unsqueeze:** Insert a new dimension after the head axis: $(B, n_{\text{kv\_heads}}, 1, S, d_{\text{head}})$.</span>
* <span style="font-size: 14px;">**Expand:** Broadcast along the new dimension to size $n_{\text{rep}}$: $(B, n_{\text{kv\_heads}}, n_{\text{rep}}, S, d_{\text{head}})$. This is zero-copy; it creates a view with stride 0 along the repeated dimension.</span>
* <span style="font-size: 14px;">**Reshape:** Collapse the KV head and repeat dimensions: $(B, n_{\text{kv\_heads}} \times n_{\text{rep}}, S, d_{\text{head}}) = (B, n_{\text{heads}}, S, d_{\text{head}})$.</span>

<span style="font-size: 14px;">An alternative is `torch.repeat_interleave(K, n_rep, dim=1)`, which produces the same result but always allocates new memory. The `expand` + `reshape` path is preferred because expand is zero-copy. The repeat must happen **before** the attention score computation; skipping it causes a shape mismatch in $Q \cdot K^T$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The GQA paper by Ainslie et al. (2023), titled "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", makes several key contributions:</span>

* <span style="font-size: 14px;">**Uptrained from MHA:** A pretrained MHA model can be converted to GQA by mean-pooling adjacent KV heads into groups, then fine-tuning for about 5% of the original training budget. This is how LLaMA 2 70B was produced.</span>
* <span style="font-size: 14px;">**Quality-memory tradeoff:** Benchmarks show GQA with $n_{\text{kv\_heads}} = 8$ matches MHA quality within noise while achieving throughput close to MQA. The sweet spot is 4-8 KV groups regardless of model size.</span>
* <span style="font-size: 14px;">**Inference speedup:** Fewer KV heads reduce memory bandwidth required to load cached keys and values at each step. Since autoregressive generation is memory-bandwidth-bound, this directly translates to higher tokens-per-second.</span>

<span style="font-size: 14px;">LLaMA 2 (Touvron et al., 2023) adopted GQA for the 70B model only, citing inference efficiency as critical for serving large models. LLaMA 3 (Dubey et al., 2024) extended GQA to all sizes including 8B, reflecting the trend toward longer context windows (up to 128K tokens) where even small models benefit from reduced KV cache.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let $B = 1$, $S = 3$, $n_{\text{heads}} = 4$, $n_{\text{kv\_heads}} = 2$, $d_{\text{head}} = 2$. This gives $d = n_{\text{heads}} \times d_{\text{head}} = 8$ and $n_{\text{rep}} = 4 / 2 = 2$.</span>

<span style="font-size: 14px;">**Step 1 -- Q projection.** After $Q = x \, W_Q^T$ and reshaping to $(1, 4, 3, 2)$, we get 4 query heads. Showing heads 0 and 2 (one from each group):</span>

$$
Q_0 = \begin{pmatrix} 0.5 & 0.3 \\ 0.1 & 0.7 \\ 0.4 & 0.2 \end{pmatrix}, \quad Q_2 = \begin{pmatrix} 0.4 & 0.6 \\ 0.3 & 0.2 \\ 0.7 & 0.1 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 2 -- K, V projection.** After projection and reshaping to $(1, 2, 3, 2)$, we get 2 KV heads:</span>

$$
K_0 = \begin{pmatrix} 0.3 & 0.5 \\ 0.7 & 0.1 \\ 0.2 & 0.4 \end{pmatrix}, \quad K_1 = \begin{pmatrix} 0.6 & 0.2 \\ 0.1 & 0.8 \\ 0.4 & 0.3 \end{pmatrix}
$$

$$
V_0 = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \\ 0.5 & 0.5 \end{pmatrix}, \quad V_1 = \begin{pmatrix} 0.0 & 1.0 \\ 1.0 & 0.0 \\ 0.5 & 0.5 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 3 -- Repeat KV heads.** With $n_{\text{rep}} = 2$, $K_0$ is assigned to query heads 0 and 1; $K_1$ is assigned to query heads 2 and 3:</span>

$$
K_{\text{exp}} = [K_0, K_0, K_1, K_1], \quad V_{\text{exp}} = [V_0, V_0, V_1, V_1]
$$

<span style="font-size: 14px;">**Step 4 -- Attention for head 0.** Query head 0 uses $K_0$. Scores scaled by $1/\sqrt{2}$:</span>

$$
Q_0 \cdot K_0^T = \begin{pmatrix} 0.30 & 0.38 & 0.22 \\ 0.38 & 0.14 & 0.30 \\ 0.22 & 0.30 & 0.16 \end{pmatrix}, \quad \text{scores}_0 = \frac{1}{\sqrt{2}} \begin{pmatrix} 0.30 & 0.38 & 0.22 \\ 0.38 & 0.14 & 0.30 \\ 0.22 & 0.30 & 0.16 \end{pmatrix} \approx \begin{pmatrix} 0.212 & 0.269 & 0.156 \\ 0.269 & 0.099 & 0.212 \\ 0.156 & 0.212 & 0.113 \end{pmatrix}
$$

<span style="font-size: 14px;">Applying row-wise softmax:</span>

$$
\text{weights}_0 \approx \begin{pmatrix} 0.340 & 0.360 & 0.300 \\ 0.365 & 0.308 & 0.327 \\ 0.326 & 0.345 & 0.329 \end{pmatrix}
$$

<span style="font-size: 14px;">Context for head 0 using $V_0$:</span>

$$
\text{context}_0 = \text{weights}_0 \cdot V_0 \approx \begin{pmatrix} 0.490 & 0.510 \\ 0.529 & 0.471 \\ 0.491 & 0.509 \end{pmatrix}
$$

<span style="font-size: 14px;">Head 1 also uses $K_0, V_0$ (same group) but produces different context because $Q_1 \neq Q_0$. Heads 2 and 3 use $K_1, V_1$. This is core GQA: different queries attend over the same key-value space within each group.</span>

<span style="font-size: 14px;">**Step 5 -- Concatenate and project.** All 4 context vectors of shape $(3, 2)$ are concatenated to $(3, 8)$, then multiplied by $W_O^T$ to produce output $(1, 3, 8)$.</span>

---

## <span style="font-size: 16px;">KV Cache Memory Analysis</span>

<span style="font-size: 14px;">Consider LLaMA 3 8B with $d = 4096$, $n_{\text{heads}} = 32$, $d_{\text{head}} = 128$, $n_{\text{layers}} = 32$, sequence length $S = 8192$, and FP16 storage (2 bytes per element). Compare three attention configurations:</span>

<span style="font-size: 14px;">**MHA** ($n_{\text{kv\_heads}} = 32$):</span>

$$
\text{Cache} = 2 \times 32 \times 32 \times 8192 \times 128 \times 2 \approx 32 \text{ GB}
$$

<span style="font-size: 14px;">**GQA** ($n_{\text{kv\_heads}} = 8$, the actual LLaMA 3 8B config):</span>

$$
\text{Cache} = 2 \times 8 \times 32 \times 8192 \times 128 \times 2 \approx 8 \text{ GB}
$$

<span style="font-size: 14px;">**MQA** ($n_{\text{kv\_heads}} = 1$):</span>

$$
\text{Cache} = 2 \times 1 \times 32 \times 8192 \times 128 \times 2 \approx 1 \text{ GB}
$$

<span style="font-size: 14px;">GQA achieves a 4x reduction over MHA (matching $n_{\text{heads}} / n_{\text{kv\_heads}} = 32/8 = 4$). For the 70B model with $S = 128{,}000$, MHA would require hundreds of gigabytes of KV cache, making GQA a necessity for practical deployment.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong $n_{\text{rep}}$ calculation:** The repeat factor is $n_{\text{heads}} / n_{\text{kv\_heads}}$, not $n_{\text{kv\_heads}} / n_{\text{heads}}$. Getting this backwards produces $n_{\text{rep}} < 1$ (a fraction), which is meaningless. Always verify: $n_{\text{rep}} \times n_{\text{kv\_heads}} = n_{\text{heads}}$.</span>
* <span style="font-size: 14px;">**Repeating Q instead of K/V:** A common mistake is expanding the query tensor instead of the key and value tensors. Q already has $n_{\text{heads}}$ heads and needs no expansion. It is K and V that have fewer heads and must be repeated to match Q.</span>
* <span style="font-size: 14px;">**Dimension mismatch after repeat:** After repeating, K and V must have shape $(B, n_{\text{heads}}, S, d_{\text{head}})$, exactly matching Q. If the repeat is along the wrong axis (e.g., the sequence dimension), shapes may appear correct but attention will silently produce garbage.</span>
* <span style="font-size: 14px;">**Forgetting the output projection:** After concatenating heads, the result must be projected through $W_O$. Omitting this means the output lacks the learned linear mixing across heads that combines different attention patterns.</span>
* <span style="font-size: 14px;">**Causal mask shape with GQA:** The mask has shape $(S, S)$ or $(1, 1, S, S)$, independent of head count. A common error is sizing the mask head dimension to $n_{\text{kv\_heads}}$ instead of $n_{\text{heads}}$. The mask broadcasts across heads, so it must match the post-repeat count.</span>
* <span style="font-size: 14px;">**Assuming divisibility is optional:** $n_{\text{heads}}$ must be evenly divisible by $n_{\text{kv\_heads}}$. If $n_{\text{heads}} \% n_{\text{kv\_heads}} \neq 0$, the groups cannot be formed evenly and the repeat operation will produce incorrect shapes. Always validate this constraint.</span>
* <span style="font-size: 14px;">**Confusing $d_{\text{head}}$ with $d$:** K and V projection size is $n_{\text{kv\_heads}} \times d_{\text{head}}$, not $n_{\text{kv\_heads}} \times d$. Since $d_{\text{head}} = d / n_{\text{heads}}$, using $d$ instead of $d_{\text{head}}$ when reshaping causes shape errors or silently corrupts the head decomposition.</span>

---