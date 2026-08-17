# <span style="font-size: 20px;">Scaled Dot-Product Attention</span>

<span style="font-size: 14px;">Scaled dot-product attention is the fundamental operation inside every Transformer. It computes a weighted sum of **value** vectors, where the weights are determined by how well each **query** vector matches each **key** vector. The scaling factor $1/\sqrt{d_k}$ prevents dot products from growing so large that softmax saturates into a one-hot distribution.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">At its core, attention is a **soft dictionary lookup**. You have a query and a set of key-value pairs. Instead of finding the single best-matching key (hard lookup), attention computes a compatibility score between the query and every key, normalizes those scores via softmax, and returns a weighted combination of the corresponding values.</span>

<span style="font-size: 14px;">Given a matrix $Q$ of queries, a matrix $K$ of keys, and a matrix $V$ of values, the output is a matrix where each row is a weighted blend of the value rows. The weights are determined by the dot products between queries and keys.</span>

<span style="font-size: 14px;">In the Transformer, $Q$, $K$, and $V$ are derived from the same input sequence (self-attention) or different sequences (cross-attention). This problem focuses on the raw mechanism: given pre-computed $Q$, $K$, $V$, compute the attention output. No masking, no multi-head splitting.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The complete attention function in one line:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

<span style="font-size: 14px;">where $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{m \times d_k}$, $V \in \mathbb{R}^{m \times d_v}$, $n$ is the number of queries, $m$ is the number of keys (and values), $d_k$ is the key/query dimension, and $d_v$ is the value dimension.</span>

<span style="font-size: 14px;">Breaking this into sub-steps:</span>

<span style="font-size: 14px;">**Step 1 -- Raw scores:**</span>

$$
S = QK^T \in \mathbb{R}^{n \times m}
$$

<span style="font-size: 14px;">Each entry $S_{ij} = q_i^T k_j = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}$ is the dot product between query $i$ and key $j$.</span>

<span style="font-size: 14px;">**Step 2 -- Scaling:**</span>

$$
S' = \frac{S}{\sqrt{d_k}}
$$

<span style="font-size: 14px;">Every score is divided by $\sqrt{d_k}$. This is an element-wise division by a scalar.</span>

<span style="font-size: 14px;">**Step 3 -- Softmax (row-wise):**</span>

$$
W_{ij} = \frac{e^{S'_{ij}}}{\sum_{l=1}^{m} e^{S'_{il}}}
$$

<span style="font-size: 14px;">Softmax is applied independently to each row. After this step, each row of $W$ sums to 1 and contains non-negative values. $W \in \mathbb{R}^{n \times m}$ is the attention weight matrix.</span>

<span style="font-size: 14px;">**Step 4 -- Weighted value sum:**</span>

$$
O = WV \in \mathbb{R}^{n \times d_v}
$$

<span style="font-size: 14px;">Each output row $o_i = \sum_{j=1}^{m} W_{ij} v_j$ is a weighted combination of value vectors.</span>

---

## Why Scale by $\sqrt{d_k}$?

<span style="font-size: 14px;">The scaling factor is not arbitrary. It comes from a variance analysis of dot products.</span>

<span style="font-size: 14px;">Assume the components of $q$ and $k$ are independent random variables with mean 0 and variance 1. The dot product $q^T k = \sum_{l=1}^{d_k} q_l k_l$ is a sum of $d_k$ independent terms. Each term $q_l k_l$ has mean 0 and variance $\text{Var}(q_l) \cdot \text{Var}(k_l) = 1$. Therefore:</span>

$$
\text{Var}(q^T k) = d_k
$$

<span style="font-size: 14px;">The standard deviation of the dot product is $\sqrt{d_k}$. With $d_k = 64$, typical dot products have standard deviation 8, so values in $[-16, 16]$ are common.</span>

<span style="font-size: 14px;">**Why this causes problems.** Softmax with large inputs pushes nearly all probability onto the maximum entry. Softmax of $[10, 0, 0]$ gives approximately $[0.9999, 0.00005, 0.00005]$ -- essentially one-hot. When softmax saturates:</span>

* <span style="font-size: 14px;">**Gradient vanishing:** The gradient of softmax approaches zero in the saturated regime. Training stalls because the model cannot adjust which keys a query attends to.</span>
* <span style="font-size: 14px;">**Loss of blending:** Attention loses its ability to softly combine multiple values. The output collapses to approximately a single value vector, defeating the purpose of the mechanism.</span>

<span style="font-size: 14px;">Dividing by $\sqrt{d_k}$ rescales the dot products to have variance 1 regardless of the dimension. After scaling, the inputs to softmax remain in a moderate range where the function produces meaningful distributions and gradients flow normally.</span>

<span style="font-size: 14px;">Vaswani et al. (2017) note this explicitly in Section 3.2.1 of "Attention Is All You Need": "We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$."</span>

---

## <span style="font-size: 16px;">Step by Step</span>

<span style="font-size: 14px;">The complete procedure to compute scaled dot-product attention from raw $Q$, $K$, $V$ matrices:</span>

<span style="font-size: 14px;">1. **Compute raw scores:** Multiply $Q$ by $K^T$. If $Q$ has shape $(n, d_k)$ and $K$ has shape $(m, d_k)$, the result $S = QK^T$ has shape $(n, m)$. Each entry $S_{ij}$ is the dot product of the $i$-th query with the $j$-th key.</span>

<span style="font-size: 14px;">2. **Scale:** Divide every entry of $S$ by $\sqrt{d_k}$, where $d_k$ is the last dimension of $Q$ (and $K$).</span>

<span style="font-size: 14px;">3. **Apply softmax row-wise:** For each row $i$ of $S'$, compute softmax across the $m$ columns. This converts each row from raw scores into a probability distribution that sums to 1.</span>

<span style="font-size: 14px;">4. **Multiply by values:** Compute $O = WV$. If $V$ has shape $(m, d_v)$, then $O$ has shape $(n, d_v)$. Each output row is a weighted combination of value vectors.</span>

<span style="font-size: 14px;">The output $O$ has the same number of rows as $Q$ and the same number of columns as $V$.</span>

---

## <span style="font-size: 16px;">The Softmax Function</span>

<span style="font-size: 14px;">Softmax converts a vector of arbitrary real numbers into a probability distribution. Given a vector $z = [z_1, z_2, \ldots, z_m]$:</span>

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{m} e^{z_j}}
$$

<span style="font-size: 14px;">**Properties:**</span>

* <span style="font-size: 14px;">Every output is in $(0, 1)$ -- strictly positive, never exactly zero.</span>
* <span style="font-size: 14px;">The outputs sum to exactly 1.</span>
* <span style="font-size: 14px;">Monotonic: if $z_i > z_j$, then $\text{softmax}(z)_i > \text{softmax}(z)_j$.</span>
* <span style="font-size: 14px;">Translation invariant: $\text{softmax}(z + c) = \text{softmax}(z)$ for any constant $c$.</span>

<span style="font-size: 14px;">**Row-wise application in attention.** In the score matrix $S' \in \mathbb{R}^{n \times m}$, softmax is applied independently to each row. Row $i$ represents how query $i$ scores against all $m$ keys. After softmax, row $i$ becomes a distribution over keys: which keys does this query attend to, and how much?</span>

<span style="font-size: 14px;">**Numerical stability trick.** Computing $e^{z_i}$ directly risks overflow for large $z_i$ and underflow for very negative $z_i$. The fix exploits translation invariance:</span>

$$
\text{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{m} e^{z_j - \max(z)}}
$$

<span style="font-size: 14px;">Subtracting $\max(z)$ from every element makes the largest exponent $e^0 = 1$. All others are $\leq 1$. This prevents overflow while producing identical results. Every production softmax uses this trick.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Scaled dot-product attention was introduced in "Attention Is All You Need" (Vaswani et al., 2017), Section 3.2.1. The paper contrasts it with additive attention (Bahdanau et al., 2015), which uses a feed-forward network for compatibility scores.</span>

<span style="font-size: 14px;">The paper notes that dot-product attention is "much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code." The only modification over plain dot-product attention is the scaling factor, necessary for larger $d_k$.</span>

<span style="font-size: 14px;">**GPT-2's usage.** GPT-2 (Radford et al., 2019) uses this mechanism as the building block inside its multi-head attention layers. Each of GPT-2's 12 heads (117M model) computes scaled dot-product attention with $d_k = d_v = 64$. The multi-head wrapper projects the input into $h$ separate Q/K/V triplets, runs this operation on each, concatenates, and projects back. GPT-2 also adds a causal mask before softmax (setting future positions to $-\infty$) for autoregressive generation. This problem omits masking to isolate the pure attention computation.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace the full computation with $Q \in \mathbb{R}^{2 \times 3}$, $K \in \mathbb{R}^{3 \times 3}$, $V \in \mathbb{R}^{3 \times 3}$, so $n = 2$ queries, $m = 3$ keys, $d_k = 3$, $d_v = 3$.</span>

$$
Q = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \quad V = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1 -- Raw scores $S = QK^T$.** Since $K$ is the identity matrix, $QK^T = Q$:</span>

$$
S = \begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">Row 0 of $Q$ is $[1, 0, 1]$, so it scores 1 against keys 0 and 2, and 0 against key 1. Row 1 is $[0, 1, 0]$, scoring 1 only against key 1.</span>

<span style="font-size: 14px;">**Step 2 -- Scale by $\sqrt{d_k} = \sqrt{3} \approx 1.7321$:**</span>

$$
S' = \frac{S}{\sqrt{3}} = \begin{pmatrix} 0.5774 & 0 & 0.5774 \\ 0 & 0.5774 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 3 -- Row-wise softmax:**</span>

<span style="font-size: 14px;">For row 0: $z = [0.5774, 0, 0.5774]$. Exponentials: $e^{0.5774} \approx 1.7813$, $e^{0} = 1.0$, $e^{0.5774} \approx 1.7813$. Sum $= 4.5626$.</span>

* <span style="font-size: 14px;">$W_{00} = 1.7813 / 4.5626 \approx 0.3904$</span>
* <span style="font-size: 14px;">$W_{01} = 1.0 / 4.5626 \approx 0.2193$</span>
* <span style="font-size: 14px;">$W_{02} = 1.7813 / 4.5626 \approx 0.3904$</span>

<span style="font-size: 14px;">For row 1: $z = [0, 0.5774, 0]$. Exponentials: $1.0$, $1.7813$, $1.0$. Sum $= 3.7813$.</span>

* <span style="font-size: 14px;">$W_{10} = 1.0 / 3.7813 \approx 0.2644$</span>
* <span style="font-size: 14px;">$W_{11} = 1.7813 / 3.7813 \approx 0.4711$</span>
* <span style="font-size: 14px;">$W_{12} = 1.0 / 3.7813 \approx 0.2644$</span>

$$
W \approx \begin{pmatrix} 0.3904 & 0.2193 & 0.3904 \\ 0.2644 & 0.4711 & 0.2644 \end{pmatrix}
$$

<span style="font-size: 14px;">Row 0 attends roughly equally to keys 0 and 2 (both scored 1) and less to key 1 (scored 0). Row 1 attends most to key 1. Each row sums to 1.</span>

<span style="font-size: 14px;">**Step 4 -- Output $O = WV$:**</span>

<span style="font-size: 14px;">Row 0: $O_{00} = 0.3904(1) + 0.2193(4) + 0.3904(7) \approx 4.00$. Similarly $O_{01} \approx 5.00$, $O_{02} \approx 6.00$.</span>

<span style="font-size: 14px;">Row 1: $O_{10} = 0.2644(1) + 0.4711(4) + 0.2644(7) \approx 4.00$. Similarly $O_{11} \approx 5.00$, $O_{12} \approx 6.00$.</span>

$$
O \approx \begin{pmatrix} 4.00 & 5.00 & 6.00 \\ 4.00 & 5.00 & 6.00 \end{pmatrix}
$$

<span style="font-size: 14px;">**Interpreting the result.** Both rows are approximately $[4, 5, 6]$, close to the per-column mean of $V$. The identity-$K$ setup means no single value vector dominates. In real models with learned projections, the weights are far more skewed, producing outputs closer to specific value vectors.</span>

---

## <span style="font-size: 16px;">Attention as Soft Dictionary Lookup</span>

<span style="font-size: 14px;">A useful mental model is to think of attention as a **differentiable dictionary**. In a standard hash map, you provide a key and retrieve the associated value. Attention generalizes this:</span>

* <span style="font-size: 14px;">**Soft matching:** Instead of exact key equality, attention computes similarity between the query and all keys simultaneously. Every key matches to some degree.</span>
* <span style="font-size: 14px;">**Blended retrieval:** Instead of returning a single value, attention returns a weighted blend of all values. Keys that match the query well contribute more to the output.</span>

<span style="font-size: 14px;">This explains the naming: $Q$ contains **queries** (what am I looking for?), $K$ contains **keys** (what do I contain?), $V$ contains **values** (what do I want to communicate?). In self-attention, all three come from the same input, but different projections let each token express different things as query, key, and value.</span>

<span style="font-size: 14px;">The dot product $q_i^T k_j$ measures query-key compatibility. After scaling and softmax, it becomes a probability: how much should position $i$'s output draw from position $j$'s value? The multiplication $WV$ performs the retrieval -- blending values according to these probabilities.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">Scaled dot-product attention remains the core primitive in all modern Transformers, but several innovations address its computational and memory costs:</span>

* <span style="font-size: 14px;">**Flash Attention (Dao et al., 2022):** Computes exact attention without materializing the full $n \times m$ score matrix in GPU HBM. Tiles the computation into blocks in SRAM. Memory drops from $O(n^2)$ to $O(n)$ with 2-4x speedup.</span>
* <span style="font-size: 14px;">**Multi-Query Attention (Shazeer, 2019):** All heads share one set of keys and values. Reduces KV cache by $h$x, speeding up autoregressive inference.</span>
* <span style="font-size: 14px;">**Grouped-Query Attention (Ainslie et al., 2023):** Heads are divided into groups, each sharing keys and values. LLaMA 2 70B uses 8 KV heads with 32 query heads -- 4x KV cache reduction with minimal quality loss.</span>
* <span style="font-size: 14px;">**Ring Attention (Liu et al., 2023):** Distributes the sequence across devices in a ring, enabling million-token contexts without any single device holding the full sequence.</span>

<span style="font-size: 14px;">All compute the same mathematical function. The fundamental operation -- score, scale, softmax, blend -- is unchanged from the 2017 paper.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to scale.** Without dividing by $\sqrt{d_k}$, dot products grow with dimension and softmax saturates. The output collapses to hard attention (essentially argmax), gradients vanish, and training breaks. This is the most common implementation error.</span>
* <span style="font-size: 14px;">**Wrong matrix multiplication order.** The correct product is $QK^T$, not $Q^TK$ or $KQ^T$. $QK^T$ produces shape $(n, m)$ -- one row per query, one column per key. Transposing the wrong matrix gives the wrong shape or a transposed attention pattern.</span>
* <span style="font-size: 14px;">**Softmax along the wrong axis.** Softmax must be applied along the key dimension (columns), independently for each query (row). Applying along the query dimension normalizes across queries instead of across keys -- answering "how do queries compete for a key" rather than "how does one query distribute across keys."</span>
* <span style="font-size: 14px;">**Numerical overflow in softmax.** Large $z_i$ causes $e^{z_i}$ to overflow. Always subtract the row maximum before exponentiating: $e^{z_i - \max(z)}$. Mathematically equivalent (translation invariance) but numerically stable.</span>
* <span style="font-size: 14px;">**Confusing $d_k$ and $d_v$.** The scaling uses $d_k$ (key/query dimension), not $d_v$ (value dimension). Many implementations have $d_k = d_v$, hiding the bug. The variance argument applies to $q^T k$ in $d_k$-dimensional space.</span>
* <span style="font-size: 14px;">**Assuming square matrices.** $Q$ and $K$ can have different row counts ($n \neq m$). Cross-attention always has $n \neq m$. In self-attention with KV caching, $Q$ has 1 row while $K$ has $t$ rows (all past tokens).</span>

---