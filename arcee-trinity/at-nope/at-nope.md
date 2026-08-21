# <span style="font-size: 20px;">NoPE: Causal Attention Without Positional Encoding</span>

<span style="font-size: 14px;">NoPE (No Positional Encoding) is a variant of multi-head causal self-attention that deliberately omits all positional encoding from its queries and keys. In Arcee Trinity, a fraction of transformer layers use NoPE instead of RoPE, allowing those layers to attend purely based on content similarity rather than position-aware similarity. The result is a hybrid architecture where some layers know where tokens are and other layers only care about what tokens are.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">NoPE is standard multi-head causal self-attention with one deliberate omission: no positional encoding is applied to the query and key vectors. In a typical RoPE layer, queries and keys are rotated by position-dependent angles before the dot product. NoPE skips this rotation entirely.</span>

<span style="font-size: 14px;">This means the attention score between two tokens depends exclusively on their content, not on where they sit in the sequence. The dot product $q_i^T k_j$ measures only how semantically related position $i$'s content is to position $j$'s content, with no positional bias. This is what "content-based similarity" means.</span>

<span style="font-size: 14px;">A NoPE layer performs the same operations as any attention layer:</span>

* <span style="font-size: 14px;">**RMSNorm pre-normalization:** Normalize input hidden states before projecting.</span>
* <span style="font-size: 14px;">**Linear projections:** Project into queries, keys, and values.</span>
* <span style="font-size: 14px;">**Scaled dot-product attention with causal masking:** Score, mask, softmax, weighted sum.</span>
* <span style="font-size: 14px;">**Output projection and residual:** Concatenate heads, project back, add original input.</span>

<span style="font-size: 14px;">The only difference is what is missing: no RoPE rotation is applied between the projection step and the dot-product step.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The NoPE layer begins with RMSNorm applied to the input. Given an input vector $x \in \mathbb{R}^{d}$ and a learnable scale parameter $\gamma \in \mathbb{R}^{d}$:</span>

$$
\hat{x} = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma
$$

<span style="font-size: 14px;">RMSNorm differs from LayerNorm in that it does not subtract the mean. It only divides by the root-mean-square of the activations. The $\epsilon$ term (typically $10^{-6}$) prevents division by zero. The element-wise multiplication by $\gamma$ provides a learnable re-scaling.</span>

<span style="font-size: 14px;">Next, the normalized input $\hat{x}$ is projected into queries, keys, and values. For a sequence of $n$ tokens with normalized representations $\hat{X} \in \mathbb{R}^{n \times d}$:</span>

$$
Q = \hat{X} W_q^T, \quad K = \hat{X} W_k^T, \quad V = \hat{X} W_v^T
$$

<span style="font-size: 14px;">where $W_q, W_k \in \mathbb{R}^{d_h \cdot n_h \times d}$ and $W_v \in \mathbb{R}^{d_v \cdot n_h \times d}$. Here $n_h$ is the number of attention heads and $d_h$ is the per-head dimension. The critical difference from a RoPE layer happens right here: in a RoPE layer, $Q$ and $K$ would be rotated by position-dependent matrices. In NoPE, $Q$ and $K$ are used as-is.</span>

<span style="font-size: 14px;">For a single head with $Q_h, K_h \in \mathbb{R}^{n \times d_h}$, the raw attention scores are:</span>

$$
S = \frac{Q_h K_h^T}{\sqrt{d_h}}
$$

<span style="font-size: 14px;">The scaling by $\sqrt{d_h}$ prevents dot products from growing too large as the head dimension increases, which would push softmax into regions with vanishing gradients.</span>

<span style="font-size: 14px;">A causal mask is then applied to prevent each position from attending to future positions:</span>

$$
S_{\text{masked}} = S + M
$$

<span style="font-size: 14px;">where $M \in \mathbb{R}^{n \times n}$ is the causal mask with $M_{ij} = 0$ if $i \geq j$ and $M_{ij} = -\infty$ if $i < j$. The $-\infty$ entries become zero after softmax.</span>

<span style="font-size: 14px;">The attention weights and output are:</span>

$$
A = \text{softmax}(S_{\text{masked}})
$$

$$
O_h = A V_h
$$

<span style="font-size: 14px;">Each row of $O_h$ is a weighted sum of value vectors, where the weights come from the softmax distribution over past and present positions only. Finally, the outputs from all heads are concatenated, projected, and added to the original input:</span>

$$
\text{output} = x + \text{Concat}(O_1, O_2, \ldots, O_{n_h}) W_o^T
$$

<span style="font-size: 14px;">where $W_o \in \mathbb{R}^{d \times d_v \cdot n_h}$. The residual adds back the pre-norm input $x$ (not $\hat{x}$), ensuring gradient flow.</span>

---

## <span style="font-size: 16px;">Why No Positional Encoding Works</span>

<span style="font-size: 14px;">Not all attention patterns benefit from positional information. Consider the two types of patterns a transformer might learn:</span>

* <span style="font-size: 14px;">**Position-dependent patterns:** "Attend to the token 1 or 2 positions back" or "attend to the beginning of the sentence." These patterns fundamentally need positional encoding to function.</span>
* <span style="font-size: 14px;">**Content-dependent patterns:** "Find the most semantically relevant token anywhere in the context" or "find all tokens matching a certain semantic category." These patterns work purely on what tokens mean, regardless of where they appear.</span>

<span style="font-size: 14px;">When every layer uses RoPE, even the content-dependent heads receive positional signal. RoPE naturally biases toward nearby tokens, which may interfere with a head trying to find a semantically relevant token far away in the context.</span>

<span style="font-size: 14px;">By removing positional encoding from some layers, the model gains flexibility. NoPE layers can attend to any token based purely on content similarity without a distance penalty. This is especially valuable for tasks like long-range fact retrieval, coreference resolution across distant spans, and global topic detection.</span>

<span style="font-size: 14px;">There is also a redundancy argument. If most layers already encode position via RoPE, adding position to every layer is redundant. Dedicating some layers to pure content matching may be a more efficient allocation of model capacity.</span>

---

## <span style="font-size: 16px;">How NoPE Differs from Standard Attention</span>

<span style="font-size: 14px;">The difference is surgically minimal. In a RoPE layer, after computing $Q$ and $K$, each vector is multiplied by a position-dependent rotation matrix:</span>

$$
q_m^{\text{RoPE}} = R_m \, q_m, \quad k_m^{\text{RoPE}} = R_m \, k_m
$$

<span style="font-size: 14px;">where $R_m$ is a block-diagonal rotation matrix with angles proportional to position $m$. The dot product $q_i^T k_j$ in a RoPE layer therefore encodes both content similarity and relative position $(i - j)$.</span>

<span style="font-size: 14px;">In a NoPE layer, this rotation step is simply skipped:</span>

$$
q_m^{\text{NoPE}} = q_m, \quad k_m^{\text{NoPE}} = k_m
$$

<span style="font-size: 14px;">That is the entire difference. The dot product $q_i^T k_j$ in a NoPE layer encodes only content similarity. Everything else is identical:</span>

* <span style="font-size: 14px;">**Causal masking:** Still applied. NoPE tokens cannot attend to future tokens.</span>
* <span style="font-size: 14px;">**Scaling:** Still divide by $\sqrt{d_h}$.</span>
* <span style="font-size: 14px;">**Softmax:** Still applied row-wise to produce valid probability distributions.</span>
* <span style="font-size: 14px;">**Multi-head structure:** Still split into multiple heads with separate Q/K/V projections.</span>
* <span style="font-size: 14px;">**RMSNorm:** Still applied to the input before projections.</span>
* <span style="font-size: 14px;">**Residual connection:** Still adds the original input back to the layer output.</span>

---

## <span style="font-size: 16px;">Paper Context: Arcee Trinity's Interleaving Strategy</span>

<span style="font-size: 14px;">Arcee Trinity interleaves NoPE and RoPE layers throughout the transformer stack. It does not use NoPE everywhere -- that would leave the model with no positional information. Instead, a controlled fraction of layers omit positional encoding while the rest use RoPE.</span>

<span style="font-size: 14px;">The interleaving is governed by a parameter called `rope_ratio`, which specifies the fraction of layers that use RoPE. If `rope_ratio = 0.5`, half the layers use RoPE and half use NoPE. A value of $1.0$ means every layer uses RoPE (standard behavior), and $0.0$ would mean pure NoPE with no positional signal at all.</span>

<span style="font-size: 14px;">The interleaving pattern is deterministic and evenly spaced. If 1 in every 4 layers is NoPE, then layers 3, 7, 11, 15, ... are NoPE while the rest are RoPE. This ensures positional information is regularly refreshed and no long stretch of consecutive layers lacks position awareness.</span>

<span style="font-size: 14px;">The design philosophy is that position and content are separable concerns. Dedicating some layers purely to content matching gives the model more expressive power without sacrificing positional understanding.</span>

---

## <span style="font-size: 16px;">The Causal Mask</span>

<span style="font-size: 14px;">Even without positional encoding, a NoPE layer still requires a causal mask. The mask is not about positional encoding -- it is about the autoregressive property. The model must predict each token based only on previous tokens. Attending to future tokens would leak information and break autoregressive generation.</span>

<span style="font-size: 14px;">The causal mask $M \in \mathbb{R}^{n \times n}$ is defined as:</span>

$$
M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}
$$

<span style="font-size: 14px;">When added to the attention scores, the $-\infty$ entries cause those positions to have zero probability after softmax, since $e^{-\infty} = 0$. This ensures that token $i$ can only attend to tokens at positions $0, 1, \ldots, i$.</span>

<span style="font-size: 14px;">In implementation, $-\infty$ is approximated by a very large negative number like $-10^{9}$ or the minimum representable float value. The mask is the same lower-triangular structure used in every autoregressive transformer layer, and it is completely independent of whether positional encoding is present.</span>

<span style="font-size: 14px;">Positional encoding tells the model where tokens are. The causal mask tells the model which tokens it is allowed to see. These are orthogonal concerns. A NoPE layer without a causal mask would be bidirectional attention, which is valid for encoder models (like BERT) but catastrophic for autoregressive generation.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a tiny NoPE layer with 3 tokens, model dimension $d = 4$, and a single attention head ($n_h = 1$, $d_h = 4$).</span>

<span style="font-size: 14px;">**Input.** Let the input hidden states be:</span>

$$
X = \begin{pmatrix} 1.0 & 0.5 & -0.5 & 0.2 \\ 0.3 & -0.7 & 1.2 & 0.4 \\ 0.8 & 0.1 & 0.3 & -0.6 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 1: RMSNorm.** For the first token $x_1 = (1.0, 0.5, -0.5, 0.2)$:</span>

$$
\text{RMS}(x_1) = \sqrt{\frac{1.0^2 + 0.5^2 + 0.5^2 + 0.2^2}{4}} = \sqrt{\frac{1.54}{4}} = \sqrt{0.385} \approx 0.6205
$$

<span style="font-size: 14px;">With $\gamma = (1, 1, 1, 1)$ for simplicity: $\hat{x}_1 = x_1 / 0.6205 \approx (1.612, 0.806, -0.806, 0.322)$. Applying to all tokens:</span>

$$
\hat{X} \approx \begin{pmatrix} 1.612 & 0.806 & -0.806 & 0.322 \\ 0.205 & -0.478 & 0.819 & 0.273 \\ 0.762 & 0.095 & 0.286 & -0.572 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 2: Q, K, V projections.** Using $W_q = W_k = W_v = I_4$ (identity) for clarity, we get $Q = K = V = \hat{X}$. In a real model these are different learned matrices.</span>

<span style="font-size: 14px;">**Step 3: Attention scores.** Compute $S = Q K^T / \sqrt{d_h}$ with $\sqrt{d_h} = 2$. For example, $S_{12}$ (unscaled) is:</span>

$$
q_1^T k_2 = (1.612)(0.205) + (0.806)(-0.478) + (-0.806)(0.819) + (0.322)(0.273) = -0.627
$$

<span style="font-size: 14px;">Computing all entries and dividing by 2:</span>

$$
S = \frac{QK^T}{2} \approx \begin{pmatrix} 2.001 & -0.314 & 0.446 \\ -0.314 & 0.522 & -0.067 \\ 0.446 & -0.067 & 0.489 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 4: Apply causal mask.** Set upper-triangle entries to $-\infty$:</span>

$$
S_{\text{masked}} = \begin{pmatrix} 2.001 & -\infty & -\infty \\ -0.314 & 0.522 & -\infty \\ 0.446 & -0.067 & 0.489 \end{pmatrix}
$$

<span style="font-size: 14px;">**Step 5: Softmax (row-wise).** Row 1 has one entry, so $A_{11} = 1.0$. For row 2:</span>

$$
e^{-0.314} \approx 0.731, \quad e^{0.522} \approx 1.685, \quad \text{sum} = 2.416
$$

$$
A_{21} = 0.731 / 2.416 \approx 0.302, \quad A_{22} = 1.685 / 2.416 \approx 0.698
$$

<span style="font-size: 14px;">For row 3:</span>

$$
e^{0.446} \approx 1.562, \quad e^{-0.067} \approx 0.935, \quad e^{0.489} \approx 1.631, \quad \text{sum} = 4.128
$$

$$
A_{31} \approx 0.378, \quad A_{32} \approx 0.227, \quad A_{33} \approx 0.395
$$

<span style="font-size: 14px;">**Step 6: Weighted value sum.** For token 3: $o_3 = 0.378 \cdot v_1 + 0.227 \cdot v_2 + 0.395 \cdot v_3$:</span>

$$
o_3 \approx \begin{pmatrix}0.609\\0.305\\-0.305\\0.122\end{pmatrix} + \begin{pmatrix}0.047\\-0.109\\0.186\\0.062\end{pmatrix} + \begin{pmatrix}0.301\\0.038\\0.113\\-0.226\end{pmatrix} = \begin{pmatrix}0.957\\0.234\\-0.006\\-0.042\end{pmatrix}
$$

<span style="font-size: 14px;">**Step 7: Residual connection.** With $W_o = I_4$, the final output for token 3 is:</span>

$$
\text{output}_3 = x_3 + o_3 = \begin{pmatrix}0.8\\0.1\\0.3\\-0.6\end{pmatrix} + \begin{pmatrix}0.957\\0.234\\-0.006\\-0.042\end{pmatrix} = \begin{pmatrix}1.757\\0.334\\0.294\\-0.642\end{pmatrix}
$$

<span style="font-size: 14px;">The key observation: token 3's attention weights ($0.378, 0.227, 0.395$) are driven entirely by content similarity. Token 1 gets the most attention not because it is nearby, but because its content vector has the highest dot product with token 3's query. In a RoPE layer, positional rotations would have biased these scores.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the causal mask:** A NoPE layer without a causal mask is bidirectional attention, not autoregressive. It would allow the model to see future tokens during generation, producing invalid outputs. The causal mask is mandatory regardless of whether positional encoding is present.</span>
* <span style="font-size: 14px;">**Confusing NoPE with "no attention":** NoPE does not remove or disable the attention mechanism. It removes positional encoding from the attention computation. The layer still computes full query-key dot products, applies softmax, and produces weighted value sums. It is fully functional attention -- just position-unaware.</span>
* <span style="font-size: 14px;">**Applying RoPE by accident in NoPE layers:** In a hybrid architecture, the code must branch on a per-layer flag to decide whether to apply RoPE. Applying RoPE universally and forgetting to skip it for NoPE layers silently defeats the purpose of the hybrid design.</span>
* <span style="font-size: 14px;">**Incorrect head reshaping:** After the projection produces $Q \in \mathbb{R}^{n \times (n_h \cdot d_h)}$, it must be reshaped to $\mathbb{R}^{n_h \times n \times d_h}$. Mixing up the head and sequence dimensions corrupts the attention computation.</span>
* <span style="font-size: 14px;">**Forgetting RMSNorm before projections:** Arcee Trinity uses pre-normalization. Omitting RMSNorm means projections receive unnormalized inputs, leading to training instability.</span>
* <span style="font-size: 14px;">**Thinking NoPE layers have no positional information at all:** Tokens entering a NoPE layer have already been processed by earlier RoPE layers. Their hidden representations carry positional information baked in from those layers. NoPE does not erase this -- it simply does not add more positional signal via the attention score computation.</span>
* <span style="font-size: 14px;">**Using NoPE for all layers:** With no RoPE layers at all, the model has no mechanism for position-dependent attention patterns. The causal mask provides ordering, but within the visible context all tokens are positionally equivalent. Arcee Trinity avoids this by using NoPE only for a fraction of layers.</span>

---