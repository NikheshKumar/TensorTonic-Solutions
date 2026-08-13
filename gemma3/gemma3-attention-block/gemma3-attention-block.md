# <span style="font-size: 20px;">Gemma 3 Attention Block</span>

<span style="font-size: 14px;">The Gemma 3 attention block is the complete attention sub-layer used in every Transformer block of the Gemma 3 architecture (Google DeepMind, 2025). It chains seven operations: RMSNorm on input, Q/K/V projection and reshape, QK-Norm, RoPE, GQA head expansion, masked attention with layer-dependent routing, and output projection with a residual connection.</span>

<span style="font-size: 14px;">This block integrates every attention-related component -- QK-Norm, RoPE, GQA, sliding window masking -- into one coherent pipeline. The ordering of these operations matters: normalizing before rotating, rotating before expanding, expanding before computing attention. Getting any step out of order produces subtly wrong results.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The attention block is one of two sub-layers inside each Gemma 3 Transformer block (the other being the feedforward network). It receives a hidden state tensor $x \in \mathbb{R}^{B \times S \times d}$ and returns an output of the same shape with a residual connection. The seven steps in order:</span>

* <span style="font-size: 14px;">**Step 1 -- Pre-attention RMSNorm:** Normalize the input to stabilize magnitudes before projection.</span>
* <span style="font-size: 14px;">**Step 2 -- Q/K/V Projection + Reshape:** Linear projections produce queries, keys, and values. Reshape into multi-head format: $h_q$ query heads and $h_{kv}$ key-value heads.</span>
* <span style="font-size: 14px;">**Step 3 -- QK-Norm:** Apply RMSNorm independently to Q and K per head to prevent attention logit explosion.</span>
* <span style="font-size: 14px;">**Step 4 -- RoPE:** Apply rotary position embeddings to normalized Q and K.</span>
* <span style="font-size: 14px;">**Step 5 -- GQA Expansion:** Repeat each KV head $n_{\text{rep}} = h_q / h_{kv}$ times so every query head has a matching key-value pair.</span>
* <span style="font-size: 14px;">**Step 6 -- Masked Attention:** Compute scaled dot-product attention. Global layers use full causal mask, local layers use sliding window.</span>
* <span style="font-size: 14px;">**Step 7 -- Output Projection + Residual:** Concatenate heads, project to $d$ dimensions, add to original input.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Step 1 -- RMSNorm on input:**</span>

$$
\hat{x} = \frac{x \cdot \gamma}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}}
$$

<span style="font-size: 14px;">where $\gamma \in \mathbb{R}^d$ is a learnable scale vector and $\epsilon \approx 10^{-6}$.</span>

<span style="font-size: 14px;">**Step 2 -- Projection and reshape:**</span>

$$
Q = \hat{x} \, W_q^T, \quad K = \hat{x} \, W_k^T, \quad V = \hat{x} \, W_v^T
$$

<span style="font-size: 14px;">$W_q \in \mathbb{R}^{(h_q \cdot d_h) \times d}$, $W_k, W_v \in \mathbb{R}^{(h_{kv} \cdot d_h) \times d}$. Reshape Q to $(B, h_q, S, d_h)$, K and V to $(B, h_{kv}, S, d_h)$.</span>

<span style="font-size: 14px;">**Step 3 -- QK-Norm:**</span>

$$
Q_{\text{norm}} = \frac{Q \cdot \gamma_q}{\sqrt{\frac{1}{d_h}\sum_{j=1}^{d_h} Q_j^2 + \epsilon}}, \quad K_{\text{norm}} = \frac{K \cdot \gamma_k}{\sqrt{\frac{1}{d_h}\sum_{j=1}^{d_h} K_j^2 + \epsilon}}
$$

<span style="font-size: 14px;">$\gamma_q, \gamma_k \in \mathbb{R}^{d_h}$ are learnable per-head scale vectors. V is not normalized.</span>

<span style="font-size: 14px;">**Step 4 -- RoPE:**</span>

$$
Q'_{2i} = Q_{2i} \cos\theta_i - Q_{2i+1} \sin\theta_i, \quad Q'_{2i+1} = Q_{2i} \sin\theta_i + Q_{2i+1} \cos\theta_i
$$

<span style="font-size: 14px;">Same rotation applied to K. Angles $\theta_i = m \cdot b^{-2i/d_h}$ depend on position $m$ and dimension index.</span>

<span style="font-size: 14px;">**Step 5 -- GQA expansion:**</span>

$$
n_{\text{rep}} = h_q / h_{kv}, \quad K_{\text{exp}} = \text{repeat}(K, n_{\text{rep}}), \quad V_{\text{exp}} = \text{repeat}(V, n_{\text{rep}})
$$

<span style="font-size: 14px;">K, V expand from $(B, h_{kv}, S, d_h)$ to $(B, h_q, S, d_h)$.</span>

<span style="font-size: 14px;">**Step 6 -- Masked attention:**</span>

$$
\text{Attn} = \text{softmax}\!\left(\frac{Q' \, K_{\text{exp}}^T}{\sqrt{d_h}} + \text{mask}\right) V_{\text{exp}}
$$

<span style="font-size: 14px;">Mask is full causal ($-\infty$ above diagonal) for global layers or sliding window ($-\infty$ outside recent $w$ positions) for local layers.</span>

<span style="font-size: 14px;">**Step 7 -- Output projection + residual:**</span>

$$
\text{out} = x + \text{reshape}(\text{Attn}) \cdot W_o^T
$$

<span style="font-size: 14px;">$W_o \in \mathbb{R}^{d \times (h_q \cdot d_h)}$. Attention output reshaped from $(B, h_q, S, d_h)$ to $(B, S, h_q \cdot d_h)$ before projection.</span>

---

## <span style="font-size: 16px;">Step-by-Step Walkthrough</span>

<span style="font-size: 14px;">**Step 1: Pre-attention RMSNorm.** The input $x$ arrives from the residual stream. RMSNorm scales each token's hidden vector to a controlled magnitude without centering (no mean subtraction). This ensures Q/K/V projections receive consistent-scale inputs regardless of layer depth or residual accumulation.</span>

<span style="font-size: 14px;">**Step 2: Q/K/V Projection + Reshape.** Three bias-free linear layers produce raw Q, K, V. The number of query heads ($h_q$) exceeds KV heads ($h_{kv}$) because of GQA. After projection, tensors are reshaped from $(B, S, h \cdot d_h)$ to $(B, h, S, d_h)$ to separate individual heads.</span>

<span style="font-size: 14px;">**Step 3: QK-Norm.** RMSNorm on Q and K independently, per head and per token, along $d_h$. This prevents dot-product logits from exploding during training. V is untouched because V magnitude does not affect the softmax distribution.</span>

<span style="font-size: 14px;">**Step 4: RoPE.** Rotary embeddings inject position by rotating dimension pairs in Q and K. Since RoPE is norm-preserving, controlled magnitudes from QK-Norm carry through unchanged. This is why QK-Norm must precede RoPE.</span>

<span style="font-size: 14px;">**Step 5: GQA Expansion.** Each KV head is duplicated $n_{\text{rep}}$ times to match query head count. KV cache stores only $h_{kv}$ heads, but attention proceeds as if there are $h_q$ KV heads.</span>

<span style="font-size: 14px;">**Step 6: Masked Attention.** The score matrix $Q K^T / \sqrt{d_h}$ has shape $(B, h_q, S, S)$. A mask is added before softmax. Global layers use full causal; local layers use sliding window of width $w$.</span>

<span style="font-size: 14px;">**Step 7: Output Projection + Residual.** Per-head outputs are concatenated into $(B, S, h_q \cdot d_h)$, projected through $W_o$ back to $d$ dimensions, and added to the original input $x$ (not the normalized $\hat{x}$).</span>

---

## <span style="font-size: 16px;">How It Differs from Standard Attention</span>

<span style="font-size: 14px;">**QK-Norm (absent in standard attention).** Standard attention has no normalization between projection and dot product. Logits can grow arbitrarily as weight norms increase during training. QK-Norm adds dynamic magnitude control the fixed $1/\sqrt{d_h}$ scaling cannot provide.</span>

<span style="font-size: 14px;">**Grouped Query Attention (standard uses equal heads).** Standard MHA has $h_q = h_{kv}$. Gemma 3 uses fewer KV heads, reducing KV cache memory by $h_q / h_{kv}$ during inference with negligible quality loss.</span>

<span style="font-size: 14px;">**Sliding window masking (standard uses only full causal).** Standard attention uses full causal at every layer. Gemma 3 routes local layers to sliding window, reducing per-layer cost from $O(S^2)$ to $O(S \cdot w)$ and encouraging local-pattern specialization.</span>

<span style="font-size: 14px;">**Operation ordering.** Gemma 3 pipeline: RMSNorm, project, QK-Norm, RoPE, GQA expand, attention, output project. Standard: project, attention, output project. The extra internal steps require strict ordering.</span>

---

## <span style="font-size: 16px;">GQA Mechanism</span>

<span style="font-size: 14px;">Grouped Query Attention reduces memory by sharing key-value heads across query heads. The group count is $n_{\text{rep}} = h_q / h_{kv}$.</span>

<span style="font-size: 14px;">**How expansion works.** After Q has shape $(B, h_q, S, d_h)$ and K, V have shape $(B, h_{kv}, S, d_h)$, each KV head is repeated $n_{\text{rep}}$ times along dimension 1. If $h_q = 8$ and $h_{kv} = 2$, then $n_{\text{rep}} = 4$: each of 2 KV heads is copied 4 times. Query heads 0-3 share KV head 0, heads 4-7 share KV head 1.</span>

<span style="font-size: 14px;">**Implementation.** The expansion can be: `K.unsqueeze(2).expand(B, h_kv, n_rep, S, d_h).reshape(B, h_q, S, d_h)`. The expand creates a view with stride 0 (no new memory).</span>

<span style="font-size: 14px;">**Spectrum.** MHA: $n_{\text{rep}} = 1$ (no sharing). MQA: $h_{kv} = 1$ (all queries share one KV head). GQA sits between, balancing quality and efficiency.</span>

---

## <span style="font-size: 16px;">Layer Routing: Global vs Local Mask</span>

<span style="font-size: 14px;">Gemma 3 assigns each layer a type -- global or local -- which determines the attention mask.</span>

<span style="font-size: 14px;">**Global layers** use a full causal mask. Token $t$ attends to positions $0, 1, \ldots, t$. Positions $j > t$ are masked to $-\infty$. This captures long-range dependencies across the entire context.</span>

<span style="font-size: 14px;">**Local layers** use a sliding window mask. Token $t$ attends only to positions $\max(0, t - w + 1), \ldots, t$. Positions outside this window get $-\infty$ in addition to the causal constraint.</span>

<span style="font-size: 14px;">**Routing rule.** Assignment is deterministic by layer index. Every $r$-th layer is global, the rest local. The `local_ratio` parameter controls the fraction. For example, with $r = 6$, layers 0, 6, 12, ... are global and all others are local.</span>

<span style="font-size: 14px;">**Why mix?** Pure global is $O(S^2)$ per layer. Pure local loses long-range connections. Mixing gives cheap local layers for nearby patterns and periodic global layers for full-sequence information flow.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Gemma 3 (Google DeepMind, 2025) uses this attention block at every layer. Key configurations:</span>

* <span style="font-size: 14px;">**1B:** $d = 1536$, $h_q = 8$, $h_{kv} = 4$, $d_h = 192$, $n_{\text{rep}} = 2$.</span>
* <span style="font-size: 14px;">**4B:** $d = 2304$, $h_q = 8$, $h_{kv} = 4$, $d_h = 256$, $n_{\text{rep}} = 2$.</span>
* <span style="font-size: 14px;">**12B:** $d = 3840$, $h_q = 16$, $h_{kv} = 8$, $d_h = 256$, $n_{\text{rep}} = 2$.</span>
* <span style="font-size: 14px;">**27B:** $d = 4608$, $h_q = 32$, $h_{kv} = 16$, $d_h = 128$, $n_{\text{rep}} = 2$.</span>

<span style="font-size: 14px;">Across all sizes, $n_{\text{rep}} = 2$: each KV head serves exactly two query heads. The sliding window size $w$ is 512 for local layers. Gemma 3 dropped logit softcapping (used in Gemma 2) in favor of QK-Norm, addressing attention logit stability upstream rather than clamping after the fact.</span>

<span style="font-size: 14px;">The interleaved global/local pattern draws on Longformer (Beltagy et al., 2020) and BigBird (Zaheer et al., 2020), but Gemma 3 simplifies by making entire layers either global or local rather than mixing patterns within a single layer.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace all 7 steps with $B = 1$, $S = 3$, $d = 4$, $h_q = 2$, $h_{kv} = 1$, $d_h = 2$, window $w = 2$.</span>

<span style="font-size: 14px;">**Input** $x$ (shape $(1, 3, 4)$):</span>

$$
x = \begin{bmatrix} 1.0 & -0.5 & 0.8 & 0.2 \\ -0.3 & 0.7 & 0.4 & -0.6 \\ 0.9 & 0.1 & -0.3 & 0.5 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 1: RMSNorm** ($\gamma = [1,1,1,1]$). Token 0: $\text{RMS} = \sqrt{(1.0 + 0.25 + 0.64 + 0.04)/4} = \sqrt{0.4825} \approx 0.695$. Result: $\hat{x}_0 \approx [1.440, -0.720, 1.152, 0.288]$. Similarly: $\hat{x}_1 \approx [-0.580, 1.354, 0.774, -1.161]$, $\hat{x}_2 \approx [1.566, 0.174, -0.522, 0.870]$.</span>

<span style="font-size: 14px;">**Step 2: Project + Reshape.** After projection ($W_q \in \mathbb{R}^{4 \times 4}$, $W_k, W_v \in \mathbb{R}^{2 \times 4}$) and reshape:</span>

* <span style="font-size: 14px;">Q head 0: $[[0.8, -0.3], [0.5, 0.9], [-0.4, 0.6]]$; Q head 1: $[[0.2, 0.7], [-0.6, 0.4], [0.3, -0.8]]$</span>
* <span style="font-size: 14px;">K head 0: $[[0.5, 0.4], [-0.2, 0.8], [0.7, -0.1]]$; V head 0: $[[0.3, -0.5], [0.6, 0.2], [-0.1, 0.8]]$</span>

<span style="font-size: 14px;">**Step 3: QK-Norm.** Q head 0, token 0: $[0.8, -0.3]$. $\text{RMS} = \sqrt{(0.64 + 0.09)/2} \approx 0.604$. $Q_{\text{norm}} \approx [1.324, -0.497]$. K head 0, token 0: $\text{RMS} \approx 0.453$. $K_{\text{norm}} \approx [1.104, 0.883]$.</span>

<span style="font-size: 14px;">**Step 4: RoPE.** Position 0 ($\theta = 0$): vectors unchanged. Position 1 with $\theta_0 = 0.3$: $Q'_0 = 1.324 \cos(0.3) + 0.497 \sin(0.3) \approx 1.412$, $Q'_1 = 1.324 \sin(0.3) - 0.497 \cos(0.3) \approx -0.083$. Same rotation applied to K.</span>

<span style="font-size: 14px;">**Step 5: GQA Expansion.** $n_{\text{rep}} = 2/1 = 2$. Single KV head copied to match both query heads: K, V go from $(1,1,3,2)$ to $(1,2,3,2)$.</span>

<span style="font-size: 14px;">**Step 6: Masked Attention (local, $w=2$).** Scores: $Q' K'^T / \sqrt{2}$, giving $3 \times 3$ per head. Sliding window mask:</span>

$$
\text{mask} = \begin{bmatrix} 0 & -\infty & -\infty \\ 0 & 0 & -\infty \\ -\infty & 0 & 0 \end{bmatrix}
$$

<span style="font-size: 14px;">Token 2 cannot attend to position 0 (outside window). After softmax and V multiplication, each head outputs $(1, 3, 2)$. For a global layer, token 2 would also see position 0.</span>

<span style="font-size: 14px;">**Step 7: Output + Residual.** Concatenate heads to $(1, 3, 4)$. Project through $W_o$ and add to original $x$. Output: $(1, 3, 4)$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Applying RoPE before QK-Norm.**</span>

<span style="font-size: 14px;">The correct order is QK-Norm first, then RoPE. RoPE is norm-preserving, so normalizing first means controlled magnitudes carry through to the dot product. Reversing the order lets normalization erase the directional information RoPE encoded in dimension pairs.</span>

<span style="font-size: 14px;">**2. Expanding KV heads before RoPE.**</span>

<span style="font-size: 14px;">RoPE should be applied to the original $h_{kv}$ key heads, not the expanded $h_q$ copies. Applying RoPE after expansion wastes computation (rotating identical copies) and risks bugs if different frequencies are used per copy. Apply RoPE first, then expand.</span>

<span style="font-size: 14px;">**3. Wrong GQA expansion dimension.**</span>

<span style="font-size: 14px;">The repeat must happen along the head dimension (dim 1 in $(B, h_{kv}, S, d_h)$). Repeating along the sequence or feature dimension produces wrong shapes or semantics. Each KV head must be copied $n_{\text{rep}}$ times consecutively along the head axis.</span>

<span style="font-size: 14px;">**4. Using the wrong mask type for the layer.**</span>

<span style="font-size: 14px;">Global layers need full causal; local layers need sliding window. Full causal on a local layer defeats sliding window benefits. Sliding window on a global layer blocks long-range attention.</span>

<span style="font-size: 14px;">**5. Adding residual to $\hat{x}$ instead of $x$.**</span>

<span style="font-size: 14px;">The residual adds to the original $x$, not the normalized $\hat{x}$. Adding to $\hat{x}$ means the residual stream is normalized at every layer, destroying the identity-mapping property of residual connections.</span>

<span style="font-size: 14px;">**6. Forgetting $1/\sqrt{d_h}$ scaling.**</span>

<span style="font-size: 14px;">Even with QK-Norm, the $1/\sqrt{d_h}$ scaling is required. QK-Norm controls individual vector norms; the scaling compensates for the expected dot product of two unit-norm vectors growing with $\sqrt{d_h}$. Without it, logits are too large by $\sqrt{d_h}$.</span>

<span style="font-size: 14px;">**7. Sharing QK-Norm parameters across heads.**</span>

<span style="font-size: 14px;">Each head needs its own $\gamma_q$ and $\gamma_k$. Sharing forces identical per-dimension scaling across all heads, reducing attention pattern diversity. Gemma 3 uses per-head parameters: $h_q$ separate $\gamma_q$ and $h_{kv}$ separate $\gamma_k$.</span>

---
