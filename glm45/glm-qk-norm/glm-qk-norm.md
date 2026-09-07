# <span style="font-size: 20px;">QK-Norm</span>

<span style="font-size: 14px;">QK-Norm applies an RMSNorm to the query and key projections (along the head dimension) right before the attention dot product. Introduced by Henry et al. (2020) to stabilize attention logits, it is enabled in GLM-4.5 via the `use_qk_norm=True` config flag and is now a standard ingredient in modern open-weight LLMs (Falcon, Gemma-2, OLMo, GLM-4.5).</span>

---

## <span style="font-size: 16px;">The Problem: Unstable Attention Logits</span>

<span style="font-size: 14px;">Scaled dot-product attention computes raw scores $s_{ij} = q_i^\top k_j / \sqrt{d_k}$ before the softmax. Two failure modes appear at scale:</span>

* <span style="font-size: 14px;">**Logit explosion.** As $\|q\|$ and $\|k\|$ grow during training, the $\sqrt{d_k}$ scaling becomes insufficient. Logits drift into regions where softmax saturates and gradients vanish, freezing attention patterns.</span>
* <span style="font-size: 14px;">**Mixed-precision overflow.** In `bfloat16` or `float16`, large pre-softmax logits can overflow to $\pm\infty$, producing NaNs that propagate through the network.</span>

<span style="font-size: 14px;">Henry et al. observed both effects in deep encoder-decoder models and proposed normalizing $q$ and $k$ before the dot product, decoupling the logit scale from the projection norms.</span>

---

## <span style="font-size: 16px;">The Formulation</span>

<span style="font-size: 14px;">Given $q \in \mathbb{R}^{n \times H_q \times d}$ and $k \in \mathbb{R}^{n \times H_{kv} \times d}$ (post-projection, pre-attention), QK-Norm computes:</span>

$$
\hat{q} = \gamma_q \odot \frac{q}{\sqrt{\frac{1}{d}\sum_{i=1}^d q_i^2 + \epsilon}}, \qquad \hat{k} = \gamma_k \odot \frac{k}{\sqrt{\frac{1}{d}\sum_{i=1}^d k_i^2 + \epsilon}}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$d$ is `head_dim`. The mean of squares is taken along the last axis (per token, per head).</span>
* <span style="font-size: 14px;">$\gamma_q, \gamma_k \in \mathbb{R}^d$ are learned per-channel scales, shared across all heads and all tokens.</span>
* <span style="font-size: 14px;">$\epsilon$ is a small constant (GLM-4.5 uses $10^{-5}$) preventing division by zero on zero-magnitude vectors.</span>

<span style="font-size: 14px;">After QK-Norm the attention output is the usual:</span>

$$
\text{Attention}(\hat{q}, \hat{k}, v) = \text{softmax}\!\left(\frac{\hat{q}\hat{k}^\top}{\sqrt{d}}\right) v
$$

<span style="font-size: 14px;">Note that $v$ is **not** normalized: only the inputs to the dot product are.</span>

---

## <span style="font-size: 16px;">Why RMSNorm and Not LayerNorm</span>

<span style="font-size: 14px;">The original 2020 paper used LayerNorm (with mean centering). Modern implementations (GLM-4.5, Gemma-2, OLMo) use RMSNorm. Three reasons:</span>

* <span style="font-size: 14px;">**Cheaper.** RMSNorm skips the mean subtraction. With $H \times d$ activations per token, that is one fewer reduction and one fewer broadcast subtract.</span>
* <span style="font-size: 14px;">**RoPE-compatible.** Subtracting the mean would shift the rotated coordinates relative to each other; RMSNorm preserves the rotation structure because it only rescales.</span>
* <span style="font-size: 14px;">**Empirically equivalent.** Zhang and Sennrich (2019) showed RMSNorm matches LayerNorm quality on transformers, so the mean-centering offers no measurable benefit here.</span>

---

## <span style="font-size: 16px;">Where GLM-4.5 Places It</span>

<span style="font-size: 14px;">In the `Glm4MoeAttention.forward` flow the order is:</span>

<span style="font-size: 14px;">1. **Linear projections:** $q = xW_Q$, $k = xW_K$, $v = xW_V$. Reshape to `(n, num_heads, head_dim)` and `(n, num_kv_heads, head_dim)`.</span>

<span style="font-size: 14px;">2. **QK-Norm:** apply `RMSNorm(head_dim)` to $q$ and $k$ independently with learned scales $\gamma_q, \gamma_k$. This is the step implemented here.</span>

<span style="font-size: 14px;">3. **Partial RoPE:** rotate the first 64 channels of each head with rotary embeddings; pass the remaining channels through unchanged.</span>

<span style="font-size: 14px;">4. **GQA expansion:** repeat $k$ and $v$ along the head axis so each query head finds a matching key/value head.</span>

<span style="font-size: 14px;">5. **Scaled dot-product attention** with the $1/\sqrt{d}$ scaling.</span>

<span style="font-size: 14px;">The norm sits **before** RoPE in GLM-4.5, matching the order in Llama-3.x and Gemma-2. Order matters: applying RMSNorm after RoPE would mix rotated and unrotated channels in the variance estimate.</span>

---

## <span style="font-size: 16px;">Why Per-Channel Gamma Instead of a Scalar</span>

<span style="font-size: 14px;">An obvious simplification would be a single learned scalar per head, or even one scalar for the whole layer. The per-channel vector wins for three reasons:</span>

* <span style="font-size: 14px;">**Channel-specific dynamic range.** Different head channels carry different kinds of information after the linear projection. Some encode position-like features that benefit from sharper attention; others encode semantic features that need broader attention. A per-channel scale lets each direction in head space tune its contribution independently.</span>
* <span style="font-size: 14px;">**Negligible cost.** A vector of length $d_{\text{head}}$ is the same shape as the bias of a typical projection and costs the same to store and learn.</span>
* <span style="font-size: 14px;">**Matches the residual stream norm.** GLM-4.5 already uses a per-channel `RMSNorm` on the residual stream input. Using the same parameterization on Q and K keeps the codebase consistent and lets the same fused kernels handle both.</span>

---

## <span style="font-size: 16px;">Comparison With Cosine Attention and Scaled Init</span>

<span style="font-size: 14px;">QK-Norm is one of three families of stability fixes for attention logits:</span>

* <span style="font-size: 14px;">**Scaled initialization** (e.g., $1/\sqrt{N}$ on output projections, Megatron-style): keeps logits small at init but cannot prevent drift later in training.</span>
* <span style="font-size: 14px;">**Cosine attention** (Liu et al. 2021, Swin V2): replaces $q^\top k$ with $q^\top k / (\|q\|\|k\|)$, giving logits in $[-1, 1]$. Effective but discards magnitude information and changes the inductive bias.</span>
* <span style="font-size: 14px;">**QK-Norm**: normalizes magnitude but retains learnable scales $\gamma_q, \gamma_k$, so the network can still produce sharp or diffuse attention. The post-norm vector has RMS equal to the learned scale, not strictly unit norm.</span>

<span style="font-size: 14px;">Empirically, QK-Norm strikes the best balance for billion-parameter LLMs: stable enough for `bf16` training, expressive enough not to hurt downstream loss.</span>

---

## <span style="font-size: 16px;">Interaction With Grouped-Query Attention</span>

<span style="font-size: 14px;">GLM-4.5 uses GQA: `num_heads` query heads share `num_kv_heads` key/value heads (typical ratio 4:1 or 8:1). QK-Norm interacts with GQA in two important ways:</span>

* <span style="font-size: 14px;">**$\gamma_q$ and $\gamma_k$ are per-channel, not per-head.** Each is a 1D vector of length `head_dim`. The same scale is reused across all heads. This keeps parameter count tiny: $2 \cdot d$ extra parameters per layer, regardless of how many heads there are.</span>
* <span style="font-size: 14px;">**Normalization happens before head repetition.** The norm is applied while $k$ still has `num_kv_heads` heads. Repeating $k$ across the head axis afterwards copies an already-normalized tensor, so the repeated heads share identical RMS profiles. Doing it the other way around would normalize $k$ values that were about to be replicated, wasting compute.</span>

<span style="font-size: 14px;">A subtle consequence: because $\gamma_k$ is per-channel and shared across `num_kv_heads`, every KV head is forced to use the same channel-wise scale profile. This is fine in practice (the projection matrix $W_K$ can absorb any per-head differences), and it is the same constraint that already applies to the RMSNorm on the residual stream.</span>

---

## <span style="font-size: 16px;">Parameter Count and Compute</span>

<span style="font-size: 14px;">QK-Norm adds a tiny number of parameters relative to the rest of the layer:</span>

* <span style="font-size: 14px;">**Parameters:** $2 \cdot d_{\text{head}}$ per attention layer. For GLM-4.5 with $d_{\text{head}} = 128$ and $46$ MoE layers + dense layers, the total is around $12$K extra parameters across the model. Compare with the $W_Q, W_K, W_V, W_O$ projections, which contribute hundreds of millions.</span>
* <span style="font-size: 14px;">**Flops at inference:** for a sequence of length $n$ and $H$ total heads, one RMSNorm costs $O(n \cdot H \cdot d)$. Versus the $O(n^2 \cdot H \cdot d)$ attention dot product, the norm is asymptotically free.</span>
* <span style="font-size: 14px;">**KV cache:** because the norm is applied before caching, the cached keys are already normalized. Decoding a new token reuses cached $\hat{k}$ directly and only normalizes the new query and new key. This matches how dense LayerNorm-style attention caches work.</span>

---

## Worked Numerical Example ($d = 4$)

<span style="font-size: 14px;">Take a single query head vector $q = [1, 2, 2, 1]$, key $k = [0.5, 0.5, 0.5, 0.5]$, scales $\gamma_q = \gamma_k = [1, 1, 1, 1]$, $\epsilon = 10^{-5}$.</span>

<span style="font-size: 14px;">1. **Mean of squares for $q$:** $(1 + 4 + 4 + 1)/4 = 2.5$.</span>

<span style="font-size: 14px;">2. **RMS for $q$:** $\sqrt{2.5 + 10^{-5}} \approx 1.5811$.</span>

<span style="font-size: 14px;">3. **$\hat{q}$:** $[1, 2, 2, 1] / 1.5811 \approx [0.6325, 1.2649, 1.2649, 0.6325]$.</span>

<span style="font-size: 14px;">4. **Mean of squares for $k$:** $4 \cdot 0.25 / 4 = 0.25$.</span>

<span style="font-size: 14px;">5. **RMS for $k$:** $\sqrt{0.25 + 10^{-5}} \approx 0.5000$.</span>

<span style="font-size: 14px;">6. **$\hat{k}$:** $[0.5, 0.5, 0.5, 0.5] / 0.5 = [1, 1, 1, 1]$.</span>

<span style="font-size: 14px;">7. **Pre-norm logit:** $q \cdot k / \sqrt{4} = (0.5 + 1 + 1 + 0.5) / 2 = 1.5$.</span>

<span style="font-size: 14px;">8. **Post-norm logit:** $\hat{q} \cdot \hat{k} / \sqrt{4} = (0.6325 + 1.2649 + 1.2649 + 0.6325)/2 = 1.8974$.</span>

<span style="font-size: 14px;">Now imagine $q$ scaled by 100 during a noisy training step. The pre-norm logit blows up to $150$; the post-norm logit is unchanged at $1.8974$ because RMSNorm strips the magnitude.</span>

---

## <span style="font-size: 16px;">Effect on Training Dynamics</span>

<span style="font-size: 14px;">QK-Norm changes the optimization landscape in measurable ways:</span>

* <span style="font-size: 14px;">**Smaller weight-norm growth.** Without QK-Norm, $W_Q$ and $W_K$ are weakly regularized: the only pressure on their magnitude comes through downstream gradients. With QK-Norm, the post-norm output is scale-invariant in $W_Q$ and $W_K$, so the optimizer can freely shrink the projection matrices while the network still produces the same logits. Combined with weight decay, this leads to systematically smaller $\|W_Q\|, \|W_K\|$ at convergence.</span>
* <span style="font-size: 14px;">**Decoupled learning rates for $\gamma$.** The norm parameters $\gamma_q, \gamma_k$ control logit temperature directly. They are typically initialized to one and stay close to one through training; their effective gradient scale differs from $W_Q$ and $W_K$, which is why some training setups (Megatron, NeMo) apply a smaller learning rate or zero weight decay to norm parameters.</span>
* <span style="font-size: 14px;">**No more loss spikes from attention.** The most cited benefit. Models trained without QK-Norm at scale (e.g., the original PaLM training logs) show occasional loss spikes attributed to attention logit explosions; QK-Norm essentially eliminates this failure mode.</span>

---

## <span style="font-size: 16px;">Modern Variants and Where the Field Is Headed</span>

<span style="font-size: 14px;">QK-Norm has become a default in open LLMs released since 2024. A few notable choices:</span>

* <span style="font-size: 14px;">**Gemma-2 (2024):** uses QK-Norm with RMSNorm, identical structure to GLM-4.5. Reports cleaner loss curves on `bf16` training.</span>
* <span style="font-size: 14px;">**OLMo-2 (Groeneveld et al., 2024):** adopts QK-Norm and credits it for resolving spikes in pre-training loss.</span>
* <span style="font-size: 14px;">**GLM-4.5 (2025, arxiv 2508.06471):** ships `use_qk_norm=True` by default and pairs it with partial RoPE for additional stability.</span>
* <span style="font-size: 14px;">**Falcon-2:** uses QK-LayerNorm (with mean centering) rather than QK-RMSNorm, the only major model that still keeps the centering step.</span>

<span style="font-size: 14px;">A related technique is **Query Norm**, used in some early experiments (e.g., GLU variants), which normalizes only $q$. In practice symmetric normalization of both $q$ and $k$ stabilizes the logit distribution far better, which is why GLM-4.5 and peers normalize both.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Normalizing over the wrong axis.** RMSNorm runs along `head_dim` (the last axis). Reducing along the head axis instead averages out all heads and produces a single shared scale, destroying per-head specialization.</span>
* <span style="font-size: 14px;">**Sharing one gamma between Q and K.** GLM-4.5 stores `q_norm.weight` and `k_norm.weight` as separate parameters. Reusing $\gamma_q$ for $k$ ties their scales and reduces capacity. The reference implementation uses two distinct learnable vectors.</span>
* <span style="font-size: 14px;">**Forgetting the gamma multiplication.** Without the learned scale, $\hat{q}$ and $\hat{k}$ have fixed RMS of 1, which is too restrictive: the network needs the freedom to set logit scale per channel. Models trained without gamma show worse perplexity.</span>
* <span style="font-size: 14px;">**Using variance instead of mean of squares.** `torch.var` subtracts the mean before squaring. RMSNorm does **not** mean-center: the denominator is $\sqrt{\text{mean}(x^2) + \epsilon}$, not $\sqrt{\text{var}(x) + \epsilon}$. The two are equal only when the input already has zero mean.</span>
* <span style="font-size: 14px;">**Dropping epsilon.** On the rare token whose head vector is all zeros (e.g., padded positions zeroed out upstream), the RMS is exactly $0$ and division produces NaN. The $10^{-5}$ floor is cheap insurance.</span>
* <span style="font-size: 14px;">**Applying QK-Norm after RoPE.** RoPE rotates pairs of channels; normalizing after rotation mixes the rotated coordinates into the variance estimate and breaks the rotation invariance the position encoding relies on. The standard order is project, normalize, then rotate.</span>
* <span style="font-size: 14px;">**Casting to `float16` before the norm.** The squaring step is the first place attention numerics blow up. Production implementations compute RMSNorm in `float32` even when the surrounding model runs in `bf16`. This is one reason QK-Norm helps mixed-precision stability.</span>
* <span style="font-size: 14px;">**Normalizing the value tensor too.** Only $q$ and $k$ are normalized. Applying the same operation to $v$ would distort the magnitude of the attention output and hurt training.</span>

---
