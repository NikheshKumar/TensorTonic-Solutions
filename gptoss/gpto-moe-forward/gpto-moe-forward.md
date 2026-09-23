# <span style="font-size: 20px;">GPT-OSS MoE Forward Pass</span>

<span style="font-size: 14px;">GPT-OSS replaces the dense FFN of each transformer block with a **sparse Mixture of Experts** (MoE). A learned router sends every token to a small subset of experts (top-$k$), each expert applies a clamped SwiGLU MLP, and the outputs are combined as a router-weighted sum. This is the same sparse-FFN family used by Mixtral and DeepSeek-V3, with GPT-OSS specific choices for the activation (clamped, $\alpha = 1.702$, with a $+1$ bias on the linear branch) that make the block stable under MXFP4-quantized weights.</span>

---

## <span style="font-size: 16px;">Why Sparse MoE</span>

<span style="font-size: 14px;">A dense FFN with hidden width $4 \cdot d_{\text{model}}$ accounts for roughly two thirds of the FLOPs and parameters of a transformer block. Scaling that block makes the model smarter but multiplies inference cost linearly with the new width. Sparse MoE breaks that linkage: you can keep the per-token compute fixed while growing the parameter count by an order of magnitude.</span>

* <span style="font-size: 14px;">**Parameter count grows with $E$** (number of experts), but **active parameters per token grow only with $k$** (experts per token).</span>
* <span style="font-size: 14px;">**Per-token FLOPs** are $k \cdot \text{FLOPs}(\text{single expert})$, independent of total $E$.</span>
* <span style="font-size: 14px;">**Memory bandwidth** dominates inference: only $k$ expert weight matrices need to be fetched per token, not all $E$.</span>

<span style="font-size: 14px;">GPT-OSS sets $E = 128$ and $k = 4$, so each token activates $4/128 = 3.1\%$ of the FFN parameters. The model behaves like a much wider dense model in capacity, while costing only the activated slice in FLOPs.</span>

---

## <span style="font-size: 16px;">The Forward Pass at a Glance</span>

<span style="font-size: 14px;">Given $N$ tokens with hidden states $x \in \mathbb{R}^{N \times H}$ (after RMSNorm), the GPT-OSS MoE block does:</span>

$$
g = x W_g + b_g \in \mathbb{R}^{N \times E}
$$

$$
\mathcal{I}_t = \text{top-}k(g_t), \quad w_t = \text{softmax}(g_{t, \mathcal{I}_t}) \in \mathbb{R}^{k}
$$

$$
y_t = \sum_{j=1}^{k} w_{t,j} \cdot \text{Expert}_{\mathcal{I}_{t,j}}(x_t)
$$

* <span style="font-size: 14px;">$W_g \in \mathbb{R}^{H \times E}$, $b_g \in \mathbb{R}^{E}$: the **router** (one linear layer).</span>
* <span style="font-size: 14px;">$\mathcal{I}_t$: the indices of the $k$ highest-scoring experts for token $t$.</span>
* <span style="font-size: 14px;">$w_t$: router weights, computed by softmax over **the $k$ selected logits only**.</span>

---

## <span style="font-size: 16px;">Routing: Top-k Then Softmax</span>

<span style="font-size: 14px;">Order matters. GPT-OSS computes top-$k$ on the **raw logits**, then applies softmax to **only those $k$ values**, so $\sum_j w_{t,j} = 1$ over the selected experts:</span>

$$
w_{t,j} = \frac{\exp(g_{t, \mathcal{I}_{t,j}})}{\sum_{j'=1}^{k} \exp(g_{t, \mathcal{I}_{t,j'}})}
$$

<span style="font-size: 14px;">The naive alternative (softmax over all $E$ logits first, then take top-$k$ of those probabilities) gives weights that do not sum to 1 and squashes the distribution toward uniform because all $E - k$ unused experts steal probability mass. The gpt-oss source uses the correct order: `experts = topk(g, k, sorted=True); expert_weights = softmax(experts.values, dim=1)`.</span>

---

## <span style="font-size: 16px;">The Expert: Clamped SwiGLU MLP</span>

<span style="font-size: 14px;">Each expert is a SwiGLU MLP with two GPT-OSS twists: **asymmetric clamping** on the pre-activations and a **$+1$ bias** on the linear (up) branch. The exact gpt-oss source is:</span>

$$
\text{swiglu}(x_{\text{glu}}, x_{\text{lin}}) = \big(x_{\text{glu}} \cdot \sigma(\alpha\, x_{\text{glu}})\big) \cdot (x_{\text{lin}} + 1)
$$

* <span style="font-size: 14px;">$\alpha = 1.702$: this is the "GELU-approx" sigmoid coefficient (Hendrycks and Gimpel 2016). It makes the activation close to GELU rather than plain SiLU.</span>
* <span style="font-size: 14px;">Gate branch $x_{\text{glu}}$ is clamped with `min=None, max=limit`: **only the upper tail is clipped**. Large positive pre-activations would otherwise saturate $\sigma(\alpha x)$ at 1 and push the output far from the trained regime.</span>
* <span style="font-size: 14px;">Linear branch $x_{\text{lin}}$ is clamped with `min=-limit, max=limit`: **both tails are clipped**. After adding 1, this gives a multiplier in $[1 - L,\, 1 + L]$.</span>
* <span style="font-size: 14px;">$+1$ bias: at $x_{\text{lin}} = 0$ the gated output passes through unchanged, so the linear branch acts as a multiplicative residual instead of a multiplicative gate.</span>

<span style="font-size: 14px;">With $L = 7.0$ (the default `swiglu_limit`), the clamps almost never fire during normal training. They are an MXFP4-stability guard: low-precision weights occasionally produce extreme pre-activations, and unclamped SwiGLU explodes when one branch goes to $\pm \infty$ while the other is finite.</span>

<span style="font-size: 14px;">In the production checkpoint the gate and linear pre-activations are interleaved inside a single $2I$-wide matrix `mlp1` and split as `x_glu = t[..., ::2]` and `x_linear = t[..., 1::2]`. For this problem the interleaved layout is decomposed into two clean matrices $W_1$ (gate) and $W_2$ (up), so each expert is:</span>

$$
a = x W_1 + b_1, \quad b = x W_2 + b_2
$$

$$
a \leftarrow \min(a, L), \quad b \leftarrow \text{clip}(b, -L, L)
$$

$$
\text{Expert}(x) = \big(a \cdot \sigma(\alpha a)\big) \odot (b + 1)\; W_3 + b_3
$$

---

## <span style="font-size: 16px;">Per-Expert Tensor Shapes</span>

<span style="font-size: 14px;">For $E$ experts with hidden size $H$ and intermediate size $I$:</span>

* <span style="font-size: 14px;">$W_1 \in \mathbb{R}^{E \times H \times I}$, $b_1 \in \mathbb{R}^{E \times I}$ (gate projection per expert).</span>
* <span style="font-size: 14px;">$W_2 \in \mathbb{R}^{E \times H \times I}$, $b_2 \in \mathbb{R}^{E \times I}$ (up projection per expert).</span>
* <span style="font-size: 14px;">$W_3 \in \mathbb{R}^{E \times I \times H}$, $b_3 \in \mathbb{R}^{E \times H}$ (down projection per expert).</span>

<span style="font-size: 14px;">In the gpt-oss source these are `mlp1_weight` ($E \times 2I \times H$, holding $W_1$ and $W_2$ interleaved), `mlp1_bias` ($E \times 2I$), `mlp2_weight` ($E \times H \times I$), `mlp2_bias` ($E \times H$). For GPT-OSS 20B/120B, $H = 2880$, $I = 2880$, so a single expert holds about 25M parameters and the full MoE bank holds $128 \cdot 25\text{M} = 3.2\text{B}$ parameters per block.</span>

---

## <span style="font-size: 16px;">Weighted Sum of Experts</span>

<span style="font-size: 14px;">After running the $k$ selected experts for token $t$, the block output is:</span>

$$
y_t = \sum_{j=1}^{k} w_{t,j} \cdot \text{Expert}_{\mathcal{I}_{t,j}}(x_t)
$$

<span style="font-size: 14px;">Two properties fall out:</span>

* <span style="font-size: 14px;">**Weights sum to 1.** $w_t$ is a softmax over $k$ values, so the weighted sum is a convex combination of expert outputs.</span>
* <span style="font-size: 14px;">**Gradients flow back through the router.** The router weight on the selected experts is multiplied into the output, so gradients reach $W_g$ even though the unselected experts are skipped.</span>

<span style="font-size: 14px;">The full block also adds the input residual: `return x + t`. This problem returns only $t$, the MoE-computed delta, so the caller can decide where to apply the residual.</span>

---

## <span style="font-size: 16px;">Worked Example ($N=1$, $H=2$, $I=2$, $E=2$, $k=1$)</span>

<span style="font-size: 14px;">Let $x = [0.2, -0.1]$, $W_g = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $b_g = [0, 0]$. Then $g = [0.2, -0.1]$, so $\mathcal{I} = \{0\}$, $w_0 = 1$ (softmax of a single value).</span>

<span style="font-size: 14px;">Take expert 0 with $W_1 = W_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, all biases zero, $W_3 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$. Then $a = b = [0.2, -0.1]$. With $L = 7$ neither clamp fires. Compute $\sigma(1.702 \cdot 0.2) \approx 0.584$ and $\sigma(1.702 \cdot -0.1) \approx 0.458$. Gate output is $[0.2 \cdot 0.584, -0.1 \cdot 0.458] = [0.117, -0.046]$. Multiply by $(b + 1) = [1.2, 0.9]$ element-wise: $[0.140, -0.041]$. Apply $W_3$ (identity) and the final $y = [0.140, -0.041]$, scaled by $w_0 = 1$.</span>

---

## <span style="font-size: 16px;">Comparison with Other MoE Designs</span>

* <span style="font-size: 14px;">**Mixtral 8x7B (Jiang et al. 2024):** $E = 8$, $k = 2$. Same top-$k$ then softmax routing. Activation is **plain SwiGLU** (SiLU times linear, no $+1$, no clamping). No MXFP4, so no stability hardening needed.</span>
* <span style="font-size: 14px;">**DeepSeek-V3 (DeepSeek-AI 2024):** $E = 256$ routed experts plus 1 shared expert. Routing uses **sigmoid scores** (not softmax) with a learned bias for load balancing. The shared expert runs on every token to capture global features that all experts would otherwise relearn. GPT-OSS keeps the simpler softmax-only routing without a shared expert.</span>
* <span style="font-size: 14px;">**Switch Transformer (Fedus et al. 2022):** $k = 1$ (hard routing). Cheap but less expressive. GPT-OSS's $k = 4$ trades some compute for smoother routing gradients and better quality.</span>
* <span style="font-size: 14px;">**LLaMA dense FFN:** non-clamped SwiGLU with $\sigma$ replaced by SiLU (no $\alpha$ scaling). Single FFN per block, no routing. Sets the baseline that MoE replaces.</span>

---

## <span style="font-size: 16px;">Why $\alpha = 1.702$ and the $+1$ Bias</span>

<span style="font-size: 14px;">The constant 1.702 makes $x \cdot \sigma(1.702 x)$ a very close approximation to GELU. Hendrycks and Gimpel (2016) showed that this matches the exact GELU shape (which involves $\Phi$, the standard normal CDF) within $10^{-4}$ across the active range. Using sigmoid is faster on hardware than the `erf` in the exact form.</span>

<span style="font-size: 14px;">The $+1$ on the linear branch turns the gate into a **multiplicative residual**. If the up branch were $x_{\text{lin}}$ alone, the down projection would receive zero whenever $x_{\text{lin}} = 0$, wasting the gate computation. With $(x_{\text{lin}} + 1)$, the gated value passes through at $x_{\text{lin}} = 0$ and is modulated up or down by the up branch around that baseline. This is conceptually similar to a GLU with a learned identity offset.</span>

---

## <span style="font-size: 16px;">Interaction with MXFP4</span>

<span style="font-size: 14px;">GPT-OSS ships expert weights in **MXFP4** (4-bit microscaled floating point), the topic of a separate problem (`gpto-mxfp4-dequant`). At inference, weights are dequantized to bf16 (or fp32) before the MoE forward pass. This problem assumes that dequantization is already done and inputs arrive as fp64 numpy arrays.</span>

* <span style="font-size: 14px;">**Why clamping is in the activation, not the matmul.** MXFP4 has range $\pm 6$ for the elements within a microscale block. Combined with a block scale, individual pre-activations can still drift to extreme values when many quantization errors align. Clamping inside SwiGLU absorbs those tails before they hit the gradient (training) or the next layer (inference).</span>
* <span style="font-size: 14px;">**Why $L = 7$.** It is just above the worst-case pre-activation magnitude observed in fp32 training runs of comparable models, so the clamp is essentially a no-op for in-distribution inputs and only fires for genuinely pathological values produced by MXFP4 rounding.</span>

---

## <span style="font-size: 16px;">Complexity</span>

* <span style="font-size: 14px;">**Router:** $O(N \cdot H \cdot E)$. Cheap relative to the experts because $E \ll I$.</span>
* <span style="font-size: 14px;">**Top-k:** $O(N \cdot E \cdot \log k)$ with a heap, or $O(N \cdot E)$ with a quickselect.</span>
* <span style="font-size: 14px;">**Experts:** $O(N \cdot k \cdot (H \cdot I + I \cdot H)) = O(N \cdot k \cdot H \cdot I)$. With $k = 4$, $H = I = 2880$, this is about $4 \cdot 16.6\text{M} = 66\text{M}$ FLOPs per token, identical to a dense FFN with $4 \cdot H$ width.</span>
* <span style="font-size: 14px;">**Memory:** the full bank is $E \cdot 3 \cdot H \cdot I$ parameters ($\sim 3.2\text{B}$ per block) but only $k$ expert matrices are touched per token at inference.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Softmax before top-k.** Computing softmax over all $E$ logits, then taking top-$k$ of the resulting probabilities, yields router weights that do not sum to 1 and are heavily flattened by the unused experts. The correct order is top-$k$ first, softmax over those $k$ values only. The gpt-oss source matches this exactly.</span>
* <span style="font-size: 14px;">**Forgetting the $+1$ on the linear branch.** Plain SwiGLU multiplies the gated value by $x_{\text{lin}}$ alone. Drop the $+1$ and outputs collapse toward zero whenever the up branch is near zero, which is a large fraction of activations. Behavior diverges from the GPT-OSS reference even on tiny inputs.</span>
* <span style="font-size: 14px;">**Using plain SiLU instead of $\alpha = 1.702$.** Plain SiLU is $x \cdot \sigma(x)$. GPT-OSS uses $x \cdot \sigma(1.702 x)$, which approximates GELU. The difference is small for $|x| < 1$ but compounds across layers and is detectable on most test inputs.</span>
* <span style="font-size: 14px;">**Symmetric clamp on the gate.** Gate clamping is one-sided: `min=None, max=limit`. Clamping the lower tail too suppresses the negative pre-activations that SwiGLU specifically uses to produce small negative outputs. Use `np.minimum(a, L)` for the gate, not `np.clip(a, -L, L)`.</span>
* <span style="font-size: 14px;">**Asymmetric clamp on the up branch.** Conversely, the linear (up) branch needs both tails clipped. Using only `min=...` or only `max=...` here leaves the explosive direction unprotected.</span>
* <span style="font-size: 14px;">**Hard-coded $k = 1$.** The router weight on a single expert is always 1.0, which hides whether the weighted-sum step is even running. Test with $k \geq 2$ to catch bugs in routing and weighting.</span>
* <span style="font-size: 14px;">**Reversed weight indexing.** `top_idx` is sorted descending by logit. The router weight for `top_idx[t, 0]` is the largest, for `top_idx[t, k-1]` is the smallest. Iterating with reversed weights silently produces a worse-but-finite output that may pass partial tests.</span>
* <span style="font-size: 14px;">**Unweighted accumulation.** Summing expert outputs without multiplying by $w_{t,j}$ inflates the output by roughly a factor of $k$ on average and breaks every test. The weighted sum is what makes MoE a smooth function of the router.</span>
* <span style="font-size: 14px;">**Treating experts as shared.** Each expert has its own $W_1, b_1, W_2, b_2, W_3, b_3$. Indexing the wrong slice (e.g. using expert index $j$ instead of $\mathcal{I}_{t,j}$) is a common bug when implementing MoE from scratch and produces output that looks plausible but is incorrect.</span>

---
