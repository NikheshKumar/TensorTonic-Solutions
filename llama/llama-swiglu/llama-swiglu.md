# <span style="font-size: 20px;">SwiGLU Feed-Forward Network</span>

<span style="font-size: 14px;">The SwiGLU FFN is a gated feed-forward network used in LLaMA (and PaLM, Mistral, Gemma) that replaces the standard two-matrix ReLU or GELU FFN with a three-matrix design. It follows $\text{SwiGLU}(x) = (\text{swish}(xW_{\text{gate}}^T) \odot xW_{\text{up}}^T) \cdot W_{\text{down}}^T$, where swish is the self-gated activation $\text{swish}(x) = x \cdot \sigma(x)$. Two parallel projections expand the input -- one gated through swish, the other passed through linearly -- and their element-wise product is projected back down. This gating mechanism gives the network learned control over which features flow forward, yielding consistent improvements over standard FFNs.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A standard Transformer FFN has two weight matrices: one expands the hidden dimension, applies an activation, and the other contracts it back. SwiGLU replaces this with a gated linear unit variant using three weight matrices and the swish activation:</span>

* <span style="font-size: 14px;">**Gate projection ($W_{\text{gate}}$):** maps $d_{\text{model}} \to d_{\text{intermediate}}$, then passes through swish. This branch learns *what to activate* -- producing a gate signal that controls information flow.</span>
* <span style="font-size: 14px;">**Up projection ($W_{\text{up}}$):** maps $d_{\text{model}} \to d_{\text{intermediate}}$ with no activation. This branch learns *what information to carry* -- the raw content to be gated.</span>
* <span style="font-size: 14px;">**Element-wise product ($\odot$):** the swish-activated gate and the linear up projection are multiplied element-wise. The gate selectively amplifies or suppresses each dimension of the up projection.</span>
* <span style="font-size: 14px;">**Down projection ($W_{\text{down}}$):** maps $d_{\text{intermediate}} \to d_{\text{model}}$, compressing the gated result for the residual connection.</span>

<span style="font-size: 14px;">The "GLU" stands for Gated Linear Unit. The original GLU (Dauphin et al., 2017) used a sigmoid gate: $\text{GLU}(x) = (xW_1) \odot \sigma(xW_2)$. SwiGLU replaces the sigmoid with swish, hence "Swish-Gated Linear Unit." In LLaMA, SwiGLU is the FFN sub-layer in every Transformer block, applied independently to each token position after attention.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The full SwiGLU computation for input $x \in \mathbb{R}^{d}$:</span>

$$
\text{SwiGLU}(x) = \bigl(\text{swish}(x W_{\text{gate}}^T) \odot x W_{\text{up}}^T\bigr) \cdot W_{\text{down}}^T
$$

<span style="font-size: 14px;">where swish (also called SiLU) is:</span>

$$
\text{swish}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
$$

<span style="font-size: 14px;">Breaking this into steps:</span>

<span style="font-size: 14px;">**Step 1 -- Gate projection.** Compute the gate signal:</span>

$$
g = \text{swish}(x W_{\text{gate}}^T), \quad W_{\text{gate}} \in \mathbb{R}^{d_{\text{ff}} \times d}, \quad g \in \mathbb{R}^{d_{\text{ff}}}
$$

<span style="font-size: 14px;">**Step 2 -- Up projection.** Compute the content signal (no activation):</span>

$$
u = x W_{\text{up}}^T, \quad W_{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d}, \quad u \in \mathbb{R}^{d_{\text{ff}}}
$$

<span style="font-size: 14px;">**Step 3 -- Element-wise gating.** Multiply gate and content element-wise:</span>

$$
h = g \odot u, \quad h \in \mathbb{R}^{d_{\text{ff}}}
$$

<span style="font-size: 14px;">**Step 4 -- Down projection.** Compress back to model dimension:</span>

$$
\text{SwiGLU}(x) = h \cdot W_{\text{down}}^T, \quad W_{\text{down}} \in \mathbb{R}^{d \times d_{\text{ff}}}, \quad \text{output} \in \mathbb{R}^{d}
$$

<span style="font-size: 14px;">LLaMA uses no bias terms in any projection. The output dimension matches the input, required for the residual connection $x + \text{SwiGLU}(\text{RMSNorm}(x))$.</span>

---

## <span style="font-size: 16px;">Why SwiGLU Over ReLU FFN</span>

<span style="font-size: 14px;">Shazeer (2020) systematically evaluated GLU variants -- replacing the standard $\text{FFN}(x) = \text{ReLU}(xW_1^T)W_2^T$ with gated alternatives -- and found that GLU-based FFNs consistently outperformed standard FFNs. SwiGLU was the best-performing variant, adopted by LLaMA, PaLM, Mistral, and Gemma.</span>

<span style="font-size: 14px;">The advantages come from the gating mechanism:</span>

* <span style="font-size: 14px;">**Learned feature selection.** In a standard FFN, the activation applies a fixed nonlinearity to each neuron. In SwiGLU, the gate projection learns an input-dependent mask -- not just "on or off" but a continuous signal tuned during training.</span>
* <span style="font-size: 14px;">**Greater expressiveness.** The element-wise product of two separate projections creates a multiplicative interaction, capturing second-order feature combinations that a single projection followed by a pointwise activation cannot.</span>
* <span style="font-size: 14px;">**Smoother optimization.** Swish is smooth everywhere (unlike ReLU's hard kink at zero) and non-monotonic. Gradients flow through all neurons, avoiding dead neurons at scale.</span>
* <span style="font-size: 14px;">**Empirical results.** On the same parameter budget, SwiGLU reduced perplexity compared to ReLU, GELU, and other GLU variants (ReGLU, GEGLU). The improvement was consistent across model sizes.</span>

---

## <span style="font-size: 16px;">The Three Projections</span>

<span style="font-size: 14px;">Each of SwiGLU's three matrices has a distinct role:</span>

<span style="font-size: 14px;">**$W_{\text{gate}}$ -- what to activate.** Produces the gating signal. After swish, each element controls how strongly the corresponding up-projection dimension is expressed. Negative gate values slightly suppress; values near zero strongly suppress; large positive values pass information through.</span>

<span style="font-size: 14px;">**$W_{\text{up}}$ -- what to pass.** Produces content to be gated. No activation function is applied -- it is an unconstrained linear transformation, the "data path" that the gate selectively allows through.</span>

<span style="font-size: 14px;">**$W_{\text{down}}$ -- compress back.** Takes the gated intermediate ($d_{\text{ff}}$ dimensions) and maps back to $d_{\text{model}}$. Analogous to $W_2$ in a standard FFN: the contraction layer producing output for the residual stream.</span>

<span style="font-size: 14px;">In a standard FFN, a single matrix handles both content and gating -- $W_1$ expands, the activation gates, $W_2$ contracts. SwiGLU separates these into two independent learned transformations. The gate does not need to preserve content fidelity; the up projection does not need to produce values suitable for a fixed activation. This separation is why GLU variants outperform standard FFNs despite comparable parameter counts.</span>

---

## <span style="font-size: 16px;">The Swish Activation</span>

<span style="font-size: 14px;">Swish (Ramachandran et al., 2017), also called SiLU, is defined as:</span>

$$
\text{swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

<span style="font-size: 14px;">Key properties:</span>

* <span style="font-size: 14px;">**Self-gated.** $x$ is multiplied by its own sigmoid. For large positive $x$, $\sigma(x) \approx 1$ so $\text{swish}(x) \approx x$. For large negative $x$, $\sigma(x) \approx 0$ so $\text{swish}(x) \approx 0$. This mirrors ReLU but with smooth transitions.</span>
* <span style="font-size: 14px;">**Smooth and differentiable.** Unlike ReLU's discontinuous derivative at $x = 0$, swish is infinitely differentiable everywhere, improving optimization stability in deep networks.</span>
* <span style="font-size: 14px;">**Non-monotonic.** Swish has a global minimum at $x \approx -1.28$ reaching approximately $-0.28$. Small negative inputs produce slightly negative outputs before being suppressed for more negative inputs.</span>
* <span style="font-size: 14px;">**Comparison with GELU.** GELU uses $x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF. Both are smooth, self-gated, and non-monotonic. In SwiGLU, swish only needs to produce good gate values, not serve double duty as both activation and implicit gate.</span>

---

## <span style="font-size: 16px;">Parameter Count</span>

<span style="font-size: 14px;">SwiGLU uses three weight matrices instead of two. To keep the parameter count roughly equal to a standard FFN, the intermediate dimension is reduced:</span>

* <span style="font-size: 14px;">**Standard FFN:** $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$. Total: $8d^2$.</span>

<span style="font-size: 14px;">For SwiGLU to match with three matrices:</span>

$$
3 \times d_{\text{ff}} \times d = 8d^2 \implies d_{\text{ff}} = \frac{8}{3}d \approx 2.667d
$$

<span style="font-size: 14px;">In practice, this is rounded to a hardware-friendly multiple. LLaMA's choices:</span>

* <span style="font-size: 14px;">**LLaMA-7B:** $d = 4096$, $d_{\text{ff}} = 11008$ (vs. $\frac{8}{3} \times 4096 = 10923$, rounded to multiple of 256)</span>
* <span style="font-size: 14px;">**LLaMA-13B:** $d = 5120$, $d_{\text{ff}} = 13824$</span>
* <span style="font-size: 14px;">**LLaMA-65B:** $d = 8192$, $d_{\text{ff}} = 22016$</span>

<span style="font-size: 14px;">**Per-block FFN parameters for LLaMA-7B (no biases):** $3 \times 11008 \times 4096 = 135{,}266{,}304$. Compare to a standard 4x FFN: $2 \times 16384 \times 4096 = 134{,}217{,}728$. Nearly identical -- the three-matrix design redistributes parameters into a more expressive architecture.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">SwiGLU originates from "GLU Variants Improve Transformer" (Shazeer, 2020), which tested ReGLU (ReLU gate), GEGLU (GELU gate), and SwiGLU (swish gate). SwiGLU delivered the best perplexity across text-to-text tasks.</span>

<span style="font-size: 14px;">LLaMA (Touvron et al., 2023) adopted SwiGLU for efficient, high-quality open-source language models, crediting Shazeer and PaLM. Other adopters:</span>

* <span style="font-size: 14px;">**PaLM (Chowdhery et al., 2022):** Google's 540B model used SwiGLU throughout.</span>
* <span style="font-size: 14px;">**Mistral (Jiang et al., 2023):** Same gated FFN design as LLaMA.</span>
* <span style="font-size: 14px;">**Gemma (Google, 2024):** Follows the SwiGLU pattern.</span>

<span style="font-size: 14px;">The core insight -- three smaller matrices in a gated configuration outperform two larger matrices with a fixed activation -- has made SwiGLU the default FFN in modern large language models.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Trace through SwiGLU with $d = 4$ and $d_{\text{ff}} = 6$.</span>

<span style="font-size: 14px;">**Input:** $x = [1.0, -0.5, 0.3, 0.8]$.</span>

<span style="font-size: 14px;">**Step 1 -- Gate projection.** Compute $xW_{\text{gate}}^T$ using $W_{\text{gate}} \in \mathbb{R}^{6 \times 4}$:</span>

$$
W_{\text{gate}} = \begin{pmatrix} 0.2 & -0.1 & 0.4 & 0.3 \\ -0.3 & 0.5 & 0.1 & -0.2 \\ 0.1 & 0.2 & -0.3 & 0.6 \\ 0.4 & -0.4 & 0.2 & 0.1 \\ -0.2 & 0.3 & 0.5 & -0.1 \\ 0.3 & 0.1 & -0.2 & 0.4 \end{pmatrix}
$$

* <span style="font-size: 14px;">$g_0 = (1.0)(0.2) + (-0.5)(-0.1) + (0.3)(0.4) + (0.8)(0.3) = 0.61$</span>
* <span style="font-size: 14px;">$g_1 = (1.0)(-0.3) + (-0.5)(0.5) + (0.3)(0.1) + (0.8)(-0.2) = -0.68$</span>
* <span style="font-size: 14px;">$g_2 = (1.0)(0.1) + (-0.5)(0.2) + (0.3)(-0.3) + (0.8)(0.6) = 0.39$</span>
* <span style="font-size: 14px;">$g_3 = (1.0)(0.4) + (-0.5)(-0.4) + (0.3)(0.2) + (0.8)(0.1) = 0.74$</span>
* <span style="font-size: 14px;">$g_4 = (1.0)(-0.2) + (-0.5)(0.3) + (0.3)(0.5) + (0.8)(-0.1) = -0.28$</span>
* <span style="font-size: 14px;">$g_5 = (1.0)(0.3) + (-0.5)(0.1) + (0.3)(-0.2) + (0.8)(0.4) = 0.51$</span>

<span style="font-size: 14px;">Pre-activation gate: $[0.61, -0.68, 0.39, 0.74, -0.28, 0.51]$. Apply $\text{swish}(x) = x \cdot \sigma(x)$:</span>

* <span style="font-size: 14px;">$\text{swish}(0.61) = 0.61 \cdot 0.648 = 0.395$</span>
* <span style="font-size: 14px;">$\text{swish}(-0.68) = -0.68 \cdot 0.336 = -0.228$</span>
* <span style="font-size: 14px;">$\text{swish}(0.39) = 0.39 \cdot 0.596 = 0.232$</span>
* <span style="font-size: 14px;">$\text{swish}(0.74) = 0.74 \cdot 0.677 = 0.501$</span>
* <span style="font-size: 14px;">$\text{swish}(-0.28) = -0.28 \cdot 0.430 = -0.120$</span>
* <span style="font-size: 14px;">$\text{swish}(0.51) = 0.51 \cdot 0.625 = 0.319$</span>

<span style="font-size: 14px;">Gate after swish: $g = [0.395, -0.228, 0.232, 0.501, -0.120, 0.319]$.</span>

<span style="font-size: 14px;">**Step 2 -- Up projection.** Compute $xW_{\text{up}}^T$ using $W_{\text{up}} \in \mathbb{R}^{6 \times 4}$:</span>

$$
W_{\text{up}} = \begin{pmatrix} 0.5 & 0.1 & -0.2 & 0.3 \\ -0.1 & 0.4 & 0.3 & -0.5 \\ 0.3 & -0.3 & 0.2 & 0.1 \\ -0.2 & 0.2 & 0.6 & 0.4 \\ 0.4 & -0.1 & -0.4 & 0.2 \\ 0.1 & 0.3 & 0.1 & -0.3 \end{pmatrix}
$$

* <span style="font-size: 14px;">$u_0 = (1.0)(0.5) + (-0.5)(0.1) + (0.3)(-0.2) + (0.8)(0.3) = 0.63$</span>
* <span style="font-size: 14px;">$u_1 = (1.0)(-0.1) + (-0.5)(0.4) + (0.3)(0.3) + (0.8)(-0.5) = -0.61$</span>
* <span style="font-size: 14px;">$u_2 = (1.0)(0.3) + (-0.5)(-0.3) + (0.3)(0.2) + (0.8)(0.1) = 0.59$</span>
* <span style="font-size: 14px;">$u_3 = (1.0)(-0.2) + (-0.5)(0.2) + (0.3)(0.6) + (0.8)(0.4) = 0.20$</span>
* <span style="font-size: 14px;">$u_4 = (1.0)(0.4) + (-0.5)(-0.1) + (0.3)(-0.4) + (0.8)(0.2) = 0.49$</span>
* <span style="font-size: 14px;">$u_5 = (1.0)(0.1) + (-0.5)(0.3) + (0.3)(0.1) + (0.8)(-0.3) = -0.26$</span>

<span style="font-size: 14px;">Up projection: $u = [0.63, -0.61, 0.59, 0.20, 0.49, -0.26]$.</span>

<span style="font-size: 14px;">**Step 3 -- Element-wise gating.** $h = g \odot u$:</span>

* <span style="font-size: 14px;">$h_0 = 0.395 \times 0.63 = 0.249$, $h_1 = -0.228 \times -0.61 = 0.139$, $h_2 = 0.232 \times 0.59 = 0.137$</span>
* <span style="font-size: 14px;">$h_3 = 0.501 \times 0.20 = 0.100$, $h_4 = -0.120 \times 0.49 = -0.059$, $h_5 = 0.319 \times -0.26 = -0.083$</span>

<span style="font-size: 14px;">Gated intermediate: $h = [0.249, 0.139, 0.137, 0.100, -0.059, -0.083]$.</span>

<span style="font-size: 14px;">**Step 4 -- Down projection.** $hW_{\text{down}}^T$ yields $\text{SwiGLU}(x) = [0.12, -0.08, 0.15, 0.06]$. Residual: $x + \text{SwiGLU}(x) = [1.12, -0.58, 0.45, 0.86]$.</span>

<span style="font-size: 14px;">Notice how gating produced mixed signs. Dimensions 0-3 had positive gates, preserving the up-projection signs. Dimensions 4-5 had negative gates from swish's non-monotonic region, flipping signs. This input-dependent, dimension-wise control is what makes SwiGLU more expressive than a fixed activation.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**1. Confusing gate and up projections.**</span>

<span style="font-size: 14px;">The gate projection has swish applied; the up projection does not. Swapping them changes the computation. The gate controls information flow; the up branch provides content. Getting this backwards produces incorrect outputs even if shapes are right.</span>

<span style="font-size: 14px;">**2. Using the wrong activation function.**</span>

<span style="font-size: 14px;">SwiGLU uses swish ($x \cdot \sigma(x)$), not ReLU, GELU, or plain sigmoid. Using $\sigma(xW_{\text{gate}}^T)$ gives standard GLU, not SwiGLU. Using GELU gives GEGLU. Shazeer's experiments showed measurable differences between variants.</span>

<span style="font-size: 14px;">**3. Forgetting the element-wise product.**</span>

<span style="font-size: 14px;">Gate and up projections must be combined via element-wise multiplication ($\odot$), not addition or concatenation. Addition defeats the gating mechanism. Concatenation doubles the intermediate dimension and requires a different $W_{\text{down}}$ shape.</span>

<span style="font-size: 14px;">**4. Wrong intermediate dimension.**</span>

<span style="font-size: 14px;">Because SwiGLU has three weight matrices instead of two, the intermediate dimension is $\frac{8}{3}d$ (not $4d$) to maintain a similar parameter count. Using $4d$ with three matrices inflates total parameters by 50%.</span>