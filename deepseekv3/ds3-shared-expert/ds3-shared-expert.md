# <span style="font-size: 20px;">Shared Expert Mechanism</span>

<span style="font-size: 14px;">In Mixture-of-Experts (MoE) transformer models, a router network decides which expert feedforward networks (FFNs) process each token. Most experts are "routed" -- they only see tokens the router selects. A shared expert is fundamentally different: it is an always-on FFN that processes every token unconditionally. Its output is added to the combined routed expert output, providing a baseline representation that every token benefits from.</span>

<span style="font-size: 14px;">DeepSeek V3 uses 1 shared expert alongside 256 routed experts per MoE layer. The shared expert uses SwiGLU, the same architecture as each routed expert but with a distinct role: capturing common, general-purpose knowledge that all tokens need while routed experts specialize.</span>

---

## <span style="font-size: 16px;">What It Is: An Always-On FFN</span>

<span style="font-size: 14px;">A standard MoE layer routes each token to a small subset of experts (e.g., 8 out of 256). This is efficient but introduces a problem: each token's representation quality depends entirely on the router making good choices.</span>

<span style="font-size: 14px;">A shared expert removes this dependency. It is a SwiGLU feedforward network with its own learned weight matrices, but it is never gated or routed. Every token, in every forward pass, flows through the shared expert. The final output of a DeepSeek V3 MoE layer for a token x is:</span>

$$
\text{output} = \text{SharedExpert}(x) + \sum_{i \in \text{TopK}} g_i \cdot \text{RoutedExpert}_i(x)
$$

<span style="font-size: 14px;">Here, TopK is the set of routed experts selected by the router, and g_i is the gating weight for each selected expert. The shared expert output is added directly -- no gating weight, no routing decision, no conditional logic.</span>

<span style="font-size: 14px;">The shared expert provides a "floor" of representation quality. Routed experts add specialized refinements on top. Even if the router makes a poor selection, the shared expert ensures the token gets reasonable processing.</span>

---

## <span style="font-size: 16px;">Key Equations: SwiGLU Feedforward Network</span>

<span style="font-size: 14px;">The shared expert (and each routed expert) uses SwiGLU, a gated linear unit where the gating function is swish (SiLU). It involves three weight matrices: W_gate, W_up, and W_down.</span>

<span style="font-size: 14px;">**Step 1: Gate projection.** The input x is projected through W_gate, then swish is applied:</span>

$$
\text{gate} = \text{swish}(x \cdot W_{\text{gate}}^T)
$$

<span style="font-size: 14px;">The swish function is defined as swish(z) = z * sigmoid(z). It is a smooth, non-monotonic activation that allows small negative values to pass through, unlike ReLU which zeros them entirely.</span>

<span style="font-size: 14px;">**Step 2: Up projection.** The input x is projected through W_up with no activation function:</span>

$$
\text{up} = x \cdot W_{\text{up}}^T
$$

<span style="font-size: 14px;">This is a simple linear transformation. The up projection provides the "content" that the gate will selectively allow through.</span>

<span style="font-size: 14px;">**Step 3: Element-wise gating.** The gate and up projections are multiplied element-wise:</span>

$$
\text{hidden} = \text{gate} \odot \text{up}
$$

<span style="font-size: 14px;">Each dimension of the up projection is scaled by the corresponding gate value. Dimensions where the gate is near zero get suppressed; dimensions where the gate is large get amplified.</span>

<span style="font-size: 14px;">**Step 4: Down projection.** The gated hidden state is projected back to the model dimension:</span>

$$
\text{output} = \text{hidden} \cdot W_{\text{down}}^T
$$

<span style="font-size: 14px;">**Full SwiGLU in one expression:**</span>

$$
\text{SwiGLU}(x) = \bigl(\text{swish}(x W_{\text{gate}}^T) \odot (x W_{\text{up}}^T)\bigr) W_{\text{down}}^T
$$

<span style="font-size: 14px;">If the model dimension is d_model and the intermediate dimension is d_ff, then W_gate and W_up are each d_ff x d_model matrices, and W_down is d_model x d_ff. SwiGLU has 3 weight matrices instead of the usual 2 in a standard ReLU FFN, so d_ff is typically set to 2/3 of what a ReLU FFN would use to keep parameter count comparable.</span>

---

## <span style="font-size: 16px;">Why a Shared Expert Matters</span>

<span style="font-size: 14px;">The shared expert exists to solve three concrete problems that pure routed-only MoE architectures suffer from.</span>

<span style="font-size: 14px;">**Preventing representation collapse from router failures.** Routers can collapse to favor a few experts while starving others, and early in training they make essentially random choices. In a purely routed MoE, a bad routing decision means a bad representation with no fallback. The shared expert provides that fallback: even with terrible routing, the token still receives a competent transformation.</span>

<span style="font-size: 14px;">**Ensuring common knowledge is always available.** Language models need certain capabilities for nearly every token: syntax processing, common word relationships, general feature extraction. In a routed-only MoE, this knowledge must be redundantly learned by every expert, wasting capacity. The shared expert centralizes it, freeing routed experts to specialize aggressively.</span>

<span style="font-size: 14px;">**Handling tokens that do not fit any specialist.** Function words like "the", punctuation, and structural tokens do not need domain-specific processing. In a routed-only MoE, these are forced through specialists anyway. The shared expert handles them naturally, providing generic processing while routed expert contributions are a bonus.</span>

---

## <span style="font-size: 16px;">SwiGLU Activation in Depth</span>

<span style="font-size: 14px;">Understanding why DeepSeek V3 chose SwiGLU over simpler activations like ReLU requires examining what swish and gating each contribute.</span>

<span style="font-size: 14px;">**Swish: the smooth ReLU.** The swish function is:</span>

$$
\text{swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

<span style="font-size: 14px;">For large positive z, swish(z) approaches z (identity). For large negative z, swish(z) approaches 0. Near z = 0, swish is smooth and differentiable, unlike ReLU's sharp kink. Crucially, swish is non-monotonic: it dips slightly below zero for small negative inputs (minimum around -0.278 at z approximately -1.28). This allows the network to propagate some information from mildly negative pre-activations rather than killing them entirely.</span>

<span style="font-size: 14px;">**Why SwiGLU outperforms ReLU FFN.** A standard ReLU FFN computes ReLU(xW_1^T)W_2^T. SwiGLU replaces this with a gated mechanism using three matrices. The gating lets the network learn to selectively pass information, which is more expressive than fixed thresholding. Swish's smoothness also provides better gradient flow -- no dead neurons from ReLU's zero-gradient region. Shazeer's 2020 paper showed SwiGLU consistently outperforms ReLU, GELU, and other variants across benchmarks.</span>

<span style="font-size: 14px;">**The gating mechanism.** The key insight of GLU variants is splitting computation into two parallel paths: one that determines "what to say" (up projection) and one that determines "how much to say" (gate projection with activation). The element-wise product merges these decisions, analogous to LSTM gates but within a single feedforward layer.</span>

---

## <span style="font-size: 16px;">Paper Context: DeepSeek V3 Architecture</span>

<span style="font-size: 14px;">DeepSeek V3 is a 671-billion-parameter MoE language model that uses 1 shared expert plus 256 routed experts per MoE layer. Each token activates the shared expert plus 8 routed experts (top-8 routing), so only about 37 billion parameters are active per token despite the massive total parameter count.</span>

<span style="font-size: 14px;">**How shared and routed outputs combine.** The output of each MoE layer is the sum of the shared expert's output and the weighted sum of routed expert outputs:</span>

$$
y = \text{SharedExpert}(x) + \sum_{i=1}^{8} g_i \cdot \text{Expert}_{s_i}(x)
$$

<span style="font-size: 14px;">Here s_1 through s_8 are the top-8 expert indices and g_i their gating weights. The shared expert has no gating weight; its full output is always included.</span>

<span style="font-size: 14px;">**Relation to dense prefix layers.** DeepSeek V3 does not use MoE in every layer. The first few layers are dense (standard FFN, no routing). The shared expert complements this: it ensures every token retains a dense-style computation path even in MoE layers. Together, the dense prefix and shared expert create a backbone of unconditional processing.</span>

<span style="font-size: 14px;">**Efficiency.** Adding 1 shared expert on top of 8 routed means each token passes through 9 FFNs instead of 8, a 12.5% increase in per-token FLOPs. This cost is worthwhile because the shared expert improves training stability and model quality, especially on tasks requiring broad general knowledge.</span>

<span style="font-size: 14px;">**Evolution from DeepSeek-MoE.** The concept was introduced in the DeepSeek-MoE paper (Dai et al., 2024), which proposed K_s shared experts alongside K_r routed experts. The initial design used 2 shared with 64 routed. V3 found a single larger shared expert outperforms multiple smaller ones and scaled to 1 shared + 256 routed.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a tiny shared expert with d_model = 3 and d_ff = 4. The input token is:</span>

$$
x = [1.0, \ -0.5, \ 0.8]
$$

<span style="font-size: 14px;">The weight matrices (shown transposed):</span>

$$
W_{\text{gate}}^T = \begin{bmatrix} 0.2 & -0.3 & 0.5 & 0.1 \\ 0.4 & 0.1 & -0.2 & 0.3 \\ -0.1 & 0.6 & 0.3 & -0.4 \end{bmatrix}
$$

$$
W_{\text{up}}^T = \begin{bmatrix} 0.3 & 0.1 & -0.4 & 0.2 \\ -0.2 & 0.5 & 0.1 & -0.3 \\ 0.4 & -0.2 & 0.6 & 0.1 \end{bmatrix}
$$

<span style="font-size: 14px;">**Step 1: Gate projection.** Compute x * W_gate^T:</span>

* <span style="font-size: 14px;">Dim 0: 1.0*0.2 + (-0.5)*0.4 + 0.8*(-0.1) = 0.2 - 0.2 - 0.08 = -0.08</span>
* <span style="font-size: 14px;">Dim 1: 1.0*(-0.3) + (-0.5)*0.1 + 0.8*0.6 = -0.3 - 0.05 + 0.48 = 0.13</span>
* <span style="font-size: 14px;">Dim 2: 1.0*0.5 + (-0.5)*(-0.2) + 0.8*0.3 = 0.5 + 0.1 + 0.24 = 0.84</span>
* <span style="font-size: 14px;">Dim 3: 1.0*0.1 + (-0.5)*0.3 + 0.8*(-0.4) = 0.1 - 0.15 - 0.32 = -0.37</span>

<span style="font-size: 14px;">Raw gate = [-0.08, 0.13, 0.84, -0.37]. Apply swish(z) = z * sigmoid(z):</span>

* <span style="font-size: 14px;">swish(-0.08) = -0.08 * 0.480 = -0.038</span>
* <span style="font-size: 14px;">swish(0.13) = 0.13 * 0.532 = 0.069</span>
* <span style="font-size: 14px;">swish(0.84) = 0.84 * 0.698 = 0.587</span>
* <span style="font-size: 14px;">swish(-0.37) = -0.37 * 0.409 = -0.151</span>

<span style="font-size: 14px;">gate = [-0.038, 0.069, 0.587, -0.151]</span>

<span style="font-size: 14px;">**Step 2: Up projection.** Compute x * W_up^T:</span>

* <span style="font-size: 14px;">Dim 0: 1.0*0.3 + (-0.5)*(-0.2) + 0.8*0.4 = 0.3 + 0.1 + 0.32 = 0.72</span>
* <span style="font-size: 14px;">Dim 1: 1.0*0.1 + (-0.5)*0.5 + 0.8*(-0.2) = 0.1 - 0.25 - 0.16 = -0.31</span>
* <span style="font-size: 14px;">Dim 2: 1.0*(-0.4) + (-0.5)*0.1 + 0.8*0.6 = -0.4 - 0.05 + 0.48 = 0.03</span>
* <span style="font-size: 14px;">Dim 3: 1.0*0.2 + (-0.5)*(-0.3) + 0.8*0.1 = 0.2 + 0.15 + 0.08 = 0.43</span>

<span style="font-size: 14px;">up = [0.72, -0.31, 0.03, 0.43]</span>

<span style="font-size: 14px;">**Step 3: Element-wise product.** hidden = gate * up:</span>

* <span style="font-size: 14px;">hidden = [-0.038*0.72, 0.069*(-0.31), 0.587*0.03, -0.151*0.43]</span>
* <span style="font-size: 14px;">hidden = [-0.027, -0.021, 0.018, -0.065]</span>

<span style="font-size: 14px;">**Step 4: Down projection.** W_down^T maps d_ff=4 back to d_model=3:</span>

$$
W_{\text{down}}^T = \begin{bmatrix} 0.5 & -0.2 & 0.3 \\ 0.1 & 0.4 & -0.1 \\ -0.3 & 0.2 & 0.6 \\ 0.2 & -0.5 & 0.1 \end{bmatrix}
$$

* <span style="font-size: 14px;">output[0] = -0.027*0.5 + (-0.021)*0.1 + 0.018*(-0.3) + (-0.065)*0.2 = -0.034</span>
* <span style="font-size: 14px;">output[1] = -0.027*(-0.2) + (-0.021)*0.4 + 0.018*0.2 + (-0.065)*(-0.5) = 0.034</span>
* <span style="font-size: 14px;">output[2] = -0.027*0.3 + (-0.021)*(-0.1) + 0.018*0.6 + (-0.065)*0.1 = -0.002</span>

<span style="font-size: 14px;">Shared expert output = [-0.034, 0.034, -0.002]. This is added directly to the routed output. If routed experts produced [0.5, -0.3, 0.2], the final MoE output is [0.466, -0.266, 0.198]. The shared expert's contribution is small but always present -- a stable baseline every token receives.</span>

---

## <span style="font-size: 16px;">Modern Context and Comparisons</span>

<span style="font-size: 14px;">**DeepSeek-MoE (2024): the origin.** The DeepSeek-MoE paper introduced shared experts, showing that without them, multiple routed experts converge to similar representations (redundancy). With shared experts absorbing common knowledge, routed experts diversified more effectively. The original design used 2 shared experts with 64 routed experts.</span>

<span style="font-size: 14px;">**DeepSeek V3: scaling up.** V3 simplified to 1 shared + 256 routed experts per layer, finding one large shared expert outperforms splitting the same budget across multiple smaller ones. V3's auxiliary-loss-free load balancing interacts favorably: because the shared expert handles common-knowledge tokens, the router has less pressure to distribute generic tokens evenly.</span>

<span style="font-size: 14px;">**Mixtral (2024): no shared expert.** Mixtral 8x7B uses pure routed MoE with 8 experts, top-2 routing, and no shared expert. Each expert must redundantly learn common transformations. Mixtral compensates with few large experts (8 vs 256), where each has enough capacity to be both general and specialized.</span>

<span style="font-size: 14px;">**Switch Transformer and GShard.** Earlier Google MoE designs used top-1 or top-2 routing without shared experts, addressing failures through auxiliary losses and token dropping. The shared expert is more principled: instead of penalizing bad routing after the fact, guarantee every token at least one competent FFN pass.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing the shared expert with a routed expert.** The shared expert is not selected by the router. It has no gating weight and processes every token unconditionally. A routed expert only processes tokens the router assigns to it. The shared expert is architecturally guaranteed to fire; routed experts are conditionally activated. Treating the shared expert as "just another expert the router picks" misses the entire point.</span>

* <span style="font-size: 14px;">**Getting SwiGLU's gate and up projections backwards.** Swish is applied to the gate projection, not the up projection. The correct order is: gate = swish(xW_gate^T), up = xW_up^T, hidden = gate * up. Applying swish to the up projection or after the element-wise product produces a network that still trains but with degraded performance. The gate controls information flow; the up provides information.</span>

* <span style="font-size: 14px;">**Forgetting that shared expert output is ADDED to routed output, not replacing it.** The final MoE output is shared_output + sum(routed_outputs). Both always contribute. Implementing it as either/or (use shared when routing fails, otherwise use routed) breaks the architecture. The outputs are summed unconditionally.</span>

* <span style="font-size: 14px;">**Assuming the shared expert learns the same thing as routed experts.** The shared expert sees every token, giving it a different gradient landscape. It converges to universally useful transformations while routed experts specialize. Initializing by copying a routed expert and freezing it prevents learning general representations from the full token distribution.</span>

* <span style="font-size: 14px;">**Ignoring the parameter cost.** The shared expert adds one full expert's worth of per-token computation. In DeepSeek V3, each token goes through 9 SwiGLU FFNs (1 shared + 8 routed) instead of 8, a 12.5% FLOPs increase that must be accounted for in efficiency comparisons.</span>

* <span style="font-size: 14px;">**Mixing up SwiGLU weight dimensions.** SwiGLU has three weight matrices, not two. W_gate and W_up are each d_ff x d_model; W_down is d_model x d_ff. The extra matrix means SwiGLU uses 50% more parameters for the same d_ff, which is why SwiGLU models use d_ff = (2/3) * d_ff_relu to maintain parameter parity.</span>

---