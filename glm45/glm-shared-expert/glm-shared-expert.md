# <span style="font-size: 20px;">Shared Expert with Routed Scaling</span>

<span style="font-size: 14px;">In GLM-4.5 (Zeng et al., 2025), every Mixture-of-Experts (MoE) layer contains one always-on **shared expert** that processes every token in addition to the sparsely activated routed experts. The shared output is summed with the routed mixture after a `routed_scaling_factor` has been applied to the routed weights, splitting the FFN computation into a dense always-on path and a sparse conditional path.</span>

---

## <span style="font-size: 16px;">What the Shared Expert Is</span>

<span style="font-size: 14px;">A conventional MoE layer replaces the dense FFN with $N$ parallel "experts". A learned router selects $k$ experts per token (top-$k$) and combines their outputs. Tokens that activate disjoint subsets of experts receive completely different transformations.</span>

<span style="font-size: 14px;">A **shared expert** is one additional FFN that does not participate in routing. It runs unconditionally on every token. The MoE layer's output for token $t$ is:</span>

$$
y_t = \text{SharedExpert}(x_t) + \sum_{j=1}^{k} w_{t,j} \cdot \text{RoutedExpert}_{i_{t,j}}(x_t)
$$

<span style="font-size: 14px;">where $i_{t,j}$ is the index of the $j$-th selected expert for token $t$ and $w_{t,j}$ is its (post-scaling) routing weight.</span>

---

## <span style="font-size: 16px;">Why Have a Shared Expert</span>

* <span style="font-size: 14px;">**Common knowledge is wasted on routing.** Every token in natural language needs basic linguistic processing (morphology, common syntactic patterns, frequent tokens). A pure top-$k$ MoE has to relearn this knowledge inside many experts, because any given expert only sees a fraction of the data.</span>
* <span style="font-size: 14px;">**Capacity that is always on.** The shared expert provides a guaranteed compute floor per token. Even if the router makes a poor decision, every token still passes through one well-trained FFN.</span>
* <span style="font-size: 14px;">**Specialization for the routed experts.** Once common knowledge lives in the shared expert, the routed experts are free to specialize on rarer or domain-specific patterns. This is the same argument DeepSeekMoE (Dai et al., 2024) and DeepSeek-V3 (DeepSeek-AI, 2024) use to motivate their shared experts.</span>
* <span style="font-size: 14px;">**Gradient signal stability.** Routed experts receive sparse gradients (only when chosen). The shared expert receives a full gradient on every step, which keeps the always-on path well-conditioned and helps overall convergence.</span>

---

## <span style="font-size: 16px;">Shared Expert Architecture (SwiGLU)</span>

<span style="font-size: 14px;">GLM-4.5 implements the shared expert as a SwiGLU MLP, identical in form to the routed experts but with a wider intermediate dimension. With $H$ = hidden size, $I$ = `moe_intermediate_size`, and $n_s$ = `n_shared_experts`:</span>

* <span style="font-size: 14px;">**Gate projection:** $W_g \in \mathbb{R}^{H \times (I \cdot n_s)}$</span>
* <span style="font-size: 14px;">**Up projection:** $W_u \in \mathbb{R}^{H \times (I \cdot n_s)}$</span>
* <span style="font-size: 14px;">**Down projection:** $W_d \in \mathbb{R}^{(I \cdot n_s) \times H}$</span>

<span style="font-size: 14px;">The forward pass:</span>

$$
\text{SharedExpert}(x) = \big(\,\text{SiLU}(xW_g) \odot (xW_u)\,\big)\, W_d
$$

<span style="font-size: 14px;">Here $\text{SiLU}(z) = z \cdot \sigma(z)$ is the swish/silu activation. The product $\text{SiLU}(xW_g) \odot (xW_u)$ is the "gated" part of SwiGLU (Shazeer 2020): one branch supplies the activated gate, the other supplies the linear up-projection, and $W_d$ projects back to model dimension. In GLM-4.5 with $n_s = 1$, the shared expert is exactly one SwiGLU FFN with intermediate width $I$.</span>

---

## <span style="font-size: 16px;">Routed Mixture and the Scaling Factor</span>

<span style="font-size: 14px;">For the routed half, the router emits a score vector over all $N$ experts. The top-$k$ experts are selected per token and their scores are normalized (often via sigmoid plus renormalization in GLM-4.5) to produce per-token weights $\tilde{w}_{t,j}$. These weights are then multiplied by a constant `routed_scaling_factor` $s$:</span>

$$
w_{t,j} = s \cdot \tilde{w}_{t,j}
$$

<span style="font-size: 14px;">The scaling factor compensates for the sparse-vs-dense gap: a dense FFN sees full activation at every position, but the routed mixture only contributes through $k$ out of $N$ experts. Multiplying the routing weights by $s > 1$ keeps the variance of the routed output close to what a comparable dense FFN would produce, which stabilizes training and avoids the routed branch being effectively scaled down.</span>

<span style="font-size: 14px;">**Crucial detail:** the scaling factor is applied **only to the routed weights**, never to the shared output. The shared expert is already dense-on-every-token, so it does not need any compensation. In this problem the caller has already folded $s$ into `top_weights`, so the implementation simply does $\sum_j w_{t,j} \cdot \text{RoutedExpert}_{i_{t,j}}(x_t)$ and then adds the unscaled shared output.</span>

---

## <span style="font-size: 16px;">Step by Step</span>

<span style="font-size: 14px;">1. **Compute the shared expert output for all tokens at once.** Apply $\text{SiLU}(xW_g) \odot (xW_u)$ then multiply by $W_d$. Shape: $(n,H) \to (n, I n_s) \to (n, H)$.</span>

<span style="font-size: 14px;">2. **Initialize the routed accumulator** as a tensor of zeros with shape $(n, H)$.</span>

<span style="font-size: 14px;">3. **For each token $t$ and each of its top-$k$ slots $j$**, look up the expert index $e = \text{top\_indices}[t,j]$, pull the per-expert weights $W^{(e)}_g, W^{(e)}_u, W^{(e)}_d$, and compute the routed expert output on the single-token row $x_t$.</span>

<span style="font-size: 14px;">4. **Weight and accumulate:** add $w_{t,j} \cdot \text{RoutedExpert}_e(x_t)$ into row $t$ of the accumulator. The scalar $w_{t,j}$ already includes the routed scaling factor.</span>

<span style="font-size: 14px;">5. **Sum** the shared output and the routed accumulator and return the result.</span>

---

## <span style="font-size: 16px;">Worked Example (small numbers)</span>

<span style="font-size: 14px;">Let $H=2$, $I=2$, $n_s=1$, $N=2$, $k=1$, one token $x = [0.5, -0.2]$. Suppose the shared SwiGLU output is $[0.04, 0.07]$ and the only chosen expert produces $[0.10, -0.05]$ with weight $w = 0.8$.</span>

* <span style="font-size: 14px;">**Routed contribution:** $0.8 \cdot [0.10, -0.05] = [0.08, -0.04]$</span>
* <span style="font-size: 14px;">**Final output:** $[0.04, 0.07] + [0.08, -0.04] = [0.12, 0.03]$</span>

<span style="font-size: 14px;">If `routed_scaling_factor` is $s = 2.5$ and the raw router weight is $\tilde{w} = 0.32$, then $w = s \cdot \tilde{w} = 0.8$, which is exactly the weight used above. The shared $[0.04, 0.07]$ never gets multiplied by $s$: only the routed branch does.</span>

---

## <span style="font-size: 16px;">Comparison with Other MoE Designs</span>

* <span style="font-size: 14px;">**Switch Transformer (Fedus et al., 2021):** $k=1$, no shared expert. Every token is routed to exactly one expert. Simple, but suffers from expert collapse and high variance for "common" tokens that have no dense fallback.</span>
* <span style="font-size: 14px;">**Mixtral 8x7B (Jiang et al., 2024):** $k=2$ from 8 experts, no shared expert. Higher quality than Switch but still no always-on path.</span>
* <span style="font-size: 14px;">**gpt-oss:** routed MoE without a shared expert. The router and capacity factor must do all of the load balancing.</span>
* <span style="font-size: 14px;">**DeepSeekMoE / DeepSeek-V3:** shared expert + many small routed experts. Direct precedent for GLM-4.5's design.</span>
* <span style="font-size: 14px;">**GLM-4.5:** one shared expert (`n_shared_experts = 1`) plus fine-grained routed experts, with a `routed_scaling_factor` applied to the routed weights. This combination concentrates common-knowledge capacity into the shared path and leaves the routed path free to specialize.</span>

---

## <span style="font-size: 16px;">Implementation Notes</span>

* <span style="font-size: 14px;">**Single matmul for the shared branch.** Because the shared expert runs on every token, you can batch the entire $(n,H)$ input through one SwiGLU MLP. No looping. This is significantly cheaper than the routed branch.</span>
* <span style="font-size: 14px;">**Routed branch is per-token scatter.** Different tokens hit different experts, so naive implementations loop over $(t, j)$. Production code uses index-sorted batching: group tokens by expert id, run one matmul per expert on its assigned tokens, then scatter-add back with the routing weights. The loop-form here is the readable reference.</span>
* <span style="font-size: 14px;">**Activation must be SiLU (not ReLU, not GELU).** GLM-4.5 follows the SwiGLU convention. SiLU on the gate, linear on the up branch, element-wise product, then down projection.</span>
* <span style="font-size: 14px;">**Per-expert weight slicing.** `routed_W_gate[e]` has shape $(H, I)$ and is the gate matrix for expert $e$. The order of the leading axis follows the router's expert index. Mis-ordering breaks training silently because the routing scores no longer correspond to the chosen FFN.</span>
* <span style="font-size: 14px;">**top_weights are already scaled.** In the GLM-4.5 reference, the post-softmax routing scores are multiplied by `routed_scaling_factor` before being passed into the expert combine. Treat them as opaque per-slot scalars: do not renormalize them inside the expert layer.</span>

---

## <span style="font-size: 16px;">Variants and Tunables</span>

* <span style="font-size: 14px;">**`n_shared_experts`.** GLM-4.5 sets this to 1. DeepSeekMoE-style configurations sometimes use 2 (the shared MLP is effectively twice as wide). The implementation only changes the intermediate width: $I \cdot n_s$ instead of $I$.</span>
* <span style="font-size: 14px;">**`routed_scaling_factor`.** A scalar (often around 2.5 in DeepSeek-V3 and similar in GLM-4.5). Compensates the variance of the routed branch. Setting it to 1.0 is equivalent to "no scaling" and is sometimes used for ablations.</span>
* <span style="font-size: 14px;">**Routing normalization.** GLM-4.5 uses sigmoid-based gating followed by selection of top-$k$ experts. The selected scores are renormalized to sum to 1 before the scaling factor is applied. The implementation of this problem treats the final scaled weights as inputs and does not recompute them.</span>
* <span style="font-size: 14px;">**Auxiliary loss / loss-free balancing.** GLM-4.5 uses an aux-loss-free routing strategy (per-expert bias updates) instead of the traditional auxiliary load-balance loss. This affects the router, not the combine step implemented here.</span>

---

## <span style="font-size: 16px;">Numerical and Performance Considerations</span>

* <span style="font-size: 14px;">**Numerical magnitudes.** SwiGLU outputs can be large in absolute value because the gated product is unbounded. The down projection $W_d$ usually has a smaller initialization scale (typically scaled by $1/\sqrt{I}$) to keep activations bounded.</span>
* <span style="font-size: 14px;">**fp16/bf16 routing weights.** Multiplying low-precision routing weights into fp16 expert outputs can underflow when $w$ is small. Production code keeps the combine step in fp32 and casts only the final accumulator back to the model dtype.</span>
* <span style="font-size: 14px;">**Memory.** Routed weights are $(N, H, I)$ and $(N, I, H)$, which is the bulk of an MoE layer's parameter count. The shared expert adds only $O(H \cdot I \cdot n_s)$, a tiny fraction of the routed cost.</span>
* <span style="font-size: 14px;">**FLOP budget.** Per token the shared expert costs $2 H I n_s$ FLOPs for each of the two up-projections plus $2 I n_s H$ for the down projection, while the routed branch costs $k$ times that per token (one full expert pass per selected slot). With $n_s = 1$, $k = 8$, the routed branch dominates the per-token compute by roughly $8\times$.</span>
* <span style="font-size: 14px;">**Effective parameter count.** GLM-4.5 reports 355B total parameters with 32B activated per token. The shared expert contributes to the activated count on every token, while routed experts only count toward activation when chosen, which is how an MoE keeps inference compute close to a dense model's while pushing total capacity much higher.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the shared branch.** Returning only $\sum_j w_{t,j} \cdot \text{RoutedExpert}(x_t)$ produces a syntactically valid MoE output but loses the entire always-on path. The model would lose its dense capacity floor and training would diverge or converge to a much weaker checkpoint.</span>
* <span style="font-size: 14px;">**Forgetting the routed branch.** Returning only the shared output throws away the entire sparse mixture. The model collapses to a single dense FFN with no expert specialization.</span>
* <span style="font-size: 14px;">**Wrong activation in the shared expert.** Using ReLU or GELU instead of SiLU in the shared SwiGLU silently changes the function. ReLU zeros out the negative half of the gate, which removes the smooth gating behavior that SwiGLU relies on.</span>
* <span style="font-size: 14px;">**Activation on the wrong branch.** Applying SiLU to the up branch instead of the gate, or to both, breaks the GLU structure. The convention is $\text{SiLU}(xW_g) \odot (xW_u)$ with the activation strictly on the gate.</span>
* <span style="font-size: 14px;">**Scaling the shared expert by `routed_scaling_factor`.** The scaling factor is for the routed weights only. Multiplying the shared output by it inflates the dense path beyond what training expects.</span>
* <span style="font-size: 14px;">**Wrong intermediate dimension for the shared MLP.** When `n_shared_experts > 1`, the shared intermediate is $I \cdot n_s$, not $I$. Forgetting the multiplication produces shape errors or, if the dimensions accidentally line up, a silently undersized FFN.</span>
* <span style="font-size: 14px;">**Transposed weight matrices.** Many reference implementations store weights as `(out_dim, in_dim)` and apply them via `x @ W.T`. The convention in this problem is the matmul-ready layout `(in_dim, out_dim)` so that `x @ W_gate` works directly. Confusing the two transposes the entire transformation.</span>
* <span style="font-size: 14px;">**Renormalizing `top_weights` inside the combine step.** The weights already include the scaling factor. Renormalizing them to sum to 1 strips out $s$ and pushes the routed branch back to dense-equivalent magnitude, undoing the variance compensation.</span>
* <span style="font-size: 14px;">**Indexing routed experts in the wrong order.** `routed_W_gate[e]` must match the expert id emitted by the router. Re-sorting `top_indices` or pre-sorting expert weights without sorting the other arrays correspondingly produces a quietly miscombined output.</span>
* <span style="font-size: 14px;">**Treating the shared expert like a routed expert.** Putting the shared expert inside the routed expert array and selecting it via `top_indices` is a common refactor mistake. The shared expert must always run on every token regardless of routing, and its output must not be multiplied by any routing weight.</span>

---
