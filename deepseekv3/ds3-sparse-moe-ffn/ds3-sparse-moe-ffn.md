# <span style="font-size: 20px;">Sparse MoE Feedforward Block</span>

<span style="font-size: 14px;">The Sparse Mixture-of-Experts (MoE) Feedforward Block is the complete feedforward layer used in DeepSeek V3, replacing the standard dense FFN found in conventional transformers. Instead of one monolithic feedforward network processing every token identically, it combines a shared expert that runs on all tokens with a bank of routed experts where only a small subset activates per token. The result is a layer with an enormous total parameter count but a controlled, predictable compute budget per token.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">In a standard transformer, the feedforward block is a single dense FFN applied uniformly to every token. Every parameter participates in every forward pass. A Sparse MoE FFN breaks this by maintaining many independent expert FFNs and routing each token to only a few of them.</span>

<span style="font-size: 14px;">DeepSeek V3's MoE FFN has two distinct components:</span>

* <span style="font-size: 14px;">**One shared expert:** A full SwiGLU FFN that processes every token unconditionally, capturing common knowledge the model needs regardless of content.</span>
* <span style="font-size: 14px;">**256 routed experts:** Independent SwiGLU FFNs, each with its own weights. A learned router selects exactly 8 of these 256 per token. Only those 8 run their forward pass, and their outputs are combined via learned weights.</span>

<span style="font-size: 14px;">The final output is the sum of the shared expert's output and the weighted combination of selected routed experts' outputs. This gives the model 257 expert FFNs worth of parameters but only ever runs 9 (1 shared + 8 routed) per token.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**Router logits.** Given token representation $x$ of shape $(d_{model},)$, the router computes a score for each of the $N_r$ routed experts:</span>

$$s_i = \text{sigmoid}(e_i^T \cdot x), \quad i = 1, \ldots, N_r$$

<span style="font-size: 14px;">where $e_i$ is the learned centroid vector for expert $i$. Each score is an independent probability in $[0, 1]$. DeepSeek V3 uses sigmoid routing rather than softmax, so expert scores are not coupled -- one expert scoring highly does not force another to score low.</span>

<span style="font-size: 14px;">**Top-K selection.** The $K$ experts with the highest scores are selected:</span>

$$\mathcal{T} = \text{TopK}(\{s_1, s_2, \ldots, s_{N_r}\},\; K)$$

<span style="font-size: 14px;">In DeepSeek V3, $N_r = 256$ and $K = 8$.</span>

<span style="font-size: 14px;">**Routing weights.** The selected scores are renormalized to produce combination weights:</span>

$$g_i = \frac{s_i}{\sum_{j \in \mathcal{T}} s_j}, \quad i \in \mathcal{T}$$

<span style="font-size: 14px;">This ensures the routed experts' contributions sum to a consistent scale regardless of raw score magnitudes.</span>

<span style="font-size: 14px;">**SwiGLU expert computation.** Each expert (shared and routed) is a SwiGLU FFN. For expert $i$ with weight matrices $W_{gate}^{(i)}$, $W_{up}^{(i)}$, $W_{down}^{(i)}$:</span>

$$\text{Expert}_i(x) = W_{down}^{(i)} \cdot \left[\text{swish}\!\left(W_{gate}^{(i)} \cdot x\right) \odot \left(W_{up}^{(i)} \cdot x\right)\right]$$

<span style="font-size: 14px;">where $\text{swish}(z) = z \cdot \sigma(z)$ and $\odot$ is elementwise multiplication. The gate and up projections produce two intermediate vectors; the gate path passes through swish, then they are multiplied elementwise, and the down projection maps back to $d_{model}$.</span>

<span style="font-size: 14px;">**Routed output.** The combined output of the routed experts:</span>

$$y_{routed} = \sum_{i \in \mathcal{T}} g_i \cdot \text{Expert}_i(x)$$

<span style="font-size: 14px;">**Shared expert output.** The shared expert runs unconditionally:</span>

$$y_{shared} = \text{Expert}_{shared}(x)$$

<span style="font-size: 14px;">**Final MoE FFN output.** The complete layer output adds both:</span>

$$\text{MoE\text{-}FFN}(x) = y_{shared} + y_{routed}$$

---

## <span style="font-size: 16px;">The Four-Step Pipeline</span>

<span style="font-size: 14px;">The forward pass follows a four-step sequence for each token.</span>

<span style="font-size: 14px;">**Step 1: Route.** The token $x$ is fed into the router, which computes sigmoid scores against all 256 routed expert centroids. The top-8 are selected and renormalized to produce combination weights $g_i$ summing to 1. This determines which experts fire and with what weight. The decision is per-token -- different tokens in the same batch may activate completely different expert subsets.</span>

<span style="font-size: 14px;">**Step 2: Shared expert.** The token passes through the shared expert's SwiGLU FFN unconditionally. The shared expert captures foundational transformations that benefit all tokens, such as common syntactic patterns or baseline feature extraction. Its output $y_{shared}$ is stored.</span>

<span style="font-size: 14px;">**Step 3: Routed experts.** The token passes through each of the 8 selected routed experts. Each runs its own SwiGLU computation with its own $W_{gate}$, $W_{up}$, $W_{down}$. This produces 8 output vectors. The remaining 248 experts are not computed at all.</span>

<span style="font-size: 14px;">**Step 4: Combine.** The 8 routed outputs are multiplied by their routing weights $g_i$ and summed to produce $y_{routed}$. This is added to $y_{shared}$ to produce the final MoE FFN output, replacing what a standard dense FFN would have produced.</span>

---

## <span style="font-size: 16px;">Why Sparse</span>

<span style="font-size: 14px;">"Sparse" refers to the activation pattern, not the weight matrices. Every expert has fully dense weights. The sparsity is that only a small fraction of experts activate per token.</span>

<span style="font-size: 14px;">In DeepSeek V3, 256 routed + 1 shared = 257 SwiGLU FFNs. If all were active, compute cost would be 257x a single FFN. With top-8, only 9 experts fire per token -- about 3.5% of fully dense cost.</span>

<span style="font-size: 14px;">This creates a powerful asymmetry between parameters and compute:</span>

* <span style="font-size: 14px;">**Total parameters:** 671 billion across all experts, giving enormous representational capacity where different tokens leverage entirely different subnetworks.</span>
* <span style="font-size: 14px;">**Active parameters per token:** About 37 billion, which determines actual inference cost, memory bandwidth, and latency.</span>

<span style="font-size: 14px;">Language contains highly diverse phenomena. A token in a math equation benefits from different transformations than a token in a poem. With 256 specialized experts, the model develops distinct pathways for different content types without paying the cost of running all pathways on every token.</span>

<span style="font-size: 14px;">Sparsity also enables hardware parallelism. Because only 8 of 256 experts run per token, experts can be distributed across many devices. Each device holds a subset, and tokens are routed to the appropriate device. This expert parallelism enables scaling to parameter counts impossible to fit on a single accelerator.</span>

---

## <span style="font-size: 16px;">Shared + Routed Architecture</span>

<span style="font-size: 14px;">The inclusion of a shared expert alongside routed experts addresses fundamental limitations of pure routing-based MoE designs.</span>

<span style="font-size: 14px;">**Why the shared expert exists.** In a pure routed MoE, every piece of knowledge must live inside routed experts. Common knowledge that benefits all tokens -- basic syntactic transformations, frequent subword patterns -- would need to be redundantly learned by many experts. This wastes capacity and creates fragile dependencies: if a token routes away from experts that learned a common pattern, it loses that knowledge.</span>

<span style="font-size: 14px;">The shared expert solves this as a guaranteed baseline. It processes every token unconditionally, so universally useful transformations consolidate there. This frees routed experts to specialize in rarer, more distinctive computations.</span>

<span style="font-size: 14px;">**How they complement each other.** The additive combination $y_{shared} + y_{routed}$ means routed experts produce a residual correction on top of the shared expert. The shared expert provides the default transformation; routed experts adjust it based on token-specific needs. This parallels how residual connections let layers learn refinements rather than complete transformations from scratch.</span>

<span style="font-size: 14px;">**Practical benefits:**</span>

* <span style="font-size: 14px;">**Reduced redundancy:** Common knowledge lives in one place rather than duplicated across routed experts.</span>
* <span style="font-size: 14px;">**Routing stability:** The shared expert provides a safety net -- even with suboptimal routing, a reasonable baseline output is guaranteed.</span>
* <span style="font-size: 14px;">**Better specialization:** With basics handled by the shared expert, routed experts develop sharper, more targeted specializations.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">The Sparse MoE FFN is the core scalability mechanism of DeepSeek V3 (DeepSeek-AI, 2024). The paper specifies:</span>

* <span style="font-size: 14px;">**Routed experts ($N_r$):** 256 independent SwiGLU FFNs per MoE layer.</span>
* <span style="font-size: 14px;">**Shared experts ($N_s$):** 1 SwiGLU FFN per MoE layer, running on all tokens.</span>
* <span style="font-size: 14px;">**Top-K ($K$):** 8 routed experts selected per token.</span>
* <span style="font-size: 14px;">**Total parameters:** 671B across all experts.</span>
* <span style="font-size: 14px;">**Active parameters per token:** 37B.</span>

<span style="font-size: 14px;">Not all layers use MoE. The first few layers use standard dense FFNs, and MoE layers begin after a certain depth, reflecting that early layers perform more uniform processing while deeper layers benefit from specialization.</span>

<span style="font-size: 14px;">DeepSeek V3 uses sigmoid scoring with auxiliary-loss-free load balancing. Traditional MoE models (Switch Transformer, GShard) use softmax routing with auxiliary losses to prevent expert collapse -- all tokens routing to the same few experts. DeepSeek V3 instead adds a bias term to each expert's score during training: underutilized experts get increased bias, overloaded experts get decreased bias. This achieves balanced load without an auxiliary loss interfering with the primary training objective.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a simplified scenario: 3 routed experts, 1 shared expert, top-1 routing, $d_{model} = 4$, expert intermediate dimension $d_{inter} = 3$.</span>

<span style="font-size: 14px;">**Input token:** $x = [1.0,\; 0.5,\; -0.5,\; 2.0]$</span>

<span style="font-size: 14px;">**Step 1: Routing.** Router centroid vectors:</span>

* <span style="font-size: 14px;">$e_1 = [0.3,\; 0.1,\; -0.2,\; 0.8]$</span>
* <span style="font-size: 14px;">$e_2 = [-0.1,\; 0.5,\; 0.4,\; 0.1]$</span>
* <span style="font-size: 14px;">$e_3 = [0.6,\; -0.3,\; 0.1,\; 0.4]$</span>

<span style="font-size: 14px;">Dot products with $x$:</span>

* <span style="font-size: 14px;">$e_1^T x = 0.3 + 0.05 + 0.1 + 1.6 = 2.05$</span>
* <span style="font-size: 14px;">$e_2^T x = -0.1 + 0.25 - 0.2 + 0.2 = 0.15$</span>
* <span style="font-size: 14px;">$e_3^T x = 0.6 - 0.15 - 0.05 + 0.8 = 1.2$</span>

<span style="font-size: 14px;">Sigmoid scores: $s_1 = \sigma(2.05) = 0.886$, $s_2 = \sigma(0.15) = 0.537$, $s_3 = \sigma(1.2) = 0.769$.</span>

<span style="font-size: 14px;">Top-1 selects expert 1. With $K = 1$, $g_1 = 1.0$.</span>

<span style="font-size: 14px;">**Step 2: Shared expert SwiGLU.** The shared expert has $W_{gate}^{(s)}$, $W_{up}^{(s)}$ of shape $(3, 4)$ and $W_{down}^{(s)}$ of shape $(4, 3)$.</span>

<span style="font-size: 14px;">Gate projection $W_{gate}^{(s)} x = [1.0,\; 0.85,\; -0.7]$. Apply swish: $[0.731,\; 0.595,\; -0.232]$.</span>

<span style="font-size: 14px;">Up projection $W_{up}^{(s)} x = [1.15,\; -0.1,\; 0.95]$.</span>

<span style="font-size: 14px;">SwiGLU elementwise multiply: $[0.731 \times 1.15,\; 0.595 \times (-0.1),\; -0.232 \times 0.95] = [0.841,\; -0.060,\; -0.220]$.</span>

<span style="font-size: 14px;">Down projection: $y_{shared} = W_{down}^{(s)} \cdot [0.841,\; -0.060,\; -0.220] = [0.242,\; 0.026,\; -0.284,\; 0.298]$.</span>

<span style="font-size: 14px;">**Step 3: Routed expert 1.** Expert 1 runs the same SwiGLU procedure with its own independent weights $W_{gate}^{(1)}$, $W_{up}^{(1)}$, $W_{down}^{(1)}$:</span>

<span style="font-size: 14px;">$\text{Expert}_1(x) = [0.510,\; -0.180,\; 0.340,\; -0.090]$</span>

<span style="font-size: 14px;">**Step 4: Combine.** $y_{routed} = g_1 \cdot \text{Expert}_1(x) = [0.510,\; -0.180,\; 0.340,\; -0.090]$</span>

$$\text{MoE\text{-}FFN}(x) = y_{shared} + y_{routed} = [0.752,\; -0.154,\; 0.056,\; 0.208]$$

<span style="font-size: 14px;">The shared expert contributed a baseline transformation and the winning routed expert refined it. Experts 2 and 3 were not computed at all.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

<span style="font-size: 14px;">**Running all experts instead of top-K.** The most fundamental error. The router must select exactly $K$ experts, and only those $K$ execute their forward pass. The remaining $N_r - K$ contribute zero and should not be computed. Running all experts turns a sparse layer into a catastrophically expensive dense one -- 257x the intended cost.</span>

<span style="font-size: 14px;">**Forgetting the shared expert.** The shared expert is not optional. If you only compute routed experts, the output is missing its baseline component. The correct formula is $y_{shared} + y_{routed}$, not just $y_{routed}$. The shared expert runs unconditionally and its output is added without any routing weight.</span>

<span style="font-size: 14px;">**Wrong combination order.** A common mistake is applying routing weights to the shared expert, treating it as another expert in the weighted sum. The shared expert output is added directly with no gating coefficient. Only routed experts receive weights $g_i$. The correct combination: $\text{output} = \text{Expert}_{shared}(x) + \sum_{i \in \mathcal{T}} g_i \cdot \text{Expert}_i(x)$.</span>

<span style="font-size: 14px;">**Each expert must have separate weights.** Every routed expert is a fully independent SwiGLU FFN with its own $W_{gate}^{(i)}$, $W_{up}^{(i)}$, $W_{down}^{(i)}$. Sharing weight matrices between experts collapses them into identical functions, eliminating the specialization that makes MoE effective. Expert 1 and Expert 2 must have completely separate weight tensors.</span>

<span style="font-size: 14px;">**Not renormalizing routing weights.** After top-K selection, the sigmoid scores must be renormalized to sum to 1. Using raw scores as combination weights causes the routed output magnitude to vary wildly. Renormalization ensures consistent scale whether the top-8 all scored 0.9 or all scored 0.5.</span>

<span style="font-size: 14px;">**Confusing sigmoid with softmax routing.** DeepSeek V3 uses sigmoid, not softmax. With softmax, scores sum to 1 before selection, coupling them. With sigmoid, each expert gets an independent $[0, 1]$ score, and renormalization happens after selection. Softmax creates competition before selection; sigmoid allows multiple experts to independently score high.</span>