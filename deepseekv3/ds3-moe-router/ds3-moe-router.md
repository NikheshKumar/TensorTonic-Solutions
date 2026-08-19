# <span style="font-size: 20px;">MoE Router with Top-k Expert Selection</span>

---

## <span style="font-size: 16px;">Introduction</span>

<span style="font-size: 14px;">A Mixture-of-Experts (MoE) router is the decision-making component that determines which experts process each token in a sparse transformer. The router computes a relevance score for every expert, selects the top-k highest-scoring experts, and produces normalized combination weights. Only the selected experts perform computation, so the router is what converts a dense model into a sparse one.</span>

<span style="font-size: 14px;">DeepSeek V3 uses a softmax-based gating router across 256 routed experts with top-k = 8. Each token passes through only 8 of 256 available expert FFNs per MoE layer, achieving massive parameter capacity (671B total) while keeping per-token compute manageable (~37B active). The router is a learned linear projection followed by softmax scoring and top-k selection, yet it is the most consequential component: poor routing collapses the model into using a handful of experts while ignoring the rest.</span>

---

## <span style="font-size: 16px;">What It Is: Token-Level Routing</span>

<span style="font-size: 14px;">The MoE router operates at the token level. Each token in a sequence independently selects its own set of k experts. There is no coordination between tokens during routing: token 0 might select experts {3, 17, 42, 101, 155, 200, 230, 244} while token 1 might select {7, 23, 56, 90, 112, 178, 210, 251}. The selections can overlap, partially overlap, or be entirely disjoint.</span>

<span style="font-size: 14px;">The router itself is a linear layer with no bias and no hidden layers. It takes a token embedding of dimension d_model and projects it into a vector of dimension N (the number of experts). The resulting N-dimensional vector contains raw logits, one per expert. These logits pass through softmax to produce a probability distribution over experts.</span>

<span style="font-size: 14px;">From this distribution, the top-k experts are selected. Their softmax probabilities become the combination weights, but they must be renormalized so the k selected weights sum to 1. The unselected experts receive zero weight and are not computed.</span>

<span style="font-size: 14px;">The router's parameters are learned end-to-end via backpropagation. Gradients flow through the combination weights into the gating projection, teaching the router which experts are useful for which token representations.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The routing mechanism involves three stages: scoring, selection, and renormalization.</span>

<span style="font-size: 14px;">**Stage 1: Softmax scoring.** Given token representation x of shape (d_model,) and gating matrix W_gate of shape (N, d_model), compute logits and apply softmax:</span>

$$\ell_i = W_{\text{gate}}[i,:] \cdot x, \quad i = 1, \ldots, N$$

$$s_i = \text{softmax}(\ell)_i = \frac{e^{\ell_i}}{\sum_{j=1}^{N} e^{\ell_j}}$$

<span style="font-size: 14px;">This produces a probability distribution s over all N experts, where every s_i is in (0, 1) and the values sum to 1.</span>

<span style="font-size: 14px;">**Stage 2: Top-k selection.** Identify the k experts with the largest softmax scores:</span>

$$\mathcal{T} = \text{TopK}(\{s_1, s_2, \ldots, s_N\}, k)$$

<span style="font-size: 14px;">The set T contains exactly k indices. All experts not in T receive zero weight.</span>

<span style="font-size: 14px;">**Stage 3: Renormalization.** The selected k scores are a subset of the full distribution and generally do not sum to 1. Renormalize:</span>

$$w_i = \frac{s_i}{\sum_{j \in \mathcal{T}} s_j}, \quad i \in \mathcal{T}$$

<span style="font-size: 14px;">Now the k weights w_i sum to exactly 1, determining how much each selected expert contributes.</span>

<span style="font-size: 14px;">**Final combination.** The MoE layer output for token x is:</span>

$$\text{MoE}(x) = \sum_{i \in \mathcal{T}} w_i \cdot \text{Expert}_i(x)$$

<span style="font-size: 14px;">where Expert_i(x) is the i-th expert FFN applied to x, typically a SwiGLU feedforward network with independent parameters.</span>

---

## <span style="font-size: 16px;">Why Softmax Routing</span>

<span style="font-size: 14px;">DeepSeek V3 uses softmax for its gating function, which is a deliberate choice with important implications that differ from sigmoid-based alternatives like the one used in Arcee Trinity.</span>

<span style="font-size: 14px;">**Softmax creates competition between experts.** Because softmax normalizes across all N experts, increasing one expert's score necessarily decreases others. The router must commit: it cannot give high scores to all experts. Each token's total probability budget is exactly 1, and the router must allocate it.</span>

<span style="font-size: 14px;">**Sigmoid treats experts independently.** With sigmoid routing, each expert's score is computed via sigma(logit_i). Expert 3 getting a high score does not affect expert 7's score. Multiple experts can simultaneously have scores near 1.0, or all near 0.0.</span>

<span style="font-size: 14px;">**Implications for routing behavior:**</span>

* <span style="font-size: 14px;">**Softmax naturally produces a ranking.** The probability distribution encodes relative preferences. Top-k selection picks from a distribution where expert relevance is already ordered.</span>
* <span style="font-size: 14px;">**Softmax is more prone to winner-take-all collapse.** The exponential amplifies small logit differences. If one expert's logit is slightly higher, its softmax score is disproportionately larger, causing the same few experts to dominate across tokens.</span>
* <span style="font-size: 14px;">**Sigmoid allows more uniform scoring.** Without competition, sigmoid routing can give similar scores to many experts. This distributes load more evenly but makes top-k selection less meaningful when experts have nearly identical scores.</span>

<span style="font-size: 14px;">DeepSeek V3 chose softmax for cleaner gradient signals. The competitive pressure forces differentiation: if two experts are equally good for a token type, softmax pushes the router to prefer one, freeing the other to specialize elsewhere. The load imbalance downside is addressed through auxiliary-loss-free balancing rather than weakening the routing signal.</span>

---

## <span style="font-size: 16px;">Token-Level Independence</span>

<span style="font-size: 14px;">A critical property of MoE routing is that each token makes its routing decision independently. There is no mechanism for tokens to communicate about which experts they are selecting.</span>

<span style="font-size: 14px;">**Different tokens go to different experts.** In a sentence like "The cat sat on the mat", each token produces its own logit vector, softmax distribution, and top-k selection. Function words like "the" might consistently route to experts specializing in syntactic structure, while content words like "cat" might route to experts handling semantic content.</span>

<span style="font-size: 14px;">**No sequence-level coordination.** Token 3 does not know what experts tokens 0, 1, and 2 selected. Multiple tokens can independently select the same expert, causing it to process a larger batch. Conversely, some experts might receive zero tokens from a given sequence. This uneven distribution is inherent to token-level routing.</span>

<span style="font-size: 14px;">**Batching implications.** In a training batch with thousands of tokens, each MoE layer must dynamically determine which tokens go to which experts. Tokens destined for expert i are gathered, processed, and results scattered back to original positions. This gather-compute-scatter pattern is the main systems challenge in MoE.</span>

<span style="font-size: 14px;">**Expert capacity.** Because assignment is dynamic, some experts receive many more tokens than others. Hardware operates best with balanced workloads, so extreme imbalance wastes compute on idle experts while overloaded ones drop tokens or become bottlenecks. This motivates load-balancing mechanisms.</span>

---

## <span style="font-size: 16px;">Paper Context: DeepSeek V3's MoE</span>

<span style="font-size: 14px;">DeepSeek V3 is a 671-billion-parameter MoE language model with ~37 billion active parameters per token.</span>

<span style="font-size: 14px;">**Scale of the MoE configuration:**</span>

* <span style="font-size: 14px;">**256 routed experts** per MoE layer, each a complete SwiGLU FFN with independent parameters.</span>
* <span style="font-size: 14px;">**Top-k = 8:** each token selects 8 of 256 experts, so only ~3.1% of routed expert parameters are active per token.</span>
* <span style="font-size: 14px;">**1 shared expert** that processes every token regardless of routing, providing a stable baseline representation.</span>
* <span style="font-size: 14px;">**61 MoE layers** out of the total transformer depth, with remaining layers being dense.</span>

<span style="font-size: 14px;">**The router in context.** The gating matrix W_gate has shape (256, d_model). For DeepSeek V3 with d_model = 7168, this is a 256 x 7168 matrix with about 1.8 million parameters per MoE layer. Compared to the billions of parameters in the expert FFNs, the router itself is tiny, yet it controls the entire compute graph.</span>

<span style="font-size: 14px;">**Auxiliary-loss-free balancing.** Rather than adding a load-balancing loss (as in Switch Transformer and GShard), DeepSeek V3 introduces a per-expert bias adjusted dynamically based on observed load. This avoids interference with the primary language modeling objective. The router's softmax scores remain trained purely by language modeling loss, while balance is achieved separately.</span>

<span style="font-size: 14px;">**Device-level balance.** DeepSeek V3 also constrains how many tokens each device processes, preventing any single GPU from becoming a bottleneck. The router provides relevance signals; the balancing system prevents degeneration into imbalanced computation.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a simplified MoE with N = 4 experts, top-k = 2, and d_model = 3.</span>

<span style="font-size: 14px;">**Input token:** x = [0.5, 1.2, -0.3]</span>

<span style="font-size: 14px;">**Gating matrix W_gate** (shape 4 x 3):</span>

* <span style="font-size: 14px;">**Row 0:** [0.4, 0.1, -0.2], **Row 1:** [-0.3, 0.6, 0.5], **Row 2:** [0.7, -0.1, 0.3], **Row 3:** [0.1, 0.8, -0.4]</span>

<span style="font-size: 14px;">**Step 1: Compute logits.** Each logit is the dot product of a row of W_gate with x:</span>

* <span style="font-size: 14px;">**l_0** = 0.4(0.5) + 0.1(1.2) + (-0.2)(-0.3) = 0.20 + 0.12 + 0.06 = **0.38**</span>
* <span style="font-size: 14px;">**l_1** = -0.3(0.5) + 0.6(1.2) + 0.5(-0.3) = -0.15 + 0.72 - 0.15 = **0.42**</span>
* <span style="font-size: 14px;">**l_2** = 0.7(0.5) - 0.1(1.2) + 0.3(-0.3) = 0.35 - 0.12 - 0.09 = **0.14**</span>
* <span style="font-size: 14px;">**l_3** = 0.1(0.5) + 0.8(1.2) - 0.4(-0.3) = 0.05 + 0.96 + 0.12 = **1.13**</span>

<span style="font-size: 14px;">**Step 2: Apply softmax.** Exponentials: e^0.38 = 1.462, e^0.42 = 1.522, e^0.14 = 1.150, e^1.13 = 3.096. Sum = 7.230.</span>

* <span style="font-size: 14px;">**s_0** = 1.462 / 7.230 = **0.202**, **s_1** = 1.522 / 7.230 = **0.211**</span>
* <span style="font-size: 14px;">**s_2** = 1.150 / 7.230 = **0.159**, **s_3** = 3.096 / 7.230 = **0.428**</span>

<span style="font-size: 14px;">Verify: 0.202 + 0.211 + 0.159 + 0.428 = 1.000.</span>

<span style="font-size: 14px;">**Step 3: Top-2 selection.** Highest scores: s_3 = 0.428 and s_1 = 0.211. Selected set T = {1, 3}.</span>

<span style="font-size: 14px;">**Step 4: Renormalize.** Sum of selected = 0.428 + 0.211 = 0.639. So w_1 = 0.211 / 0.639 = **0.330**, w_3 = 0.428 / 0.639 = **0.670**. Verify: 0.330 + 0.670 = 1.000.</span>

<span style="font-size: 14px;">**Step 5: Combine expert outputs.** Suppose Expert_1(x) = [0.3, -0.1, 0.5] and Expert_3(x) = [0.7, 0.4, -0.2]:</span>

$$\text{MoE}(x) = 0.330 \cdot [0.3, -0.1, 0.5] + 0.670 \cdot [0.7, 0.4, -0.2]$$

$$= [0.099, -0.033, 0.165] + [0.469, 0.268, -0.134]$$

$$= [0.568, 0.235, 0.031]$$

<span style="font-size: 14px;">Expert 3 dominates with weight 0.670. Only 2 of 4 expert FFNs were computed, halving compute cost versus a dense layer.</span>

---

## <span style="font-size: 16px;">Load Balancing Connection</span>

<span style="font-size: 14px;">A learned router with softmax gating and no balancing will almost always degenerate into using only a few experts. This is the "rich get richer" problem: if expert 5 produces slightly better outputs early in training, more tokens route to it, it gets more gradient updates, improves further, and attracts even more tokens. Underutilized experts stagnate.</span>

<span style="font-size: 14px;">**Why imbalance is catastrophic:**</span>

* <span style="font-size: 14px;">**Wasted parameters.** If 200 of 256 experts never get selected, their parameters contribute nothing. The model effectively shrinks from 671B to a fraction of its intended capacity.</span>
* <span style="font-size: 14px;">**Compute bottlenecks.** The few popular experts process most tokens, creating hot spots on whatever device hosts them. Other devices sit idle.</span>
* <span style="font-size: 14px;">**Reduced generalization.** Expert specialization requires diversity. If all tokens pass through the same experts, the model cannot learn specialized sub-networks for different input types.</span>

<span style="font-size: 14px;">**Traditional fix: auxiliary loss.** Models like Switch Transformer add a term penalizing uneven utilization: L_total = L_lm + alpha * L_balance. This directly conflicts with the language modeling objective, and alpha is hard to tune: too small and imbalance persists, too large and routing quality degrades.</span>

<span style="font-size: 14px;">**DeepSeek V3's approach: auxiliary-loss-free balancing.** Instead of a loss term, DeepSeek V3 maintains a per-expert bias b_i. Selection uses s_i + b_i, but combination weights use only s_i. Biases adjust dynamically: overloaded experts get decreased b_i, underloaded experts get increased b_i, steering allocation without contaminating the loss.</span>

<span style="font-size: 14px;">The router implemented here is the foundation making load balancing necessary. Learned routers always exhibit preference collapse without countermeasures, and understanding basic routing is the prerequisite for understanding DeepSeek V3's balancing solution.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">Several subtle errors can break a router implementation while producing outputs of the correct shape.</span>

* <span style="font-size: 14px;">**Forgetting renormalization.** After top-k selection, the k softmax scores no longer sum to 1. If you skip renormalization, the output is scaled down. For example, if top-2 scores are 0.3 and 0.2, their sum is 0.5, so the output is at half its intended magnitude. Downstream layers see systematically suppressed inputs, degrading training.</span>

* <span style="font-size: 14px;">**Softmax numerical instability.** Computing e^(l_i) directly overflows for large logits. The fix: subtract the maximum logit before exponentiation, computing s_i = e^(l_i - max(l)) / sum(e^(l_j - max(l))). Mathematically equivalent but numerically stable. Most frameworks handle this, but manual implementations must include it.</span>

* <span style="font-size: 14px;">**All tokens routing to the same expert.** If W_gate initialization gives one expert consistently higher logits, softmax amplifies this exponentially. A single expert dominates all routing. Proper initialization (small random values, orthogonal, or uniform bias) is essential.</span>

* <span style="font-size: 14px;">**Tie-breaking in top-k with equal scores.** When experts have identical softmax scores, top-k must break ties deterministically. Different argsort/topk implementations break ties differently across hardware, causing reproducibility issues. Near-ties cause different selections on different devices in distributed training.</span>

* <span style="font-size: 14px;">**Gradient flow through non-selected experts.** Only top-k experts receive gradients via combination weights. Non-selected experts get zero gradient. An expert never selected stops learning, creating a feedback loop where unused experts become irrelevant without load balancing.</span>

* <span style="font-size: 14px;">**Confusing logits with combination weights.** Raw logits from W_gate * x are not weights. They must pass through softmax, top-k selection, then renormalization. Using raw logits or un-renormalized scores produces incorrect outputs.</span>

---