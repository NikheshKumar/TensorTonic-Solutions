# <span style="font-size: 20px;">Sigmoid MoE Router</span>

<span style="font-size: 14px;">In a Mixture-of-Experts (MoE) model, the router decides which experts process each token. Traditional routers use softmax to produce a probability distribution over experts, forcing them to compete for allocation. A sigmoid router scores each expert independently through the sigmoid function, allowing multiple experts to be "strongly selected" without suppressing one another. Arcee Trinity uses sigmoid-based routing in its coarse-grained MoE architecture, reflecting a growing trend away from softmax gating in large-scale sparse models.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">A router (also called a gating network) sits at the heart of every MoE layer. It takes a token's hidden representation as input and produces a set of weights that determine which experts are activated and how much each contributes to the final output.</span>

<span style="font-size: 14px;">In a sigmoid router, the gating network projects the hidden state into a vector of expert scores using a learned weight matrix, then applies the sigmoid activation element-wise. Each expert receives an independent score between 0 and 1. The top-k experts by score are selected, and their weights are renormalized to sum to 1 before computing the weighted combination of expert outputs.</span>

<span style="font-size: 14px;">The key difference from softmax routing: sigmoid does not create a probability distribution. Expert scores are computed in isolation. One expert getting a high score does not force another expert's score down. This independence is the central design choice and has meaningful consequences for load balancing, gradient flow, and expert specialization.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">Step 1: Compute Expert Scores</span>

<span style="font-size: 14px;">Given a token hidden state $\mathbf{x} \in \mathbb{R}^d$ and a learned gating matrix $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{N \times d}$ (where $N$ is the number of experts):</span>

$$
\mathbf{s} = \sigma(\mathbf{W}_{\text{gate}} \cdot \mathbf{x})
$$

<span style="font-size: 14px;">where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is applied element-wise. Each $s_i \in (0, 1)$ is the independent activation score for expert $i$.</span>

### <span style="font-size: 14px;">Step 2: Top-K Selection</span>

<span style="font-size: 14px;">Select the $k$ experts with the highest scores:</span>

$$
\text{indices}, \text{values} = \text{TopK}(\mathbf{s}, k)
$$

<span style="font-size: 14px;">This yields $k$ index-value pairs. Let $\mathcal{T}$ be the set of selected indices and $\{s_i : i \in \mathcal{T}\}$ the corresponding scores.</span>

### <span style="font-size: 14px;">Step 3: Renormalization</span>

<span style="font-size: 14px;">The selected sigmoid scores do not sum to 1 in general. Renormalize to produce proper weights:</span>

$$
w_i = \frac{s_i}{\sum_{j \in \mathcal{T}} s_j}, \quad i \in \mathcal{T}
$$

### <span style="font-size: 14px;">Step 4: Weighted Expert Combination</span>

<span style="font-size: 14px;">The final output for the token is the weighted sum of the selected experts' outputs:</span>

$$
\mathbf{y} = \sum_{i \in \mathcal{T}} w_i \cdot \text{Expert}_i(\mathbf{x})
$$

---

## <span style="font-size: 16px;">Sigmoid vs Softmax Routing</span>

### <span style="font-size: 14px;">Softmax Routing (Traditional)</span>

<span style="font-size: 14px;">Given logits $\mathbf{z} = \mathbf{W}_{\text{gate}} \cdot \mathbf{x}$, softmax computes:</span>

$$
p_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}
$$

<span style="font-size: 14px;">This is a probability distribution: all $p_i$ sum to 1. Increasing one expert's score necessarily decreases others. Experts compete in a zero-sum game for routing weight.</span>

### <span style="font-size: 14px;">Sigmoid Routing</span>

<span style="font-size: 14px;">Sigmoid computes each score independently:</span>

$$
s_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

<span style="font-size: 14px;">There is no normalization across experts at this stage. Multiple experts can simultaneously have scores near 1.0 or near 0.0. The scores are independent and do not form a distribution.</span>

### <span style="font-size: 14px;">Concrete Comparison</span>

<span style="font-size: 14px;">Consider 4 experts with logits $\mathbf{z} = [2.0, 1.8, -1.0, -0.5]$:</span>

* <span style="font-size: 14px;">**Softmax scores:** $[0.438, 0.358, 0.022, 0.036]$ (sum = 1.0, the two positive logits dominate but suppress each other)</span>
* <span style="font-size: 14px;">**Sigmoid scores:** $[0.881, 0.858, 0.269, 0.378]$ (no constraint on sum, the top two experts both score high without competition)</span>

<span style="font-size: 14px;">With softmax, Expert 1 getting 43.8% means Expert 2 can only get 35.8% even though their logits are close (2.0 vs 1.8). With sigmoid, both experts get similar high scores (0.881 vs 0.858), faithfully reflecting the small gap in their logits.</span>

### <span style="font-size: 14px;">Practical Consequences</span>

* <span style="font-size: 14px;">**Gradient independence:** In sigmoid routing, the gradient of $s_i$ with respect to $z_i$ depends only on $s_i$ itself ($s_i(1-s_i)$). In softmax, the gradient of $p_i$ depends on all other scores, creating coupling.</span>
* <span style="font-size: 14px;">**Load balancing:** Softmax naturally distributes weight, which can help with load balance but also creates artificial competition. Sigmoid requires explicit auxiliary losses or capacity constraints to prevent all tokens from routing to the same expert.</span>
* <span style="font-size: 14px;">**Score semantics:** A sigmoid score of 0.9 always means "this expert is highly relevant." A softmax score of 0.9 could mean "this expert is slightly better than weak alternatives" or "this expert is overwhelmingly dominant" depending on other experts' scores.</span>

---

## <span style="font-size: 16px;">Why Renormalize</span>

<span style="font-size: 14px;">After top-k selection, the selected sigmoid scores are arbitrary positive numbers in $(0, 1)$ that do not sum to 1. Renormalization is essential for two reasons.</span>

### <span style="font-size: 14px;">Proper Weighted Average</span>

<span style="font-size: 14px;">The MoE output is a weighted combination of expert outputs. If weights do not sum to 1, the output magnitude scales unpredictably. With $k=2$ and selected scores $[0.88, 0.86]$, the raw sum is 1.74. Without renormalization, the combined output would have ~1.74x the magnitude of a single expert, artificially inflating activations.</span>

<span style="font-size: 14px;">After renormalization: $w_1 = 0.88/1.74 = 0.506$, $w_2 = 0.86/1.74 = 0.494$. The output has stable magnitude regardless of the raw sigmoid scores.</span>

### <span style="font-size: 14px;">Training Stability</span>

<span style="font-size: 14px;">Without renormalization, the effective learning rate for the MoE layer fluctuates with the sum of selected scores. When sigmoid scores are high (both near 1.0), the output is ~2x scaled, producing larger gradients. When scores are moderate (both near 0.5), the output is ~1x scaled. This creates inconsistent gradient magnitudes that destabilize training.</span>

<span style="font-size: 14px;">Renormalization decouples the output scale from the absolute magnitude of sigmoid scores. The router then only controls the relative weighting, not the overall scale.</span>

---

## <span style="font-size: 16px;">Paper Context: Arcee Trinity</span>

<span style="font-size: 14px;">Arcee Trinity is a coarse-grained MoE model that merges three specialized source models (for reasoning, creative writing, and general instruction-following) into a unified architecture with shared and expert-specific parameters.</span>

* <span style="font-size: 14px;">**Coarse-grained MoE:** Rather than having many small experts per layer (like Switch Transformer with 128 experts), Arcee Trinity uses a small number of large experts derived from full pretrained models. Each expert is itself a capable model.</span>
* <span style="font-size: 14px;">**Sigmoid routing choice:** With few experts (e.g., 3), softmax routing can be overly decisive. Softmax over 3 logits tends to produce near-one-hot distributions, which defeats the purpose of combining expert knowledge. Sigmoid allows the router to assign genuinely balanced weights when multiple experts are relevant.</span>
* <span style="font-size: 14px;">**Merging context:** In model merging, the "experts" are not randomly initialized FFN blocks but fully trained models with distinct capabilities. The router must learn which source model's knowledge is most relevant for each input, a task well suited to independent scoring.</span>

<span style="font-size: 14px;">Earlier MoE architectures chose softmax for its built-in normalization. Switch Transformer (Fedus et al., 2021) uses softmax with $k=1$. GShard (Lepikhin et al., 2020) uses softmax with $k=2$. V-MoE (Riquelme et al., 2021) introduced per-expert capacity factors with softmax. The shift to sigmoid routing reflects recent findings that the competition inherent in softmax can harm expert utilization and that independent scoring with explicit load balancing produces better results in practice.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Walk through sigmoid routing with $N = 4$ experts and $k = 2$.</span>

### <span style="font-size: 14px;">Setup</span>

<span style="font-size: 14px;">Token hidden state $\mathbf{x} \in \mathbb{R}^d$. After projection through $\mathbf{W}_{\text{gate}}$, we obtain logits:</span>

$$
\mathbf{z} = \mathbf{W}_{\text{gate}} \cdot \mathbf{x} = [1.5, \; 0.8, \; -0.3, \; 2.1]
$$

### <span style="font-size: 14px;">Step 1: Sigmoid Scores</span>

<span style="font-size: 14px;">Apply sigmoid to each logit independently:</span>

$$
s_1 = \sigma(1.5) = \frac{1}{1 + e^{-1.5}} = \frac{1}{1 + 0.223} = 0.818
$$

$$
s_2 = \sigma(0.8) = \frac{1}{1 + e^{-0.8}} = \frac{1}{1 + 0.449} = 0.690
$$

$$
s_3 = \sigma(-0.3) = \frac{1}{1 + e^{0.3}} = \frac{1}{1 + 1.350} = 0.426
$$

$$
s_4 = \sigma(2.1) = \frac{1}{1 + e^{-2.1}} = \frac{1}{1 + 0.122} = 0.891
$$

<span style="font-size: 14px;">All four scores are independent. Note that $s_1 + s_2 + s_3 + s_4 = 2.825$, far from 1.0.</span>

### <span style="font-size: 14px;">Step 2: Top-K Selection (k=2)</span>

<span style="font-size: 14px;">Rank the scores: Expert 4 (0.891) > Expert 1 (0.818) > Expert 2 (0.690) > Expert 3 (0.426).</span>

<span style="font-size: 14px;">Select top-2: Expert 4 and Expert 1. Discard Experts 2 and 3.</span>

### <span style="font-size: 14px;">Step 3: Renormalize</span>

<span style="font-size: 14px;">Sum of selected scores: $0.891 + 0.818 = 1.709$.</span>

$$
w_4 = \frac{0.891}{1.709} = 0.521
$$

$$
w_1 = \frac{0.818}{1.709} = 0.479
$$

<span style="font-size: 14px;">Verify: $0.521 + 0.479 = 1.000$.</span>

### <span style="font-size: 14px;">Step 4: Combine Expert Outputs</span>

<span style="font-size: 14px;">Suppose Expert 4 produces $\mathbf{e}_4$ and Expert 1 produces $\mathbf{e}_1$. The MoE layer output is:</span>

$$
\mathbf{y} = 0.521 \cdot \mathbf{e}_4 + 0.479 \cdot \mathbf{e}_1
$$

<span style="font-size: 14px;">The router assigns a near-even split (52/48), reflecting the relatively close sigmoid scores. With softmax on the same logits, the split would be $[0.354, 0.134, 0.045, 0.497]$, giving a 58/42 split for the same top-2 experts. Sigmoid preserves the closeness of the original logits better after renormalization.</span>

---

## <span style="font-size: 16px;">Modern Context: Routing Strategies in MoE</span>

### <span style="font-size: 14px;">Softmax Top-K Routing</span>

<span style="font-size: 14px;">The most common approach. Switch Transformer uses top-1 softmax routing for maximum sparsity. GShard uses top-2. Both require auxiliary load-balancing losses to prevent expert collapse, where a few experts receive most tokens while others go unused.</span>

### <span style="font-size: 14px;">Expert Choice Routing</span>

<span style="font-size: 14px;">Introduced by Zhou et al. (2022), this inverts the paradigm: instead of tokens choosing experts, each expert selects its top-k tokens. This guarantees perfect load balance by construction but introduces variable computation per token (some tokens may be selected by many experts, others by none).</span>

### <span style="font-size: 14px;">Hash Routing</span>

<span style="font-size: 14px;">Roller et al. (2021) showed that deterministic hash-based routing (assigning tokens to experts based on a hash of their content) can match learned routing in some settings. This eliminates the routing network entirely but sacrifices adaptivity.</span>

### <span style="font-size: 14px;">Sigmoid Routing in Context</span>

<span style="font-size: 14px;">Sigmoid routing occupies a middle ground. It retains learned, input-dependent routing (unlike hashing) while avoiding the zero-sum competition of softmax. The independent scoring makes it particularly natural for coarse-grained MoE where the number of experts is small and each expert should be evaluated on its own merits rather than relative to siblings.</span>

<span style="font-size: 14px;">Recent models beyond Arcee Trinity have also explored non-softmax gating, reflecting a broader understanding that softmax's mutual exclusivity is not always desirable in expert routing.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Forgetting Renormalization</span>

<span style="font-size: 14px;">The most common implementation error. Without renormalization, the MoE output scale depends on the sum of selected sigmoid scores. If both selected experts score near 1.0, the output is ~2x scaled; if both score near 0.5, it is ~1x scaled. This inconsistency destabilizes training. Always divide by the sum of selected scores.</span>

### <span style="font-size: 14px;">Sigmoid Saturation with Large Logits</span>

<span style="font-size: 14px;">When logits $z_i$ have large absolute values, sigmoid saturates near 0 or 1. The gradient $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ approaches 0 in both extremes. If gating weights grow unchecked, the router cannot update through gradient descent.</span>

<span style="font-size: 14px;">Mitigations include weight decay on $\mathbf{W}_{\text{gate}}$, gradient clipping, or a temperature parameter $\sigma(z/\tau)$ to control routing sharpness.</span>

### <span style="font-size: 14px;">All Experts Getting Similar Scores</span>

<span style="font-size: 14px;">Unlike softmax, sigmoid does not force differentiation. If the gating network produces logits near zero for all experts, all sigmoid scores cluster around 0.5. After top-k selection and renormalization, the weights become nearly uniform ($1/k$ each). The router is effectively not routing, just averaging. This wastes the MoE capacity.</span>

<span style="font-size: 14px;">This problem is more severe early in training when the gating weights are small. Proper initialization of $\mathbf{W}_{\text{gate}}$ (not too small) and sufficient learning rate for the router are important.</span>

### <span style="font-size: 14px;">Top-K with Ties</span>

<span style="font-size: 14px;">When multiple experts have identical sigmoid scores, top-k selection is ambiguous. Most frameworks break ties by expert index (lower index wins), creating systematic bias. This is rare with float32 but can occur with float16 or bfloat16, especially when sigmoid saturates.</span>

### <span style="font-size: 14px;">Gradient Flow: Sigmoid vs Softmax</span>

<span style="font-size: 14px;">The sigmoid gradient for expert $i$ is $\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)$, depending only on $s_i$. The softmax gradient has off-diagonal terms: $\frac{\partial p_i}{\partial z_j} = -p_i \cdot p_j$ for $i \neq j$. This means in softmax routing, updating one expert's logit affects the gradients of all others, creating implicit coupling. Sigmoid avoids this coupling, which simplifies optimization but also means the router gets no "signal" about non-selected experts from the routing loss alone. Auxiliary losses that encourage exploration of all experts become more important with sigmoid routing.</span>

### <span style="font-size: 14px;">Load Imbalance Without Auxiliary Loss</span>

<span style="font-size: 14px;">Softmax's competitive normalization provides weak implicit load balancing: boosting one expert suppresses others. Sigmoid has no such mechanism. Without an explicit load-balancing loss, sigmoid routing can collapse to always selecting the same experts. The standard mitigation is an auxiliary loss penalizing uneven expert utilization across a batch.</span>

---