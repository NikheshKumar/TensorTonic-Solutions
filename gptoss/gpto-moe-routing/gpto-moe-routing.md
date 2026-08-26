# <span style="font-size: 20px;">MoE Top-k Routing with Softmax-after-Topk</span>

<span style="font-size: 14px;">A Mixture-of-Experts (MoE) router decides which subset of expert feed-forward networks each token visits. GPT-OSS uses **top-k routing with softmax applied AFTER the top-k selection**, so the routing weights normalize over only the $k$ chosen experts. This is a small but consequential variation on the classical Switch / Mixtral recipe.</span>

---

## <span style="font-size: 16px;">Why MoE Routing Exists</span>

<span style="font-size: 14px;">A dense feed-forward block in a Transformer applies one large MLP to every token. MoE replaces that single MLP with $E$ smaller expert MLPs and a tiny **router** (a linear gate) that selects which experts process which token. Three motivations:</span>

* <span style="font-size: 14px;">**Capacity without compute.** Total parameters grow with $E$, but FLOPs per token grow only with the $k$ activated experts. GPT-OSS-120B has 128 experts and activates 4, so the per-token compute matches a much smaller dense model while the parameter count is closer to a $32\times$ larger dense model.</span>
* <span style="font-size: 14px;">**Specialization.** Different experts can learn different token populations (code, math, multilingual, dialogue). The router learns which experts each token needs, often via implicit clustering that emerges during pretraining.</span>
* <span style="font-size: 14px;">**Conditional computation.** For long inputs, most expert weights stay idle for any given token. This is friendlier to memory bandwidth than dense FFN where every weight is touched on every forward pass.</span>

<span style="font-size: 14px;">The router is therefore the central learnable component: it is a single linear layer $W_g \in \mathbb{R}^{d \times E}$ (plus bias $b_g$) that produces one **gate logit** per expert per token. Its parameter count is tiny ($d \cdot E + E$), but its decisions determine which experts see which tokens, and therefore which experts receive gradient signal during training.</span>

---

## <span style="font-size: 16px;">The Routing Equation</span>

<span style="font-size: 14px;">Given input activations $x \in \mathbb{R}^{n \times d}$ for $n$ tokens of hidden size $d$, GPT-OSS computes:</span>

$$
g = x W_g + b_g \in \mathbb{R}^{n \times E}
$$

<span style="font-size: 14px;">Per-token, the router selects the indices of the $k$ largest gate logits:</span>

$$
\mathcal{T}_t = \operatorname{top\text{-}k}(g_t) \subseteq \{1, \ldots, E\}
$$

<span style="font-size: 14px;">Then applies softmax restricted to those $k$ logits:</span>

$$
w_{t,i} = \frac{\exp(g_{t,i})}{\sum_{j \in \mathcal{T}_t} \exp(g_{t,j})}, \quad i \in \mathcal{T}_t
$$

<span style="font-size: 14px;">The output of the MoE block for token $t$ is then $\sum_{i \in \mathcal{T}_t} w_{t,i} \cdot \text{Expert}_i(x_t)$. The weights $w_{t,\cdot}$ are a proper probability distribution over the $k$ selected experts: each is in $[0, 1]$ and they sum to 1.</span>

---

## <span style="font-size: 16px;">Softmax-after-Topk vs Softmax-then-Topk</span>

<span style="font-size: 14px;">There are two natural orderings of "softmax" and "top-k". They are NOT equivalent.</span>

<span style="font-size: 14px;">**Softmax-then-topk (Switch, Mixtral, classical recipe).** First softmax over all $E$ experts, then pick the top $k$ weights:</span>

$$
p_t = \operatorname{softmax}(g_t) \in \mathbb{R}^E, \quad w_t = p_t[\mathcal{T}_t]
$$

<span style="font-size: 14px;">Here the selected weights do NOT sum to 1: the unselected experts already consumed some probability mass. Mixtral additionally renormalizes by dividing by $\sum w_t$, which is the same arithmetic as softmax-after-topk in many cases but performed in a different order.</span>

<span style="font-size: 14px;">**Softmax-after-topk (GPT-OSS).** First pick the $k$ winners by logit, then softmax only over their logits:</span>

$$
w_t = \operatorname{softmax}(g_t[\mathcal{T}_t]) \in \mathbb{R}^k
$$

<span style="font-size: 14px;">This guarantees $\sum_i w_{t,i} = 1$ exactly, and the relative weight between two selected experts depends ONLY on their two logits, not on the (potentially many) logits of unselected experts.</span>

<span style="font-size: 14px;">The two orderings produce arithmetically different weights whenever $k < E$. Example with $g = [2.0, 1.0, 0.0, -5.0]$, $k = 2$:</span>

* <span style="font-size: 14px;">**Softmax-then-topk:** softmax = $[0.7054, 0.2595, 0.0955, 0.0006]$, weights of top-2 are $[0.7054, 0.2595]$ summing to $0.9649$.</span>
* <span style="font-size: 14px;">**Softmax-after-topk:** top-2 logits are $[2.0, 1.0]$, softmax over those gives $[0.7311, 0.2689]$ summing exactly to $1.0$.</span>

<span style="font-size: 14px;">Same expert ranking, different weights. The softmax-after-topk version assigns more relative mass to the second-best expert because it does not have to "compete" with the rejected experts in the denominator.</span>

---

## <span style="font-size: 16px;">Why GPT-OSS Chose Softmax-after-Topk</span>

<span style="font-size: 14px;">The HuggingFace release notes for gpt-oss explicitly call out "softmax-after-topk" as a deliberate design choice. The practical motivations:</span>

* <span style="font-size: 14px;">**Clean probability distribution over the activated experts.** The combiner $\sum_i w_{t,i} \cdot \text{Expert}_i(x_t)$ is a true convex combination. There is no leakage of mass to experts that were never evaluated.</span>
* <span style="font-size: 14px;">**Inference simplicity.** During inference the unselected $E - k$ logits are never needed after the argsort. Softmax-after-topk operates on a tiny $(n, k)$ slice, which is friendly to fused kernels and avoids producing a full $(n, E)$ probability tensor only to discard most of it.</span>
* <span style="font-size: 14px;">**Less sensitivity to the long tail of expert logits.** With softmax-then-topk, a single very large negative or positive outlier among the unselected experts shifts the weights of the selected ones via the shared denominator. Softmax-after-topk isolates the chosen experts from this tail.</span>
* <span style="font-size: 14px;">**No need for a separate renormalization step.** Mixtral applies softmax over all experts, picks top-k, then divides by the sum of selected probabilities to make them sum to 1. That is two passes (softmax, then renormalize). Softmax-after-topk does it in one.</span>

---

## <span style="font-size: 16px;">GPT-OSS MoE Specifics</span>

<span style="font-size: 14px;">From the gpt-oss release:</span>

* <span style="font-size: 14px;">**gpt-oss-20B:** 32 experts per MoE layer, top-4 routing.</span>
* <span style="font-size: 14px;">**gpt-oss-120B:** 128 experts per MoE layer, top-4 routing.</span>
* <span style="font-size: 14px;">Every transformer block is a MoE block. There are no dense FFN layers in the body of the network.</span>
* <span style="font-size: 14px;">The router is a single linear layer with bias. No noise, no expert-choice routing, no learned temperature.</span>
* <span style="font-size: 14px;">Top-k indices come from $\operatorname{argsort}(-g)[:k]$, i.e. descending logit. The first index is the most-preferred expert.</span>

<span style="font-size: 14px;">Storing 128 experts is expensive in memory, so GPT-OSS combines this routing with MXFP4 quantization of expert weights. The router itself stays in higher precision because gate logits drive a softmax and need accurate relative ordering.</span>

---

## <span style="font-size: 16px;">Comparison with Other MoE Routers</span>

* <span style="font-size: 14px;">**Switch Transformer (Fedus et al., 2022).** Top-1 routing with softmax over all experts. The single chosen weight is then used to scale the expert output. With $k=1$, softmax-after-topk and softmax-then-topk degenerate to the same arithmetic (a single weight equal to 1 after renormalization).</span>
* <span style="font-size: 14px;">**Mixtral 8x7B (Jiang et al., 2024).** Top-2 routing. Softmax over all 8 experts, then top-2, then divide by the sum of the two selected probabilities. The renormalization step makes Mixtral arithmetically equivalent to softmax-after-topk for the WEIGHTS, but the routing decision is still based on the post-softmax probabilities (which preserves order under monotonic softmax, so the selected indices are identical).</span>
* <span style="font-size: 14px;">**DeepSeek-V3 (DeepSeek-AI, 2024).** Sigmoid on each expert logit independently, then top-k by sigmoid score. The selected scores are NOT renormalized into a probability distribution: they remain raw sigmoid values, each in $[0, 1]$. This decouples the "should I activate this expert" question (sigmoid threshold) from the "how much weight" question (raw sigmoid score). GPT-OSS made the opposite choice: softmax forces the selected weights to sum to 1.</span>
* <span style="font-size: 14px;">**Expert-Choice routing (Zhou et al., 2022).** Inverts the problem: each expert picks the top-$c$ tokens it wants. Solves load-imbalance but requires global information. Not used in GPT-OSS.</span>

---

## <span style="font-size: 16px;">Worked Numerical Example</span>

<span style="font-size: 14px;">Take $n = 2$ tokens, $d = 2$ hidden, $E = 4$ experts, $k = 2$.</span>

<span style="font-size: 14px;">Inputs: $x = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{pmatrix}$, $W_g = \begin{pmatrix} 1.0 & 0.0 & -1.0 & 0.5 \\ 0.0 & 1.0 & 0.5 & -1.0 \end{pmatrix}$, $b_g = [0.0, 0.0, 0.1, 0.2]$.</span>

<span style="font-size: 14px;">**Step 1: Gate logits.** $g = x W_g + b_g$.</span>

<span style="font-size: 14px;">Row 0: $x_0 = [1, 0]$, $x_0 W_g = [1.0, 0.0, -1.0, 0.5]$, plus bias gives $g_0 = [1.0, 0.0, -0.9, 0.7]$.</span>

<span style="font-size: 14px;">Row 1: $x_1 = [0, 1]$, $x_1 W_g = [0.0, 1.0, 0.5, -1.0]$, plus bias gives $g_1 = [0.0, 1.0, 0.6, -0.8]$.</span>

<span style="font-size: 14px;">**Step 2: Top-k by descending logit.**</span>

<span style="font-size: 14px;">Row 0 sorted: experts $[0, 3, 1, 2]$ with logits $[1.0, 0.7, 0.0, -0.9]$. Top-2 indices: $[0, 3]$.</span>

<span style="font-size: 14px;">Row 1 sorted: experts $[1, 2, 0, 3]$ with logits $[1.0, 0.6, 0.0, -0.8]$. Top-2 indices: $[1, 2]$.</span>

<span style="font-size: 14px;">**Step 3: Gather selected logits.** Row 0 gets $[1.0, 0.7]$, row 1 gets $[1.0, 0.6]$.</span>

<span style="font-size: 14px;">**Step 4: Softmax over top-k only.**</span>

<span style="font-size: 14px;">Row 0: subtract max ($1.0$) to get $[0, -0.3]$. Exponentiate: $[1.0, 0.7408]$. Sum: $1.7408$. Normalize: $[0.5744, 0.4256]$. These sum to exactly 1.</span>

<span style="font-size: 14px;">Row 1: subtract max ($1.0$) to get $[0, -0.4]$. Exponentiate: $[1.0, 0.6703]$. Sum: $1.6703$. Normalize: $[0.5987, 0.4013]$. These also sum to exactly 1.</span>

<span style="font-size: 14px;">**Final output:** indices $[[0, 3], [1, 2]]$, weights $[[0.5744, 0.4256], [0.5987, 0.4013]]$.</span>

---

## <span style="font-size: 16px;">Interaction with MXFP4 Quantization</span>

<span style="font-size: 14px;">GPT-OSS stores expert MLP weights in MXFP4 (a 4-bit microscaling float format). The router logits, however, drive a softmax whose ordering must be precise: a small noise on a borderline logit could flip which experts are activated and the network would route differently. For this reason:</span>

* <span style="font-size: 14px;">The router weight $W_g$ and bias $b_g$ are kept in higher precision (BF16 in the released checkpoints), not MXFP4.</span>
* <span style="font-size: 14px;">Softmax-after-topk further reduces precision pressure: only the top-$k$ logits ever enter the softmax denominator, so any rounding noise on the (large negative) unselected logits cannot affect the final weights at all.</span>
* <span style="font-size: 14px;">This is in contrast to softmax-then-topk, where every expert's logit contributes to the denominator. Quantization noise on the tail of unselected experts would there leak into the selected weights through the shared normalizer.</span>

---

## <span style="font-size: 16px;">Dense FFN vs Sparse MoE</span>

<span style="font-size: 14px;">A standard Transformer FFN is two linear projections with a nonlinearity:</span>

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2
$$

<span style="font-size: 14px;">Every token sees the same $W_1, W_2$. The compute is $O(d \cdot d_{\text{ff}})$ per token, and the FFN parameters are typically $2/3$ of all non-embedding weights in a Transformer.</span>

<span style="font-size: 14px;">A sparse MoE replaces the single FFN with $E$ parallel expert FFNs and a router. For each token, only $k$ experts run:</span>

$$
\text{MoE}(x_t) = \sum_{i \in \mathcal{T}_t} w_{t,i} \cdot \text{Expert}_i(x_t)
$$

<span style="font-size: 14px;">The router cost ($O(d \cdot E)$ per token) is negligible compared to expert FFN cost. The savings come from running only $k$ out of $E$ experts. With $E = 128$ and $k = 4$, the expert FFN cost is $4/128 = 3.1\%$ of running all experts, while the parameter count is $128\times$ larger.</span>

<span style="font-size: 14px;">The price of sparsity is **routing decisions**: the model has to learn which token goes where. Bad routing destroys learning signal (some experts never train) and inference quality (some experts are over-subscribed). The choice of routing function shapes that learning, which is why so many MoE papers focus exclusively on the router.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Softmax BEFORE top-k.** Easiest mistake: apply softmax over the full $E$ logits, then take the top-$k$ entries. The selected weights will be smaller than they should be (they no longer sum to 1) and the relative weighting between the two best experts changes whenever the unselected experts have non-trivial logits.</span>
* <span style="font-size: 14px;">**Forgetting the bias.** $g = x W_g + b_g$. Skipping $b_g$ silently changes the ranking when biases are large, especially in pathological tests like the "bias dominates" case.</span>
* <span style="font-size: 14px;">**Using sigmoid instead of softmax.** Sigmoid is the DeepSeek-V3 choice. With sigmoid you do NOT normalize the selected weights to sum to 1, so the combiner $\sum_i w_i \cdot \text{Expert}_i$ is no longer a convex combination and the magnitude of the MoE output drifts with $k$.</span>
* <span style="font-size: 14px;">**Bottom-k instead of top-k.** A sign error on the argsort ($\operatorname{argsort}(+g)$ instead of $\operatorname{argsort}(-g)$) selects the WORST experts. Easy to miss because the resulting weights still look like a valid probability distribution.</span>
* <span style="font-size: 14px;">**Not normalizing over the top-k slice.** Computing $\exp(g_t[\mathcal{T}_t])$ but dividing by $\sum_j \exp(g_{t,j})$ (sum over ALL experts, not just selected) gives the softmax-then-topk variant, which is wrong for GPT-OSS.</span>
* <span style="font-size: 14px;">**Skipping the max-subtraction.** $\operatorname{softmax}(z)_i = \exp(z_i) / \sum_j \exp(z_j)$ is numerically unsafe when $z$ has large positive entries: $\exp$ overflows. Always subtract $\max_j z_j$ before exponentiating, even on the small top-$k$ slice.</span>
* <span style="font-size: 14px;">**Returning indices in the wrong order.** Convention here is descending logit, with the most-preferred expert first. Sorted-by-index (ascending expert id) breaks any downstream code that assumes index 0 is the top choice.</span>
* <span style="font-size: 14px;">**Returning float weights as float32.** The test runner compares with $\operatorname{atol} = 10^{-6}$. Mixing float32 intermediates with float64 reference can fail the tolerance for tests with many close logits.</span>

---
