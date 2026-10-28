# <span style="font-size: 20px;">GLM-4.5 MoE Forward Pass</span>

<span style="font-size: 14px;">The GLM-4.5 MoE block replaces the dense FFN with a sparse **Mixture of Experts** layer that combines a small set of **routed experts** (only a few are active per token) with one **shared expert** that always runs. A learned **router** chooses which experts each token visits, using sigmoid scores and a group-restricted top-k selection inherited from DeepSeek-V3. This is the workhorse that lets GLM-4.5 scale total parameter count while keeping per-token FLOPs bounded.</span>

---

## <span style="font-size: 16px;">What the Block Computes</span>

<span style="font-size: 14px;">Given a sequence of token hidden states $x \in \mathbb{R}^{N \times H}$, the MoE block returns:</span>

$$
\text{MoE}(x) \;=\; \text{SharedExpert}(x) \;+\; \sum_{j=1}^{K} w_j \cdot \text{Expert}_{\mathcal{I}_j}(x)
$$

<span style="font-size: 14px;">where the index set $\mathcal{I} = (\mathcal{I}_1, \ldots, \mathcal{I}_K)$ is the **top-K** routed experts for that token and $w_j$ are the **router weights** assigned to them. Concretely:</span>

* <span style="font-size: 14px;">$E = $ `n_routed_experts` total routed experts (e.g. 128 in GLM-4.5)</span>
* <span style="font-size: 14px;">$K = $ `num_experts_per_tok` activated per token (e.g. 8)</span>
* <span style="font-size: 14px;">$G = $ `n_group` expert groups, $K_G = $ `topk_group` chosen groups per token</span>
* <span style="font-size: 14px;">One shared expert (`SharedExpert`) is always added on top</span>

<span style="font-size: 14px;">Each routed expert and the shared expert use a SwiGLU MLP. Only $K$ out of $E$ routed experts run per token: this is what makes the layer **sparse**.</span>

---

## <span style="font-size: 16px;">Sparse MoE in One Page</span>

<span style="font-size: 14px;">A dense FFN applies the same weights to every token. A sparse MoE swaps that for a population of experts and a per-token chooser. Two scaling levers fall out:</span>

* <span style="font-size: 14px;">**Total parameters** grow with $E$, the expert count. The model can memorize more.</span>
* <span style="font-size: 14px;">**Per-token FLOPs** stay roughly constant: only $K$ experts run, plus the cheap router and the shared expert.</span>

<span style="font-size: 14px;">The ratio $E / K$ is the **sparsity factor**. GLM-4.5 sits in the high-sparsity regime (e.g. 128 experts, 8 active), inherited from DeepSeek-V3. The shared expert is the modern twist: it absorbs the truly common features so that the routed experts can specialize on rare patterns without each one having to relearn the basics.</span>

---

## <span style="font-size: 16px;">Routing: Sigmoid Scores, Not Softmax</span>

<span style="font-size: 14px;">The router is a single linear layer $W_g \in \mathbb{R}^{H \times E}$ producing one logit per expert:</span>

$$
\text{logits} = x W_g, \qquad s = \sigma(\text{logits}) \in (0, 1)^{N \times E}
$$

<span style="font-size: 14px;">GLM-4.5 (like DeepSeek-V3) uses **sigmoid** on the logits, not softmax. The consequence is structural: $s$ is **not** a probability distribution over experts. Each expert's score is independent. This matters when you later combine the chosen experts:</span>

* <span style="font-size: 14px;">**Softmax router** (e.g. Mixtral, GPT-OSS): the $K$ chosen weights already sum to a value $\le 1$ over those experts, and a renormalization step gives a true convex combination.</span>
* <span style="font-size: 14px;">**Sigmoid router** (DeepSeek-V3, GLM-4.5): each $s_{t,e}$ is independent. The chosen $K$ weights are then optionally renormalized via `norm_topk_prob` (the default) so they sum to 1 per token.</span>

<span style="font-size: 14px;">Why this design? Sigmoid logits play more cleanly with the **auxiliary-loss-free** load balancing scheme from DeepSeek-V3 (bias correction added to logits before top-k, but excluded from the actual mixture weight). Softmax couples experts through the denominator, which fights the load-balancing nudges.</span>

---

## <span style="font-size: 16px;">Group-Restricted Top-K</span>

<span style="font-size: 14px;">Naive top-K over all $E$ experts is expensive and spreads activations across the whole MoE table. GLM-4.5 partitions experts into $G$ equal groups (size $E / G$ each) and routes in two stages:</span>

<span style="font-size: 14px;">1. **Score each group.** Within group $g$, take the **top-2 expert scores** and sum them: $\text{gs}_g = s_{(1)} + s_{(2)}$. This per-group score is what determines whether the group is worth visiting at all.</span>

<span style="font-size: 14px;">2. **Pick $K_G$ groups.** Take the top-$K_G$ groups by $\text{gs}$. Build a mask that keeps experts in those groups and zeros out experts in non-chosen groups.</span>

<span style="font-size: 14px;">3. **Top-K experts inside the chosen groups.** Apply the mask to $s$, then take top-K over all experts. Because non-chosen groups are zeroed out, the top-K naturally lands inside the $K_G$ chosen groups.</span>

<span style="font-size: 14px;">In code:</span>

$$
\text{group\_scores} \in \mathbb{R}^{N \times G}, \quad \text{chosen\_groups} \in \mathbb{Z}^{N \times K_G}
$$

$$
\text{mask}_{t, e} = \mathbb{1}[\,\text{group}(e) \in \text{chosen\_groups}_t\,], \quad \tilde s = s \odot \text{mask}
$$

$$
(\text{top\_w}, \text{top\_idx}) = \text{top-}K(\tilde s, \text{dim} = -1)
$$

<span style="font-size: 14px;">**Why top-2 inside the group?** Using the single best expert as the group score is too noisy (one strong outlier could elect a weak group). Using the full sum ignores sparsity. The top-2 sum is a robust middle ground: it rewards groups with at least two genuinely useful experts, which is exactly what helps when each token activates $K > 1$ experts.</span>

<span style="font-size: 14px;">**Hardware motivation.** Groups in DeepSeek-V3 and GLM-4.5 map to GPUs in expert-parallel training. Restricting each token to $K_G \ll G$ groups bounds **cross-GPU all-to-all traffic** to a fraction of full routing, which is the single biggest cost in MoE training. The math and the systems story are tightly coupled.</span>

---

## <span style="font-size: 16px;">Weights and Renormalization</span>

<span style="font-size: 14px;">After top-K, the router weights $w \in \mathbb{R}^{N \times K}$ are the **sigmoid scores** of the chosen experts (not the masked-then-top-k values, which are the same numbers in GLM-4.5 because no bias correction is applied in this problem). Then two post-processing steps:</span>

* <span style="font-size: 14px;">**Renormalize (`norm_topk_prob = True`).** Divide each row by its sum: $w_{t,j} \leftarrow w_{t,j} / (\sum_j w_{t,j} + \epsilon)$. After this, the K weights sum to 1 per token: the mixture is a convex combination. Without renormalization, the magnitude of the mixture depends on absolute sigmoid values and changes how much the routed branch contributes versus the shared expert.</span>

* <span style="font-size: 14px;">**Scale (`routed_scaling_factor`).** Multiply all weights by a fixed positive scalar. Default 1.0; in GLM-4.5 this is set above 1 to compensate for the fact that the routed branch sees fewer experts in expectation than a dense FFN of equivalent width. The shared expert is **not** rescaled.</span>

---

## <span style="font-size: 16px;">SwiGLU Experts and the Shared Branch</span>

<span style="font-size: 14px;">Every expert (routed and shared) is a SwiGLU MLP. With per-expert weights $W^G_e, W^U_e \in \mathbb{R}^{H \times I}$ and $W^D_e \in \mathbb{R}^{I \times H}$, the routed expert is:</span>

$$
\text{Expert}_e(x) = \big( \text{SiLU}(x W^G_e) \odot (x W^U_e) \big) \, W^D_e
$$

<span style="font-size: 14px;">where $\text{SiLU}(z) = z \cdot \sigma(z)$ and $\odot$ is element-wise multiplication. No biases (matching the GLM-4.5 / LLaMA family). The shared expert has the same shape but uses one fixed set of weights for all tokens:</span>

$$
\text{SharedExpert}(x) = \big( \text{SiLU}(x W^G_s) \odot (x W^U_s) \big) \, W^D_s
$$

<span style="font-size: 14px;">**Why include a shared expert?** Without it, every routed expert has to redundantly learn common features (basic grammar, common vocabulary, generic syntax). The shared expert absorbs that load, freeing the routed experts to specialize. Empirically (DeepSeek-V2 / V3 ablations) this both improves quality at a fixed parameter budget and stabilizes early training: even before the router learns useful routing, the model can still compute something sensible via the shared path.</span>

<span style="font-size: 14px;">**Why SwiGLU?** It is the standard for modern LLMs (PaLM, LLaMA, Mistral, DeepSeek). The gating arm gives the network a learned soft on/off per intermediate channel, which is a strictly more expressive activation than plain ReLU or GELU at matched parameter count. Inside an MoE this also helps specialization: each expert can carve out its own pattern of active channels.</span>

---

## <span style="font-size: 16px;">Putting It Together: Step-by-Step</span>

<span style="font-size: 14px;">1. **Logits:** $\text{logits} = x \cdot \text{gate\_W}$, shape $(N, E)$.</span>

<span style="font-size: 14px;">2. **Sigmoid scores:** $s = \sigma(\text{logits})$. Each entry in $(0, 1)$, independent across experts.</span>

<span style="font-size: 14px;">3. **Group score:** reshape $s$ to $(N, G, E/G)$, take top-2 along the last dim and sum: $\text{gs} \in \mathbb{R}^{N \times G}$.</span>

<span style="font-size: 14px;">4. **Group mask:** select top-$K_G$ groups per token; scatter a 1.0 into those positions; expand to expert-resolution mask of shape $(N, E)$.</span>

<span style="font-size: 14px;">5. **Masked top-K:** zero out non-chosen groups, take top-$K$ over $E$. Result: $(w, \text{idx}) \in \mathbb{R}^{N \times K}, \mathbb{Z}^{N \times K}$.</span>

<span style="font-size: 14px;">6. **Renormalize** (if `norm_topk_prob`): $w \leftarrow w / \sum w$. Then **scale**: $w \leftarrow w \cdot \text{routed\_scaling\_factor}$.</span>

<span style="font-size: 14px;">7. **Routed mixture:** for each token $t$ and each $j = 1..K$, take expert $e = \text{idx}_{t,j}$, run its SwiGLU on $x_t$, multiply by $w_{t,j}$, accumulate into $\text{out}_t$.</span>

<span style="font-size: 14px;">8. **Shared expert:** apply $\text{SharedExpert}$ to the entire $x$ in one matmul (no routing needed).</span>

<span style="font-size: 14px;">9. **Combine:** return $\text{SharedExpert}(x) + \text{out}$. Same shape as the input, $(N, H)$.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $N = 1, H = 4, E = 4, K = 2, G = 2, K_G = 1$, intermediate $I = 4$, scale = 1, norm = true.</span>

* <span style="font-size: 14px;">Compute $s = \sigma(x \cdot \text{gate\_W}) \in (0, 1)^{1 \times 4}$. Say $s = [0.6, 0.7, 0.3, 0.4]$.</span>
* <span style="font-size: 14px;">Reshape to $(1, 2, 2)$: group 0 has $[0.6, 0.7]$, group 1 has $[0.3, 0.4]$.</span>
* <span style="font-size: 14px;">Top-2 sum per group: $\text{gs} = [1.3, 0.7]$. Pick top-1 group: group 0.</span>
* <span style="font-size: 14px;">Mask: $[1, 1, 0, 0]$. Masked scores: $[0.6, 0.7, 0, 0]$.</span>
* <span style="font-size: 14px;">Top-2 over masked: weights $[0.7, 0.6]$, indices $[1, 0]$.</span>
* <span style="font-size: 14px;">Renormalize: $[0.7/1.3, 0.6/1.3] \approx [0.538, 0.462]$.</span>
* <span style="font-size: 14px;">Routed output: $0.538 \cdot \text{Expert}_1(x) + 0.462 \cdot \text{Expert}_0(x)$.</span>
* <span style="font-size: 14px;">Final: $\text{SharedExpert}(x) + \text{routed output}$.</span>

<span style="font-size: 14px;">Experts 2 and 3 (the entire second group) are **never invoked** for this token. Their SwiGLU matmuls are skipped, which is where the FLOPs savings come from.</span>

---

## <span style="font-size: 16px;">Comparison With Sibling Architectures</span>

* <span style="font-size: 14px;">**DeepSeek-V3.** Same overall recipe: sigmoid router, group-restricted top-K, shared expert, **aux-loss-free** load balancing via a per-expert correction bias added to the logits **before** top-k selection but excluded from the actual weight. GLM-4.5 inherits this whole stack. This problem omits the correction bias because the loaded buffer is zero at inference time when not actively load-balancing.</span>

* <span style="font-size: 14px;">**GPT-OSS / Mixtral-style.** Softmax router with top-K applied to the softmax outputs (or top-K-then-softmax in GPT-OSS). No shared expert. The mixture is a convex combination by construction. Simpler, but the experts must cover both general and specialized features themselves.</span>

* <span style="font-size: 14px;">**Switch Transformer.** Single expert per token ($K = 1$), softmax router, no shared expert, capacity factor handles overflow. Easier to implement but loses the benefit of mixing two views per token.</span>

* <span style="font-size: 14px;">**GLaM, Mixtral 8x7B.** Top-2 softmax routing, no shared expert. The 8x7B notation refers to 8 experts with 7B params each. Production-deployed but less aggressive sparsity than GLM-4.5 or DeepSeek-V3.</span>

---

## <span style="font-size: 16px;">GLM-4.5 Specifics</span>

* <span style="font-size: 14px;">**`n_routed_experts = 128`** in the full GLM-4.5 release: most tokens activate 8 of them, so $\approx 6\%$ utilization per token.</span>
* <span style="font-size: 14px;">**`n_group = 8`**, **`topk_group = 4`**: half the groups are visited per token, which keeps cross-device all-to-all traffic at half of an unrestricted top-K.</span>
* <span style="font-size: 14px;">**One shared expert** (`n_shared_experts = 1`) with `moe_intermediate_size` matching one routed expert: the shared branch is identical in shape to a single routed expert, not a wider dense FFN.</span>
* <span style="font-size: 14px;">**`norm_topk_prob = True`** is the default. Disabling it changes the relative scale of the routed branch versus the shared expert, which usually requires retuning `routed_scaling_factor` to recover quality.</span>
* <span style="font-size: 14px;">**`routed_scaling_factor`** is a fixed scalar set during training to balance the routed and shared contributions; GLM-4.5 reports values above 1.0 to give the routed branch slightly more weight than the shared branch.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Softmax instead of sigmoid.** Easy to confuse with Mixtral-style routing. The numbers will look similar but the weight distribution is completely different (softmax couples experts; sigmoid does not). With `norm_topk_prob = True` the result still sums to 1, but the relative weights and the resulting expert selection shift. Always check the router: GLM-4.5 is sigmoid.</span>

* <span style="font-size: 14px;">**Dropping the group restriction.** It is tempting to just do top-K over all $E$ experts and call it done. The selected experts are usually still concentrated in a few groups, but **occasionally** they spread across many groups, which spikes all-to-all traffic in expert-parallel training. The mask is what guarantees the bound. In this problem the difference also shows up numerically because the chosen experts differ.</span>

* <span style="font-size: 14px;">**Forgetting the shared expert.** A common slip: implement the routed mixture cleanly, return it. The output is in the right shape but missing the always-on shared path. Test cases catch this because the residual signal from the shared expert dominates the routed contribution at small scales.</span>

* <span style="font-size: 14px;">**Ignoring `norm_topk_prob`.** With it on, the K weights sum to 1 per token. Off, they sum to whatever $\sum_j s_{t, \text{idx}_{t,j}}$ happens to be, which depends on the absolute sigmoid magnitudes. The output magnitude visibly changes. Both branches need to be implemented; do not hardcode one.</span>

* <span style="font-size: 14px;">**Missed `routed_scaling_factor`.** It is the very last step on the weights, right before they multiply expert outputs. Easy to drop because it has a default of 1.0 that hides the bug on default-only tests. Tests in this problem include `scale = 0.25, 0.5, 2.5, 3.0` precisely to surface this.</span>

* <span style="font-size: 14px;">**Group division off by one.** `experts_per_group = n_routed_experts // n_group`, not the other way around. Reshape to `(N, G, E/G)`, not `(N, E/G, G)`. Getting this backwards still produces a tensor of the right total size, runs without errors, and silently routes to wrong experts.</span>

* <span style="font-size: 14px;">**Top-1 group score instead of top-2.** The design uses the **sum of the top 2** expert scores per group. Top-1 is too sensitive to a single outlier; full-sum ignores sparsity. Top-2 is the chosen balance.</span>

* <span style="font-size: 14px;">**Scaling the shared expert.** `routed_scaling_factor` applies only to the routed weights, never to the shared branch.</span>

---
