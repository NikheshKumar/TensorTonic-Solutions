# <span style="font-size: 20px;">Group-Routed Top-k Router</span>

<span style="font-size: 14px;">GLM-4.5 uses a **group-restricted top-k MoE router** inspired by DeepSeek-V3: experts are partitioned into fixed groups, only the highest-scoring groups are kept active per token, and the final top-k experts are picked from within those surviving groups. Routing uses sigmoid scores rather than softmax, which decouples the per-expert affinity from a global probability simplex.</span>

---

## <span style="font-size: 16px;">Why MoE Routing Is Hard</span>

<span style="font-size: 14px;">A sparse Mixture-of-Experts layer replaces a dense FFN with $E$ experts and routes each token to a small subset of $k$ experts (typically $k=8$, $E=128$ in GLM-4.5). Two pressures shape the router design:</span>

* <span style="font-size: 14px;">**Capacity:** the model has $E \cdot d_{\text{ff}}$ FFN parameters but only $k \cdot d_{\text{ff}}$ are used per token, so the router decides which slice of the parameter budget gets activated.</span>
* <span style="font-size: 14px;">**Communication:** experts live on different devices. An all-to-all dispatch must move token activations to the chosen experts and gather outputs back. The wider the routing pattern, the more bandwidth is consumed.</span>

<span style="font-size: 14px;">A naive top-k router (score every expert, pick the $k$ highest) tends to spread chosen experts uniformly across the $E$-wide expert pool. In a distributed setup this means almost every device receives some tokens, so the all-to-all traffic grows with the number of devices, not with $k$.</span>

---

## <span style="font-size: 16px;">The Group Routing Idea</span>

<span style="font-size: 14px;">Group routing, introduced in DeepSeek-V3 and adopted by GLM-4.5, breaks the $E$ experts into $G$ equal-size groups of $E/G$ experts each. Per token, the router first picks $G'$ groups (called $\texttt{topk\_group}$) and then runs the final top-k selection only inside those groups. Concretely:</span>

* <span style="font-size: 14px;">$E = \texttt{n\_routed\_experts}$ (e.g. 128 in GLM-4.5)</span>
* <span style="font-size: 14px;">$G = \texttt{n\_group}$ (e.g. 8)</span>
* <span style="font-size: 14px;">$G' = \texttt{topk\_group}$ (e.g. 4)</span>
* <span style="font-size: 14px;">$k = \texttt{num\_experts\_per\_tok}$ (e.g. 8)</span>

<span style="font-size: 14px;">Tokens can never activate experts in non-selected groups, so each token's expert footprint is bounded by $G' \cdot (E/G)$. If groups map cleanly onto devices, all-to-all traffic per token is limited to $G'$ devices instead of all $G$. For GLM-4.5 this caps device fan-out at $4/8 = 50\%$ of the expert-parallel mesh.</span>

---

## <span style="font-size: 16px;">Step-by-Step Algorithm</span>

<span style="font-size: 14px;">Given input $x \in \mathbb{R}^{T \times d}$ and router weight $W_r \in \mathbb{R}^{d \times E}$:</span>

<span style="font-size: 14px;">1. **Score every expert.** Compute logits $\ell = x W_r \in \mathbb{R}^{T \times E}$, then per-expert affinities $s = \sigma(\ell) \in (0, 1)^{T \times E}$ with sigmoid (not softmax).</span>

<span style="font-size: 14px;">2. **Score every group.** Reshape $s$ into $T \times G \times (E/G)$. The score of group $g$ is the sum of its top-2 expert affinities. For groups of size 1, the single score is used.</span>

<span style="font-size: 14px;">3. **Select groups.** For each token, take the indices of the top $G'$ groups by group score. These are the active groups for that token.</span>

<span style="font-size: 14px;">4. **Mask non-selected experts.** Replace the affinity of any expert outside the active groups with $-\infty$ so it cannot win the next top-k.</span>

<span style="font-size: 14px;">5. **Top-k inside the mask.** Take the indices of the $k$ highest masked scores. These are the experts the token routes to.</span>

<span style="font-size: 14px;">6. **Gather weights from sigmoid scores.** The routing weights are gathered from the original sigmoid affinities $s$ at the chosen indices, not from the masked scores. This matters because the masked scores have $-\infty$ values, while $s$ holds well-behaved $(0, 1)$ probabilities.</span>

<span style="font-size: 14px;">7. **Optional renormalization.** If $\texttt{norm\_topk\_prob}$ is true, divide each token's $k$ weights by their sum plus a $10^{-20}$ epsilon so they sum to one.</span>

<span style="font-size: 14px;">8. **Scale.** Multiply by $\texttt{routed\_scaling\_factor}$ (a fixed constant in the config, typically 1.0 but can be tuned).</span>

---

## <span style="font-size: 16px;">Equations</span>

<span style="font-size: 14px;">Per-expert sigmoid score:</span>

$$
s_{t,e} = \sigma((x W_r)_{t,e}) = \frac{1}{1 + e^{-(x W_r)_{t,e}}}
$$

<span style="font-size: 14px;">Group score (top-2 sum), with group $g$ containing experts $E_g$:</span>

$$
G_{t,g} = \sum_{e \in \text{top}_2(s_{t, E_g})} s_{t,e}
$$

<span style="font-size: 14px;">Active group set per token:</span>

$$
A_t = \text{argtop}_{G'}\, G_{t, :}
$$

<span style="font-size: 14px;">Masked scores ($\mathbf{1}[\cdot]$ is an indicator):</span>

$$
\tilde{s}_{t,e} = \begin{cases} s_{t,e} & \text{if } g(e) \in A_t \\ -\infty & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">Top-k indices and gathered weights:</span>

$$
I_t = \text{argtop}_k\, \tilde{s}_{t,:}, \quad w_{t,j} = s_{t, I_{t,j}}
$$

<span style="font-size: 14px;">Normalization (optional) and scaling:</span>

$$
\hat{w}_{t,j} = \frac{w_{t,j}}{\sum_j w_{t,j} + 10^{-20}}, \quad w^{\text{out}} = \alpha \cdot \hat{w}
$$

---

## <span style="font-size: 16px;">Sigmoid vs Softmax Gates</span>

<span style="font-size: 14px;">Classical MoE routers, including the original Switch Transformer (Fedus et al., 2022) and DeepSeek-V2, use softmax over the full expert dimension before top-k. GLM-4.5, following DeepSeek-V3, uses **sigmoid** scores instead. The trade-offs:</span>

* <span style="font-size: 14px;">**Softmax couples experts.** Increasing one expert's logit decreases every other expert's score because the simplex constraint $\sum_e p_e = 1$ holds. With many experts ($E=128$ here) the probabilities become small and noisy.</span>
* <span style="font-size: 14px;">**Sigmoid decouples experts.** Each expert's affinity is independent. Top-k of sigmoid scores is identical to top-k of pre-sigmoid logits, but the gathered weights are bounded in $(0, 1)$ and easier to interpret as utilization signals.</span>
* <span style="font-size: 14px;">**Renormalization recovers a simplex.** With $\texttt{norm\_topk\_prob}=\text{True}$, the gathered sigmoid scores are renormalized to sum to one across the $k$ chosen experts. The output is then a convex combination, matching the semantics softmax gave you for free.</span>

---

## <span style="font-size: 16px;">Why Top-2 Sum Per Group?</span>

<span style="font-size: 14px;">A group could be summarized by its max score, its mean, or by summing the top few experts. GLM-4.5 uses the **sum of the top-2 expert scores** per group. The intuition:</span>

* <span style="font-size: 14px;">A single max is fragile: one outlier expert can drag a whole group in, even if every other expert in that group is weak. Once the group is chosen, the final top-k will pick that one expert and ignore the rest, wasting the group selection slot.</span>
* <span style="font-size: 14px;">Summing all $E/G$ experts dilutes signal in larger groups and over-weights groups that are uniformly mediocre over groups with strong specialists.</span>
* <span style="font-size: 14px;">Top-2 sum is a compromise: a group needs at least two reasonably strong experts to win, which aligns better with the fact that final selection will pick multiple experts per group on average.</span>

<span style="font-size: 14px;">For groups of size 1 the top-2 reduces to the single score; for groups of size 2 it equals the sum.</span>

---

## <span style="font-size: 16px;">Comparison With Other Routers</span>

* <span style="font-size: 14px;">**Switch Transformer (Fedus et al., 2022):** softmax over all experts, top-1 only, auxiliary load-balance loss. No grouping. Simple but bandwidth-heavy at scale.</span>
* <span style="font-size: 14px;">**GShard (Lepikhin et al., 2020):** softmax + top-2 with capacity factors and dropping. No grouping.</span>
* <span style="font-size: 14px;">**DeepSeek-V2:** softmax + top-k, no grouping, fine-grained experts.</span>
* <span style="font-size: 14px;">**DeepSeek-V3:** sigmoid + group routing + auxiliary-loss-free balance via a learned per-expert bias added only to the choice scores (not the gathered weights).</span>
* <span style="font-size: 14px;">**GPT-oss / gpt-oss:** softmax-after-topk, no grouping. Selects top-k by raw logits, then softmaxes only the chosen subset for weights.</span>
* <span style="font-size: 14px;">**GLM-4.5:** mirrors DeepSeek-V3's design end to end. The HF $\texttt{Glm4MoeTopkRouter}$ exposes a $\texttt{e\_score\_correction\_bias}$ buffer that is added to logits before group selection but never to the gathered weights, enabling the same auxiliary-loss-free balance trick.</span>

---

## <span style="font-size: 16px;">Auxiliary-Loss-Free Balancing</span>

<span style="font-size: 14px;">The $\texttt{e\_score\_correction\_bias}$ vector in the GLM-4.5 router serves the same role it does in DeepSeek-V3. Instead of adding a load-balance auxiliary loss to the training objective (which competes with the language modeling loss and tends to hurt quality), the bias is updated **outside** the gradient. After every step, experts with above-average load get their bias decremented and underused experts get it incremented. The bias steers the choice scores (and thus group selection and top-k indices) but never the gathered $w_{t,j}$ values, so it does not warp the convex combination passed to the experts.</span>

<span style="font-size: 14px;">In this problem the bias is fixed at zero, which is the initialization state. The training-time update rule is not part of the routing forward pass.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take 1 token, $E=8$, $G=4$, $E/G=2$, $G'=2$, $k=2$, $\texttt{norm\_topk\_prob}=\text{True}$, $\alpha=1$. Suppose the sigmoid scores are:</span>

$$
s = [0.10,\ 0.80,\ 0.30,\ 0.70,\ 0.95,\ 0.05,\ 0.20,\ 0.60]
$$

<span style="font-size: 14px;">Reshape into groups of 2:</span>

* <span style="font-size: 14px;">Group 0: $[0.10, 0.80]$, top-2 sum $= 0.90$</span>
* <span style="font-size: 14px;">Group 1: $[0.30, 0.70]$, top-2 sum $= 1.00$</span>
* <span style="font-size: 14px;">Group 2: $[0.95, 0.05]$, top-2 sum $= 1.00$</span>
* <span style="font-size: 14px;">Group 3: $[0.20, 0.60]$, top-2 sum $= 0.80$</span>

<span style="font-size: 14px;">Top-2 groups: $\{1, 2\}$ (group 0 has 0.90, just below). Mask experts $\{0, 1, 6, 7\}$ to $-\infty$. Remaining masked scores: $[-\infty, -\infty, 0.30, 0.70, 0.95, 0.05, -\infty, -\infty]$. Top-2 indices: $\{4, 3\}$. Gathered weights from $s$: $[0.95, 0.70]$. Renormalize: $[0.95/1.65, 0.70/1.65] \approx [0.576, 0.424]$.</span>

<span style="font-size: 14px;">If instead we had used a single-max group score, group 2's score would be 0.95 and group 0's would be 0.80; group 0 would lose to group 2 by the same ranking. But if expert 5 had been 0.92 instead of 0.05, group 2's max would still be 0.95 while its top-2 sum would jump to 1.87, dominating other groups for a much stronger reason. The top-2 sum reflects depth of specialization.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Per token:</span>

* <span style="font-size: 14px;">Router GEMM: $O(d \cdot E)$.</span>
* <span style="font-size: 14px;">Sigmoid: $O(E)$.</span>
* <span style="font-size: 14px;">Group reshape and per-group top-2: $O(E)$ (top-2 on a small slice is $O(E/G)$ per group, times $G$ groups).</span>
* <span style="font-size: 14px;">Top-$G'$ over $G$ group scores: $O(G \log G')$.</span>
* <span style="font-size: 14px;">Top-$k$ over masked scores: $O(E \log k)$.</span>

<span style="font-size: 14px;">Total $O(d E)$ dominated by the GEMM. Group routing adds no asymptotic overhead but cuts dispatch bandwidth roughly by $G'/G$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using softmax instead of sigmoid.** Top-k indices may still come out right when only one group is selected, but weights are wrong because softmax probabilities depend on every other expert's logit. GLM-4.5 explicitly uses sigmoid; mixing the two changes the trained distribution.</span>
* <span style="font-size: 14px;">**Skipping the group restriction.** Plain top-k over all $E$ experts ignores the bandwidth motivation and gives different indices whenever the global top-k spans more groups than $\texttt{topk\_group}$ allows.</span>
* <span style="font-size: 14px;">**Wrong group-size division.** Using $\texttt{n\_group}$ as $\texttt{experts\_per\_group}$ (or vice versa) silently produces garbage when $n\_group \ne E/n\_group$. The reshape will succeed only by coincidence on certain $E$ values, masking the bug.</span>
* <span style="font-size: 14px;">**Gathering weights from masked scores instead of sigmoid scores.** The masked tensor has $-\infty$ entries; gathering at valid indices still works, but if you accidentally apply the mask before the gather you can end up with $-\infty$ weights when $\texttt{num\_experts\_per\_tok}$ exceeds $\texttt{topk\_group} \cdot (E/G)$. The HF code gathers from the un-masked sigmoid scores precisely to avoid this trap.</span>
* <span style="font-size: 14px;">**Forgetting $\texttt{norm\_topk\_prob}$.** Without it the output weights are raw sigmoid values that do not sum to one. Downstream the expert outputs get weighted by less-than-one factors and the residual stream loses magnitude over depth.</span>
* <span style="font-size: 14px;">**Forgetting $\texttt{routed\_scaling\_factor}$.** Even when it is 1.0 in the default config, omitting the multiplication hides the bug until someone tunes the scale. The HF forward always applies it.</span>
* <span style="font-size: 14px;">**Top-1 per group instead of top-2 sum.** A group's max alone biases selection toward groups with a single hot expert. Two of the public test cases in this problem distinguish the two scoring rules.</span>
* <span style="font-size: 14px;">**Top-k sorted vs unsorted.** HF uses $\texttt{sorted=False}$ for both group top-$G'$ and expert top-$k$. The set of indices is the same either way, but downstream operations that rely on a stable order can break. Compare canonicalized (sorted-by-index) tensors when validating.</span>
* <span style="font-size: 14px;">**Numerical instability with $\texttt{norm\_topk\_prob}$.** If all $k$ chosen weights are tiny, the sum can underflow. The $10^{-20}$ epsilon in the denominator prevents division-by-zero, but the weights remain near zero. Sigmoid outputs are bounded below by their pre-activation, so the realistic floor is set by the data, not the implementation.</span>

---
