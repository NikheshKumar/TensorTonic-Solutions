# <span style="font-size: 20px;">Dense Prefix Layers before MoE</span>

<span style="font-size: 14px;">In DeepSeek V3, the first few transformer layers use a standard dense feedforward network (SwiGLU FFN) instead of the sparse Mixture-of-Experts layer used by the rest of the model. This design choice -- called the **dense prefix** -- ensures that early layers build strong, universally shared representations before any token-level routing or expert specialization begins.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The dense prefix is a simple conditional applied at every transformer layer. For each layer index $i$, the system checks whether $i$ falls below a threshold $N_{\text{dense}}$. If it does, the layer uses a single dense SwiGLU FFN as its feedforward component. If $i \geq N_{\text{dense}}$, the layer switches to the full sparse MoE feedforward block with routed experts, a shared expert, and gating logic.</span>

<span style="font-size: 14px;">This is not a learned routing decision. It is a hard-coded architectural choice made before training. There is no router, no gating score, and no gradient flowing through a selection mechanism. The layer index alone determines which feedforward architecture is used.</span>

<span style="font-size: 14px;">Every layer still has a full multi-head latent attention (MLA) block. The dense prefix only affects the feedforward sub-layer. A dense prefix layer is: attention, add-and-norm, dense SwiGLU FFN, add-and-norm. An MoE layer is: attention, add-and-norm, sparse MoE FFN (shared expert + routed experts), add-and-norm. The attention mechanism is identical in both cases.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">**The conditional.** Given total layers $L$, a dense prefix count $N_{\text{dense}}$, and layer index $i \in \{0, 1, \ldots, L-1\}$:</span>

$$\text{is\_dense}(i) = \begin{cases} \text{True} & \text{if } i < N_{\text{dense}} \\ \text{False} & \text{otherwise} \end{cases}$$

<span style="font-size: 14px;">**The dense SwiGLU FFN.** For a dense prefix layer, given input $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$:</span>

$$\text{FFN}_{\text{dense}}(\mathbf{x}) = W_{\text{down}} \cdot (\text{swish}(W_{\text{gate}} \cdot \mathbf{x}) \odot W_{\text{up}} \cdot \mathbf{x})$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$W_{\text{gate}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ is the gate projection matrix</span>
* <span style="font-size: 14px;">$W_{\text{up}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ is the up projection matrix</span>
* <span style="font-size: 14px;">$W_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ is the down projection matrix</span>
* <span style="font-size: 14px;">$\text{swish}(z) = z \cdot \sigma(z)$ where $\sigma$ is the sigmoid function</span>
* <span style="font-size: 14px;">$\odot$ denotes elementwise multiplication</span>

<span style="font-size: 14px;">**SwiGLU step by step:**</span>

<span style="font-size: 14px;">1. **Gate projection:** $\mathbf{g} = W_{\text{gate}} \cdot \mathbf{x} \in \mathbb{R}^{d_{\text{ff}}}$</span>

<span style="font-size: 14px;">2. **Swish activation:** $\mathbf{g}' = \text{swish}(\mathbf{g}) = \mathbf{g} \odot \sigma(\mathbf{g})$</span>

<span style="font-size: 14px;">3. **Up projection:** $\mathbf{u} = W_{\text{up}} \cdot \mathbf{x} \in \mathbb{R}^{d_{\text{ff}}}$</span>

<span style="font-size: 14px;">4. **Gated multiplication:** $\mathbf{h} = \mathbf{g}' \odot \mathbf{u} \in \mathbb{R}^{d_{\text{ff}}}$</span>

<span style="font-size: 14px;">5. **Down projection:** $\mathbf{y} = W_{\text{down}} \cdot \mathbf{h} \in \mathbb{R}^{d_{\text{model}}}$</span>

<span style="font-size: 14px;">**The full transformer layer with the conditional.** For layer $i$:</span>

$$\mathbf{h}_i^{\text{attn}} = \mathbf{x}_i + \text{MLA}(\text{RMSNorm}(\mathbf{x}_i))$$

$$\mathbf{h}_i^{\text{out}} = \mathbf{h}_i^{\text{attn}} + \begin{cases} \text{FFN}_{\text{dense}}(\text{RMSNorm}(\mathbf{h}_i^{\text{attn}})) & \text{if } i < N_{\text{dense}} \\ \text{MoE}(\text{RMSNorm}(\mathbf{h}_i^{\text{attn}})) & \text{otherwise} \end{cases}$$

<span style="font-size: 14px;">Both branches add their output as a residual on top of the attention output. The only difference is whether the feedforward component is a single dense FFN or the full MoE block.</span>

---

## <span style="font-size: 16px;">Why Dense Before Sparse</span>

<span style="font-size: 14px;">The decision to use dense layers at the beginning and MoE layers afterward addresses practical failure modes observed when MoE is applied too early in the network.</span>

<span style="font-size: 14px;">**Early layers build shared representations.** The first few layers perform foundational processing that every token benefits from equally: basic positional and token-identity features at layer 0, local syntactic patterns at layer 1, refined intermediate representations at layer 2. These operations are universal -- they do not depend on the "type" of a token. A dense FFN processes every token through the same full set of parameters, ensuring equal representational capacity at this critical stage.</span>

<span style="font-size: 14px;">**Premature specialization via routing hurts.** When an MoE layer routes tokens to different experts in the first layers, the router receives raw or barely-processed embeddings carrying mostly surface-level information (token identity, position). This is a poor basis for meaningful routing. Routing on shallow features leads to arbitrary expert assignment that fragments the shared representation space.</span>

<span style="font-size: 14px;">**Empirical findings support this design.** The DeepSeek team found that dense prefix layers improved training stability. When all layers use MoE, router collapse -- most tokens sent to a small subset of experts -- is more likely in early layers because the router has less signal. Deferring MoE to layer 3 gives the router richer inputs and more stable expert utilization.</span>

<span style="font-size: 14px;">**Computational cost is negligible.** With 3 dense layers out of 61 total, only about 5% use the dense path. The stability and quality gains far outweigh the marginal cost increase.</span>

---

## <span style="font-size: 16px;">The Dense Layer Architecture</span>

<span style="font-size: 14px;">The FFN in each dense prefix layer is architecturally identical to the **shared expert** in the MoE layers. Both use the same SwiGLU structure with the same hidden dimension $d_{\text{ff}}$. The only difference is context: in a dense prefix layer, this SwiGLU FFN is the entire feedforward component. In an MoE layer, the same architecture appears as the shared expert alongside routed experts.</span>

<span style="font-size: 14px;">A standard transformer FFN uses two weight matrices: $\text{FFN}(\mathbf{x}) = W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x})$. SwiGLU replaces this with three matrices. The gate projection passes through swish and elementwise-multiplies with the up projection. This gating lets the network learn which intermediate dimensions to suppress or amplify.</span>

<span style="font-size: 14px;">**Parameter count per dense prefix FFN:**</span>

* <span style="font-size: 14px;">$W_{\text{gate}}$: $d_{\text{ff}} \times d_{\text{model}}$ parameters</span>
* <span style="font-size: 14px;">$W_{\text{up}}$: $d_{\text{ff}} \times d_{\text{model}}$ parameters</span>
* <span style="font-size: 14px;">$W_{\text{down}}$: $d_{\text{model}} \times d_{\text{ff}}$ parameters</span>
* <span style="font-size: 14px;">**Total:** $3 \times d_{\text{model}} \times d_{\text{ff}}$ (no biases in DeepSeek V3)</span>

<span style="font-size: 14px;">This is exactly the same parameter count as the shared expert in every MoE layer. The dense prefix and shared experts are interchangeable architecturally -- the difference is in how they fit into the layer.</span>

---

## <span style="font-size: 16px;">Paper Context: DeepSeek V3</span>

<span style="font-size: 14px;">DeepSeek V3 is a 671-billion parameter Mixture-of-Experts language model with 37 billion active parameters per token. It has 61 transformer layers total, indexed 0 to 60.</span>

<span style="font-size: 14px;">**Dense prefix configuration:**</span>

* <span style="font-size: 14px;">$N_{\text{dense}} = 3$</span>
* <span style="font-size: 14px;">**Layers 0, 1, 2:** Dense SwiGLU FFN (no routing, no experts)</span>
* <span style="font-size: 14px;">**Layers 3 through 60:** Sparse MoE FFN (1 shared expert + 256 routed experts, top-8 routing)</span>

<span style="font-size: 14px;">**Why exactly 3?** The choice is empirical. Three dense prefix layers struck the right balance between training stability and efficiency. Fewer did not give enough depth to build robust shared representations before routing. More consumed additional compute without proportional quality gains.</span>

<span style="font-size: 14px;">**Continuity with DeepSeek V2.** The dense prefix was inherited from V2, which used the same pattern. V3 scaled MoE from 160 to 256 experts, but the dense prefix count stayed at 3 -- early dense processing is beneficial regardless of scale.</span>

<span style="font-size: 14px;">**Interaction with other components.** Every layer uses MLA for attention. The dense prefix does not affect attention. The auxiliary-loss-free load balancing, multi-token prediction, and KV compression all operate on MoE layers (3-60) and are irrelevant to dense prefix layers.</span>

<span style="font-size: 14px;">**Parameter breakdown.** The 3 dense prefix layers each contain one SwiGLU FFN. The 58 MoE layers each contain 1 shared expert + 256 routed experts. The vast majority of the 671B parameters come from MoE layers. The dense prefix contributes a tiny fraction but is critical for representation quality.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a model with $L = 8$ layers (indexed 0 through 7) and $N_{\text{dense}} = 3$. We trace through layers 0 to 5.</span>

<span style="font-size: 14px;">**Layer 0** ($i = 0$): $0 < 3$ is True. Uses the **dense SwiGLU FFN**.</span>

<span style="font-size: 14px;">**Layer 1** ($i = 1$): $1 < 3$ is True. Uses the **dense SwiGLU FFN**.</span>

<span style="font-size: 14px;">**Layer 2** ($i = 2$): $2 < 3$ is True. Uses the **dense SwiGLU FFN**.</span>

<span style="font-size: 14px;">**Layer 3** ($i = 3$): $3 < 3$ is False. Uses the **sparse MoE FFN**.</span>

<span style="font-size: 14px;">**Layer 4** ($i = 4$): $4 < 3$ is False. Uses the **sparse MoE FFN**.</span>

<span style="font-size: 14px;">**Layer 5** ($i = 5$): $5 < 3$ is False. Uses the **sparse MoE FFN**.</span>

<span style="font-size: 14px;">Now trace a concrete forward pass through Layer 0 (dense) with $d_{\text{model}} = 4$, $d_{\text{ff}} = 6$.</span>

<span style="font-size: 14px;">**Input** (after attention and RMSNorm): $\mathbf{x} = [1.0, -0.5, 0.8, 0.2]$</span>

<span style="font-size: 14px;">**Step 1: Gate projection.** $\mathbf{g} = W_{\text{gate}} \cdot \mathbf{x} = [0.7, -1.2, 0.3, 0.9, -0.4, 1.5]$</span>

<span style="font-size: 14px;">**Step 2: Swish activation.** Apply $\text{swish}(g_j) = g_j \cdot \sigma(g_j)$ elementwise. For instance, $\text{swish}(0.7) = 0.7 \times 0.668 = 0.468$ and $\text{swish}(-1.2) = -1.2 \times 0.232 = -0.278$:</span>

$$\mathbf{g}' = [0.468, -0.278, 0.172, 0.640, -0.160, 1.227]$$

<span style="font-size: 14px;">**Step 3: Up projection.** $\mathbf{u} = W_{\text{up}} \cdot \mathbf{x} = [0.5, 0.9, -0.3, 1.1, 0.7, -0.6]$</span>

<span style="font-size: 14px;">**Step 4: Gated multiplication.** $\mathbf{h} = \mathbf{g}' \odot \mathbf{u}$ elementwise:</span>

$$\mathbf{h} = [0.234, -0.250, -0.052, 0.704, -0.112, -0.736]$$

<span style="font-size: 14px;">**Step 5: Down projection.** $\mathbf{y} = W_{\text{down}} \cdot \mathbf{h}$, mapping $\mathbb{R}^6 \to \mathbb{R}^4$:</span>

$$\mathbf{y} = [0.31, -0.18, 0.42, -0.05]$$

<span style="font-size: 14px;">**Step 6: Residual connection.** Final output: $\mathbf{x} + \mathbf{y} = [1.31, -0.68, 1.22, 0.15]$.</span>

<span style="font-size: 14px;">This output passes to Layer 1, which is also dense, so the same SwiGLU process repeats with Layer 1's own weights. After Layer 2 finishes, the output enters Layer 3, where the MoE router takes over.</span>

<span style="font-size: 14px;">**Contrast with Layer 3 (MoE).** At Layer 3, the model would: (a) compute router scores over 256 experts, (b) select top-8, (c) compute each expert's SwiGLU output, (d) weight and sum them, and (e) add the shared expert's output. The dense prefix layer skips all this -- one FFN, every token, no branching.</span>

---

## <span style="font-size: 16px;">Connection to Shared Expert</span>

<span style="font-size: 14px;">The dense prefix FFN and the shared expert are architecturally the same thing -- a SwiGLU FFN with identical structure and dimensions. The difference is entirely contextual.</span>

<span style="font-size: 14px;">**In a dense prefix layer (layers 0-2):**</span>

* <span style="font-size: 14px;">The SwiGLU FFN is the **sole** feedforward component</span>
* <span style="font-size: 14px;">No router, no routed experts, no gating</span>
* <span style="font-size: 14px;">Every token passes through the same FFN with the same weights</span>
* <span style="font-size: 14px;">Output = $\text{FFN}_{\text{dense}}(\mathbf{x})$</span>

<span style="font-size: 14px;">**In an MoE layer (layers 3-60):**</span>

* <span style="font-size: 14px;">The shared expert (same architecture) runs on every token unconditionally</span>
* <span style="font-size: 14px;">8 out of 256 routed experts are additionally selected per token</span>
* <span style="font-size: 14px;">Output = $\text{FFN}_{\text{shared}}(\mathbf{x}) + \sum_{i \in \mathcal{S}} w_i \cdot \text{FFN}_{\text{expert}_i}(\mathbf{x})$</span>

<span style="font-size: 14px;">A dense prefix layer is essentially a degenerate MoE layer where the number of routed experts is zero. The same SwiGLU class can be reused for both roles, with the layer index determining whether routing logic wraps around it.</span>

<span style="font-size: 14px;">**Why not just use MoE everywhere with a shared expert?** Even though the shared expert provides universal processing, the routed experts still introduce noise during early training. Poorly-initialized routing decisions destabilize shared representations. Dense prefix layers remove this noise entirely for the first 3 layers.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">Several mistakes commonly arise when implementing dense prefix layers.</span>

* <span style="font-size: 14px;">**Off-by-one in layer index.** The condition is $i < N_{\text{dense}}$, not $i \leq N_{\text{dense}}$. With $N_{\text{dense}} = 3$, layers 0, 1, 2 are dense (three total). Using $\leq$ would make layer 3 dense as well, giving four dense layers. Classic fence-post error.</span>

* <span style="font-size: 14px;">**Using MoE weights for dense layers.** Each dense prefix layer has its own SwiGLU weights ($W_{\text{gate}}$, $W_{\text{up}}$, $W_{\text{down}}$). These are separate from the shared expert weights in any MoE layer. Even though the architecture is identical, the weights are not shared across layers.</span>

* <span style="font-size: 14px;">**Forgetting that dense layers still have attention.** The dense prefix only replaces the feedforward sub-layer. The attention sub-layer (MLA) is still present. A dense prefix layer is not "just an FFN" -- it is a full transformer layer with attention followed by a dense FFN.</span>

* <span style="font-size: 14px;">**Confusing dense prefix with shared expert.** They share the same SwiGLU architecture but serve different roles. The dense prefix FFN is the entire feedforward component. The shared expert runs alongside routed experts. Sharing weight tensors between them would be a bug.</span>

* <span style="font-size: 14px;">**Applying load balancing to dense layers.** The auxiliary-loss-free load balancing applies only to MoE layers. Dense prefix layers have no router and no experts. Including them in balancing computations produces errors.</span>

* <span style="font-size: 14px;">**Wrong layer count arithmetic.** With $L = 61$ and $N_{\text{dense}} = 3$: 3 dense, 58 MoE. The count is $L - N_{\text{dense}} = 58$, not 57. Mistakes propagate into memory estimates and parallelism configs.</span>

* <span style="font-size: 14px;">**Assuming dense prefix is a separate module.** Both dense and MoE layers are the same transformer layer class with a conditional branch. They share attention, normalization, and residual logic. The only difference is a single `if` statement selecting which FFN to call.</span>

---