# <span style="font-size: 20px;">SMEBU Load Balancing</span>

<span style="font-size: 14px;">Soft-clamped Momentum Expert Bias Update (SMEBU) is a load balancing mechanism for Mixture-of-Experts (MoE) models. It dynamically adjusts per-expert bias terms to distribute tokens evenly across experts, without introducing any auxiliary loss into the training objective. SMEBU was introduced as part of the Arcee Trinity architecture, where it works alongside sigmoid gating and coarse-grained expert routing.</span>

<span style="font-size: 14px;">Unlike auxiliary-loss-based approaches, SMEBU operates entirely outside the gradient computation graph. It uses a momentum-based update rule with a tanh soft clamp to adjust expert biases, making it lightweight, stable, and easy to tune.</span>

---

## <span style="font-size: 16px;">What It Is and What It Does</span>

<span style="font-size: 14px;">In MoE models, a router network decides which expert(s) each token is sent to. In practice, routers tend to collapse: a few experts receive the vast majority of tokens while the rest starve. The router learns to favor whichever experts currently perform best, giving those experts more training signal, making them perform even better. This is a rich-get-richer feedback loop.</span>

<span style="font-size: 14px;">Expert collapse wastes parameters (starved experts never learn), reduces effective capacity (model behaves like it has far fewer experts), and causes training instability (overloaded experts get gradient explosions, underloaded ones stall).</span>

<span style="font-size: 14px;">SMEBU solves this by maintaining a bias term for each expert, added to the router logits before the routing decision. Overloaded experts get decreased biases (less attractive to router), starving experts get increased biases (more attractive). These updates happen outside backpropagation -- they are not learned parameters but adjusted by a simple rule based on observed loads.</span>

<span style="font-size: 14px;">The update cycle each training step:</span>

* <span style="font-size: 14px;">**Observe:** Count how many tokens each expert received.</span>
* <span style="font-size: 14px;">**Compare:** Compute each expert's deviation from the target load.</span>
* <span style="font-size: 14px;">**Update:** Adjust each bias via momentum-based rule.</span>
* <span style="font-size: 14px;">**Clamp:** Apply tanh soft clamp to keep biases bounded.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">SMEBU's update rule has four components applied in sequence for each expert i.</span>

<span style="font-size: 14px;">**Imbalance computation.** Measures how far each expert's current load is from the target:</span>

$$
\text{imbalance}_i = \text{expert\_load}_i - \text{target\_load}
$$

<span style="font-size: 14px;">A positive imbalance means overloaded; negative means underloaded. The target_load is typically total_tokens / num_experts.</span>

<span style="font-size: 14px;">**Gradient computation.** The imbalance is normalized by total tokens for scale invariance:</span>

$$
g_i = \frac{\text{imbalance}_i}{\text{total\_tokens} + \epsilon}
$$

<span style="font-size: 14px;">The epsilon (e.g., 1e-8) prevents division by zero. This normalization makes hyperparameters transferable across batch sizes.</span>

<span style="font-size: 14px;">**Momentum update.** The bias is updated similarly to SGD with momentum:</span>

$$
b_i^{\text{new}} = \mu \cdot b_i^{\text{old}} - \eta \cdot g_i
$$

<span style="font-size: 14px;">Here mu is the momentum coefficient (0.9 to 0.99) and eta is the bias learning rate. Momentum smooths out noisy per-batch load fluctuations. The negative sign means a positive imbalance (overloaded) decreases the bias.</span>

<span style="font-size: 14px;">**Tanh soft clamp.** The updated bias is soft-clamped to stay bounded:</span>

$$
b_i^{\text{new}} = b_{\max} \cdot \tanh\!\left(\frac{b_i^{\text{new}}}{b_{\max}}\right)
$$

<span style="font-size: 14px;">This keeps biases within (-b_max, +b_max) using a smooth function that gradually reduces update magnitude near the boundary, avoiding hard-clamp discontinuities.</span>

<span style="font-size: 14px;">**Full update cycle in pseudocode** for each expert i in {1, ..., N}:</span>

* <span style="font-size: 14px;">**Step 1:** imbalance_i = expert_load_i - (total_tokens / N)</span>
* <span style="font-size: 14px;">**Step 2:** g_i = imbalance_i / (total_tokens + epsilon)</span>
* <span style="font-size: 14px;">**Step 3:** b_i = mu * b_i - eta * g_i</span>
* <span style="font-size: 14px;">**Step 4:** b_i = b_max * tanh(b_i / b_max)</span>

<span style="font-size: 14px;">After these updates, biases are added to router logits on the next forward pass.</span>

---

## <span style="font-size: 16px;">Why Not Auxiliary Loss</span>

<span style="font-size: 14px;">The standard approach to MoE load balancing is adding an auxiliary loss term to the training objective. The Switch Transformer popularized this: penalizing uneven expert utilization with a combined loss:</span>

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + \alpha \cdot \mathcal{L}_{\text{aux}}
$$

<span style="font-size: 14px;">While effective, this approach has well-known problems:</span>

* <span style="font-size: 14px;">**Interference with training:** The auxiliary loss competes with the main language modeling objective. The router must simultaneously learn good routing and satisfy the balancing constraint, and these goals can conflict.</span>
* <span style="font-size: 14px;">**Tuning difficulty:** The coefficient alpha is notoriously hard to set. Too small and expert collapse occurs. Too large and it forces artificially uniform routing at the expense of model quality. The optimal value often changes during training.</span>
* <span style="font-size: 14px;">**Gradient pollution:** The auxiliary loss flows through the same computation graph as the main loss, distorting router gradients in subtle ways that can slow convergence.</span>

<span style="font-size: 14px;">SMEBU sidesteps all of these. Because it operates outside the gradient graph, it has zero interaction with the main training objective. The main loss function sees no auxiliary term and experiences no gradient pollution. The router learns purely from the language modeling signal while the biases shift decision boundaries as a post-hoc correction.</span>

<span style="font-size: 14px;">This separation of concerns is SMEBU's core advantage: the training loss optimizes for model quality, the bias mechanism optimizes for balance, and neither interferes with the other.</span>

---

## <span style="font-size: 16px;">The Soft Clamp</span>

<span style="font-size: 14px;">Without clamping, biases could grow without bound. If one expert is consistently underloaded, its bias would keep increasing forever, eventually producing extreme logit values that destabilize routing.</span>

<span style="font-size: 14px;">A hard clamp (clipping to [-b_max, +b_max]) introduces a discontinuity at the boundary. When a bias hits the hard limit, its effective update drops to exactly zero in one step. This creates oscillatory behavior: the bias hits the wall, stops updating, drifts back, then slams into the wall again.</span>

<span style="font-size: 14px;">The tanh soft clamp avoids this by gradually reducing update magnitude as the bias approaches the boundary:</span>

* <span style="font-size: 14px;">**Near zero:** tanh(x) approximates x, so the clamp has almost no effect. Small biases update freely.</span>
* <span style="font-size: 14px;">**Near the boundary:** tanh(x) saturates toward +/-1, compressing large updates. A bias near b_max barely changes even with a large update signal.</span>
* <span style="font-size: 14px;">**Smooth transition:** No sudden cutoff. Compression increases continuously as the bias grows.</span>

<span style="font-size: 14px;">The b_max parameter controls the range. A larger b_max allows stronger bias corrections for severe imbalances. A smaller b_max limits influence, providing stability but possibly insufficient correction power.</span>

<span style="font-size: 14px;">Mathematically, the derivative is sech^2(x / b_max), which smoothly decreases from 1 at x=0 toward 0 as x grows. Updates are never abruptly killed, only gradually attenuated.</span>

---

## <span style="font-size: 16px;">Paper Context: SMEBU in Arcee Trinity</span>

<span style="font-size: 14px;">Arcee Trinity is an MoE language model that combines several routing innovations. SMEBU is one piece of this design, working alongside other components for stable expert utilization.</span>

<span style="font-size: 14px;">Key architectural choices that interact with SMEBU:</span>

* <span style="font-size: 14px;">**Sigmoid routing:** Instead of softmax routing where expert probabilities sum to one, Trinity uses independent sigmoid gates. Each expert's gate outputs a probability between 0 and 1, independent of others. This allows multi-expert routing without the competition inherent in softmax. SMEBU biases are added to router logits before the sigmoid activation.</span>
* <span style="font-size: 14px;">**Coarse-grained MoE:** Trinity uses fewer but larger experts rather than many small ones. Each expert has significant capacity, making balanced utilization even more important since an idle large expert represents a larger waste of parameters.</span>
* <span style="font-size: 14px;">**No auxiliary loss:** Trinity explicitly removes the auxiliary load-balancing loss, relying entirely on SMEBU. This simplifies training and eliminates auxiliary loss coefficient tuning.</span>

<span style="font-size: 14px;">Sigmoid routing and SMEBU are a natural pairing. With sigmoid gates, each expert's routing decision is independent, so adjusting one expert's bias does not affect others' probabilities (unlike softmax). This makes bias updates more predictable and easier to reason about.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider 4 experts processing 1000 tokens. Target load per expert: 1000 / 4 = 250. All biases start at zero. Observed loads after one forward pass:</span>

* <span style="font-size: 14px;">**Expert 0:** 400 tokens (overloaded)</span>
* <span style="font-size: 14px;">**Expert 1:** 350 tokens (overloaded)</span>
* <span style="font-size: 14px;">**Expert 2:** 150 tokens (underloaded)</span>
* <span style="font-size: 14px;">**Expert 3:** 100 tokens (underloaded)</span>

<span style="font-size: 14px;">Using mu = 0.9, eta = 0.1, b_max = 1.0, epsilon = 1e-8.</span>

<span style="font-size: 14px;">**Step 1: Imbalances.**</span>

* <span style="font-size: 14px;">Expert 0: 400 - 250 = +150</span>
* <span style="font-size: 14px;">Expert 1: 350 - 250 = +100</span>
* <span style="font-size: 14px;">Expert 2: 150 - 250 = -100</span>
* <span style="font-size: 14px;">Expert 3: 100 - 250 = -150</span>

<span style="font-size: 14px;">**Step 2: Gradients.**</span>

* <span style="font-size: 14px;">Expert 0: 150 / 1000 = 0.15</span>
* <span style="font-size: 14px;">Expert 1: 100 / 1000 = 0.10</span>
* <span style="font-size: 14px;">Expert 2: -100 / 1000 = -0.10</span>
* <span style="font-size: 14px;">Expert 3: -150 / 1000 = -0.15</span>

<span style="font-size: 14px;">**Step 3: Momentum update.** With biases starting at 0:</span>

* <span style="font-size: 14px;">Expert 0: 0.9 * 0 - 0.1 * 0.15 = -0.015</span>
* <span style="font-size: 14px;">Expert 1: 0.9 * 0 - 0.1 * 0.10 = -0.010</span>
* <span style="font-size: 14px;">Expert 2: 0.9 * 0 - 0.1 * (-0.10) = +0.010</span>
* <span style="font-size: 14px;">Expert 3: 0.9 * 0 - 0.1 * (-0.15) = +0.015</span>

<span style="font-size: 14px;">**Step 4: Soft clamp.** With b_max = 1.0, tanh is nearly linear near zero:</span>

* <span style="font-size: 14px;">Expert 0: 1.0 * tanh(-0.015) = -0.01500</span>
* <span style="font-size: 14px;">Expert 1: 1.0 * tanh(-0.010) = -0.01000</span>
* <span style="font-size: 14px;">Expert 2: 1.0 * tanh(+0.010) = +0.01000</span>
* <span style="font-size: 14px;">Expert 3: 1.0 * tanh(+0.015) = +0.01500</span>

<span style="font-size: 14px;">**Result:** Overloaded experts (0, 1) get negative biases making them less attractive to the router. Underloaded experts (2, 3) get positive biases making them more attractive. The effect is small after one step but accumulates over training.</span>

<span style="font-size: 14px;">**After 20 steps** (assuming the same load pattern), momentum accumulates. For Expert 0, each step adds -0.015 in gradient and retains 0.9 of the previous bias, approaching approximately -0.15, large enough to noticeably redirect tokens.</span>

<span style="font-size: 14px;">**Soft clamp at larger values:** If Expert 0's pre-clamp bias reaches -0.8, the soft clamp gives 1.0 * tanh(-0.8) = -0.664 instead of -0.8. The compression progressively dampens further updates, preventing runaway values.</span>

---

## <span style="font-size: 16px;">Hyperparameters</span>

<span style="font-size: 14px;">SMEBU has four key hyperparameters controlling different aspects of balancing behavior.</span>

<span style="font-size: 14px;">**Momentum (mu).** Controls how much previous bias is retained. Typical range: 0.9 to 0.99.</span>

* <span style="font-size: 14px;">**High mu (0.99):** Bias changes slowly, heavily influenced by history. Good for stable environments, slow to respond to sudden routing shifts.</span>
* <span style="font-size: 14px;">**Low mu (0.8):** Responds quickly to recent observations. Adapts to rapid changes but more susceptible to batch-to-batch noise.</span>
* <span style="font-size: 14px;">**mu = 0:** No momentum. Updates based purely on current batch, very noisy. Generally not recommended.</span>

<span style="font-size: 14px;">**Learning rate (eta).** Controls update step size. Typical range: 0.01 to 0.5.</span>

* <span style="font-size: 14px;">**High eta (0.5):** Aggressive corrections. Can fix severe imbalances quickly but risks oscillation between overloaded and underloaded states.</span>
* <span style="font-size: 14px;">**Low eta (0.01):** Gentle corrections. Stable but may fail to keep up with rapidly shifting loads.</span>

<span style="font-size: 14px;">**Bias bound (b_max).** Maximum bias magnitude. Typical range: 0.5 to 5.0.</span>

* <span style="font-size: 14px;">**Large b_max (5.0):** Allows strong corrections but very large biases can overwhelm learned routing, effectively overriding the router.</span>
* <span style="font-size: 14px;">**Small b_max (0.5):** Gentle nudge toward balance. May be insufficient if the router develops strong expert preferences.</span>

<span style="font-size: 14px;">**Target load.** Usually total_tokens / num_experts for uniform balance. Some designs may target non-uniform loads for experts of different sizes, but Arcee Trinity uses uniform targets.</span>

<span style="font-size: 14px;">**Interaction between hyperparameters.** High momentum with high learning rate causes overshooting: bias accumulates too much momentum and swings past balance. A good starting point is mu=0.9 with eta=0.1.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Setting eta too high.** Causes bias updates to overshoot. An overloaded expert gets a large negative bias, becomes underloaded, then gets a large positive bias, and oscillates. This is worse than the original imbalance because the router never sees a stable bias landscape to learn from.</span>
* <span style="font-size: 14px;">**Forgetting momentum.** With mu = 0, updates become extremely noisy. Expert loads vary batch to batch due to natural input variation. Without momentum to smooth these fluctuations, biases jitter randomly rather than converging on stable corrections.</span>
* <span style="font-size: 14px;">**b_max too small.** If the router strongly prefers Expert 0 by a logit margin of 2.0 but b_max is only 0.5, the maximum bias correction cannot overcome the router's preference. The expert remains overloaded despite SMEBU's efforts.</span>
* <span style="font-size: 14px;">**Not updating frequently enough.** If biases update only every K steps with K too large, imbalance worsens between updates. By the time correction arrives, the routing landscape may have shifted. Updating every step is standard; every 2-5 steps is acceptable.</span>
* <span style="font-size: 14px;">**Confusing SMEBU bias with router weight bias.** The router may have its own bias in linear layers, learned via backpropagation. SMEBU biases are separate: external logit adjustments updated by the SMEBU rule, not gradient descent. Applying weight decay to them undermines the design.</span>
* <span style="font-size: 14px;">**Ignoring initialization.** SMEBU biases should start at zero. Nonzero initialization creates artificial imbalance the mechanism must correct, wasting early training.</span>

---