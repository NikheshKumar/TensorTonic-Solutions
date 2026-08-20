# <span style="font-size: 20px;">PPO with Reference-Policy KL</span>

<span style="font-size: 14px;">This is the **RL fine-tuning** stage of RLHF: the policy is optimized with Proximal Policy Optimization (PPO, Schulman et al., 2017) to maximize a learned reward, while a per-token KL penalty against a frozen reference policy keeps it from drifting away from the supervised model. This is the algorithm InstructGPT (Ouyang et al., 2022) used to turn GPT-3 into a helpful instruction follower.</span>

---

## <span style="font-size: 16px;">Where This Sits in the RLHF Pipeline</span>

<span style="font-size: 14px;">RLHF has three stages: supervised fine-tuning (SFT), reward modeling, and RL fine-tuning. This problem is the third stage. The reward model from the previous stage scores completions, and PPO updates the policy to produce higher-reward completions while a KL term anchors it to the SFT model $\pi_{ref}$.</span>

* <span style="font-size: 14px;">**Policy $\pi_\theta$:** the language model being trained, treated as a stochastic policy where the state is the prompt plus tokens generated so far and the action is the next token.</span>
* <span style="font-size: 14px;">**Reference $\pi_{ref}$:** a frozen copy of the SFT model. It never updates and defines the behavior the policy should not stray too far from.</span>
* <span style="font-size: 14px;">**Reward $r_\phi$:** the Bradley-Terry reward model, applied once at the end of a completion to score the whole response.</span>

<span style="font-size: 14px;">Preference-optimization alternatives such as DPO, GRPO, and RLOO all aim to simplify or remove parts of this exact loop. DPO removes the RL entirely, while GRPO and RLOO keep the on-policy sampling but drop the learned value critic that standard PPO requires.</span>

---

## <span style="font-size: 16px;">Token-Level MDP Formulation</span>

<span style="font-size: 14px;">Generation is cast as a token-level Markov decision process. At step $t$ the state $s_t$ is the prompt concatenated with the tokens generated so far, the action $a_t$ is the next token, and the policy probability is $\pi_\theta(a_t \mid s_t)$. The episode ends when the model emits an end-of-sequence token. The reward model produces a single scalar at the end, so most per-token rewards are zero and the terminal token carries the reward signal, with the KL penalty distributed across all tokens.</span>

<span style="font-size: 14px;">This framing has two consequences that shape the whole algorithm. First, because the environment reward is **terminal and sparse**, credit assignment across the hundreds of tokens in a completion is hard, which is why an advantage estimator and value baseline matter so much. Second, the action space is the entire vocabulary, often 50,000 to 150,000 tokens, so the policy is enormously high-dimensional and tiny per-step probability shifts accumulate over a long sequence. The clipping and KL anchoring exist precisely to keep these accumulated shifts controlled.</span>

---

## <span style="font-size: 16px;">The Objective</span>

<span style="font-size: 14px;">The loss combines the clipped PPO surrogate with the KL penalty. As a loss to minimize over a trajectory of length $T$:</span>

$$
L(\theta) = -\frac{1}{T} \sum_{t=0}^{T-1} \min\!\left(r_t\, A_t,\; \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\, A_t\right) + \beta \cdot \frac{1}{T} \sum_{t=0}^{T-1} \left(\log\pi_\theta(a_t \mid s_t) - \log\pi_{\text{ref}}(a_t \mid s_t)\right)
$$

<span style="font-size: 14px;">where the **probability ratio** $r_t = \exp(\log\pi_\theta(a_t \mid s_t) - \log\pi_{\theta_{\text{old}}}(a_t \mid s_t))$ compares the current policy to the policy that generated the data, $A_t$ is the advantage estimate, $\epsilon$ is the clip range (typically 0.2), and $\beta$ is the KL coefficient.</span>

<span style="font-size: 14px;">The leading minus sign turns the PPO objective, which is normally maximized, into a loss. The KL term keeps its sign because it is a penalty that should be minimized, so it pulls the policy toward the reference.</span>

---

## <span style="font-size: 16px;">The Clipped Surrogate</span>

<span style="font-size: 14px;">The surrogate is the heart of PPO. Vanilla policy gradient maximizes $r_t A_t$, but if the policy moves too far in one update, the importance ratio $r_t$ explodes and the update becomes unreliable because the data was collected under $\pi_{\theta_{\text{old}}}$. PPO clips $r_t$ into $[1-\epsilon, 1+\epsilon]$ and takes the minimum of the clipped and unclipped terms:</span>

* <span style="font-size: 14px;">**Positive advantage** ($A_t > 0$, the action was good): the objective is capped at $(1+\epsilon)A_t$, so once the ratio exceeds $1+\epsilon$ there is no further incentive to increase the probability. This prevents over-committing to a single good action.</span>
* <span style="font-size: 14px;">**Negative advantage** ($A_t < 0$, the action was bad): the min picks the more negative branch, so the objective is bounded by $(1-\epsilon)A_t$, limiting how aggressively probability is pushed down.</span>
* <span style="font-size: 14px;">**Why min and not clip alone:** taking the minimum makes the clipped objective a pessimistic lower bound on the true improvement, which is what gives PPO its monotonic-improvement flavor without TRPO's expensive second-order trust region.</span>
* <span style="font-size: 14px;">**No gradient in the clipped region:** when the ratio is clipped, the term becomes a constant in $\theta$ and its gradient is zero. So a token whose probability has already moved far enough simply stops receiving updates, an implicit early-stopping per token rather than a hard constraint on the whole update.</span>

---

## <span style="font-size: 16px;">The KL Penalty and Reward Hacking</span>

<span style="font-size: 14px;">The reward model is an imperfect proxy for human preference. Left unconstrained, PPO will find adversarial completions that score highly under the reward model but are degenerate to humans: repetitive text, overly long answers, or sycophantic phrasing. This is **reward hacking** or **reward over-optimization**. The KL penalty against $\pi_{ref}$ is the primary defense.</span>

<span style="font-size: 14px;">The full per-token reward optimized by PPO is the reward-model score minus the KL term:</span>

$$
R_t = r_\phi(x, y)\,\mathbb{1}[t = T-1] - \beta\left(\log\pi_\theta(a_t \mid s_t) - \log\pi_{\text{ref}}(a_t \mid s_t)\right)
$$

<span style="font-size: 14px;">The KL term acts as a dense per-token shaping reward that charges the policy whenever it assigns a token a very different probability than the reference would. This keeps generations fluent and on-distribution, since the SFT model already produces coherent language. Tuning $\beta$ trades off reward against faithfulness: too small and the model hacks the reward, too large and it never improves beyond SFT.</span>

<span style="font-size: 14px;">There is a deeper justification. The KL-regularized objective $\mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)] - \beta\,\text{KL}(\pi_\theta \| \pi_{ref})$ has a known optimal solution, the reward-tilted reference distribution:</span>

$$
\pi^{*}(y \mid x) = \frac{1}{Z(x)}\,\pi_{ref}(y \mid x)\,\exp\!\left(\tfrac{1}{\beta} r_\phi(x, y)\right)
$$

<span style="font-size: 14px;">So the policy PPO is converging to is not "whatever maximizes reward" but the reference reweighted by exponentiated reward. As $\beta \to 0$ this collapses onto the single highest-reward completion (full reward hacking), and as $\beta \to \infty$ it returns to $\pi_{ref}$. This same closed form is what DPO later exploited to skip the RL loop, so understanding it here directly motivates the next problems in this section.</span>

---

## <span style="font-size: 16px;">The KL Estimator</span>

<span style="font-size: 14px;">The exact KL divergence between $\pi_\theta$ and $\pi_{ref}$ requires summing over the full vocabulary at every step. RLHF instead uses the **plug-in estimator** on the single token actually sampled:</span>

$$
\widehat{\text{KL}}_t = \log\pi_\theta(a_t \mid s_t) - \log\pi_{\text{ref}}(a_t \mid s_t)
$$

<span style="font-size: 14px;">This is an unbiased one-sample estimate of the true KL when tokens are drawn from $\pi_\theta$, because $\mathbb{E}_{a \sim \pi_\theta}[\log\pi_\theta - \log\pi_{ref}] = \text{KL}(\pi_\theta \| \pi_{ref})$. It is cheap because the log-probabilities are already computed during the forward pass. Some implementations use the lower-variance estimator $\pi_{ref}/\pi_\theta - 1 - \log(\pi_{ref}/\pi_\theta)$, which is always non-negative, but the simple log-ratio is what this problem specifies.</span>

---

## <span style="font-size: 16px;">Advantage Estimation</span>

<span style="font-size: 14px;">PPO needs advantages $A_t$, which measure how much better an action was than the value baseline. Standard RLHF computes them with Generalized Advantage Estimation (GAE), which requires a learned **value function** (critic). The critic is usually a second head on the policy or a separate network of comparable size, doubling the memory footprint. This critic is exactly what GRPO and RLOO eliminate by replacing the learned baseline with a group-relative or leave-one-out baseline computed from sampled rewards.</span>

<span style="font-size: 14px;">GAE blends multi-step returns with a discount $\gamma$ and a trace decay $\lambda$:</span>

$$
A_t = \sum_{l=0}^{T-1-t} (\gamma\lambda)^l\,\delta_{t+l}, \qquad \delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)
$$

<span style="font-size: 14px;">where $\delta_t$ is the temporal-difference residual and $V$ is the critic. In RLHF $\gamma$ is often set near 1 because completions are short relative to the sparse terminal reward, and $\lambda$ trades bias against variance. The subtraction of the baseline $V(s_t)$ is what reduces variance without introducing bias: any state-dependent baseline leaves the policy-gradient expectation unchanged. The whole engineering cost of training and storing $V$ is what motivates the baseline tricks in GRPO and RLOO.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $T = 2$ tokens, $\epsilon = 0.2$, $\beta = 0.1$. Token 0: $\log\pi_\theta = -0.4$, $\log\pi_{\theta_{\text{old}}} = -0.5$, $\log\pi_{\text{ref}} = -0.6$, $A_0 = 1.0$. Token 1: $\log\pi_\theta = -1.0$, $\log\pi_{\theta_{\text{old}}} = -0.8$, $\log\pi_{\text{ref}} = -0.9$, $A_1 = -0.5$.</span>

<span style="font-size: 14px;">1. **Ratios**: $r_0 = e^{-0.4 - (-0.5)} = e^{0.1} \approx 1.1052$, $r_1 = e^{-1.0 - (-0.8)} = e^{-0.2} \approx 0.8187$.</span>

<span style="font-size: 14px;">2. **Surrogate token 0** ($A_0 > 0$): unclipped $1.1052 \times 1.0 = 1.1052$, clipped to $1.2$ gives $1.2 \times 1.0 = 1.2$, min is $1.1052$.</span>

<span style="font-size: 14px;">3. **Surrogate token 1** ($A_1 < 0$): unclipped $0.8187 \times -0.5 = -0.4094$, clip $0.8187$ stays in $[0.8, 1.2]$ so clipped term is also $-0.4094$, min is $-0.4094$.</span>

<span style="font-size: 14px;">4. **Surrogate loss**: $-\tfrac{1}{2}(1.1052 + (-0.4094)) = -\tfrac{1}{2}(0.6958) = -0.3479$.</span>

<span style="font-size: 14px;">5. **KL term**: token 0 log-ratio $-0.4 - (-0.6) = 0.2$, token 1 $-1.0 - (-0.9) = -0.1$. Mean $= 0.05$, times $\beta = 0.1$ gives $0.005$.</span>

<span style="font-size: 14px;">6. **Total**: $L = -0.3479 + 0.005 = -0.3429$, rounded to $-0.3429$.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

* <span style="font-size: 14px;">**Adaptive KL:** InstructGPT used an adaptive $\beta$ controller that raised or lowered the coefficient to hit a target KL, rather than a fixed value, keeping the policy in a stable band.</span>
* <span style="font-size: 14px;">**KL in reward vs loss:** the KL can be folded into the per-token reward (shaping) or added directly to the loss. Both appear in practice; this problem adds it explicitly to the loss.</span>
* <span style="font-size: 14px;">**Critic-free successors:** GRPO (DeepSeek) and RLOO (Ahmadian et al., 2024) keep on-policy sampling and the KL anchor but replace GAE and the value critic with cheaper baselines, cutting memory and engineering complexity roughly in half.</span>
* <span style="font-size: 14px;">**DPO:** Rafailov et al. (2023) showed the KL-regularized reward objective has a closed-form optimal policy, letting them skip the RL loop entirely and train directly on preference pairs.</span>

---

## <span style="font-size: 16px;">Engineering Realities</span>

<span style="font-size: 14px;">PPO-based RLHF is notoriously fiddly, which is a large part of why the field moved toward simpler alternatives. A standard implementation holds four model copies in memory at once: the policy, the frozen reference, the reward model, and the value critic. For a large model this can quadruple the memory budget over plain fine-tuning.</span>

* <span style="font-size: 14px;">**Multiple inner epochs:** PPO reuses each rollout for several gradient steps, which is why $\pi_{\theta_{\text{old}}}$ diverges from $\pi_\theta$ and the importance ratio $r_t$ deviates from 1 within a batch. The clip range bounds this drift.</span>
* <span style="font-size: 14px;">**Whitening advantages:** advantages are typically normalized to zero mean and unit variance within a batch before the surrogate, stabilizing the gradient scale across prompts of very different reward magnitudes.</span>
* <span style="font-size: 14px;">**Value clipping:** the critic loss often uses its own clipped objective mirroring the policy clip, preventing large value updates that would destabilize advantage estimates on the next iteration.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing the three policies.** $\pi_\theta$ is current, $\pi_{\theta_{\text{old}}}$ generated the rollout and defines the ratio, and $\pi_{ref}$ is the frozen SFT anchor for the KL. Using $\pi_{\theta_{\text{old}}}$ where $\pi_{ref}$ belongs, or vice versa, silently breaks either the trust region or the drift constraint.</span>
* <span style="font-size: 14px;">**Sign of the KL term.** The surrogate is negated to become a loss, but the KL penalty is added as-is. Negating the KL too turns the anchor into a repulsion that drives the policy away from the reference and straight into reward hacking.</span>
* <span style="font-size: 14px;">**The min clip on negative advantages.** Many implementations clip and forget the min, which only behaves correctly for positive advantages. For negative advantages the min selects the unclipped term when the ratio drops below $1-\epsilon$, and dropping the min removes that protection.</span>
* <span style="font-size: 14px;">**KL collapse from too-large $\beta$.** If $\beta$ dominates, the policy never moves off the SFT distribution and reward stays flat. If $\beta$ is too small, KL explodes and outputs degenerate. The coefficient must be tuned or adapted to a target KL.</span>

---