# <span style="font-size: 20px;">GRPO Group-Relative Advantage</span>

<span style="font-size: 14px;">Group Relative Policy Optimization (GRPO), introduced in DeepSeekMath (Shao et al., 2024) and central to DeepSeek-R1, is a critic-free variant of PPO for RLHF. Instead of training a value network to estimate a baseline, GRPO samples a **group** of completions per prompt and standardizes their rewards within the group, using the resulting z-score directly as the advantage.</span>

---

## <span style="font-size: 16px;">Where This Sits in the RLHF Pipeline</span>

<span style="font-size: 14px;">GRPO occupies the RL fine-tuning stage of RLHF, the same slot as PPO. The reward-modeling stage and the frozen reference policy carry over unchanged. GRPO's contribution is purely in how the advantage is estimated.</span>

* <span style="font-size: 14px;">**Reward modeling -> RL fine-tuning:** GRPO is an RL fine-tuning method, downstream of a reward model or a rule-based reward.</span>
* <span style="font-size: 14px;">**Versus PPO:** PPO needs a learned value critic for its advantage baseline; GRPO removes it.</span>
* <span style="font-size: 14px;">**Versus DPO:** DPO removes the RL loop entirely; GRPO keeps on-policy sampling but drops only the critic. RLOO is its closest sibling, using a leave-one-out group baseline instead of group standardization.</span>
* <span style="font-size: 14px;">**On-policy advantage:** because GRPO samples fresh completions from the current policy, it can discover and reinforce behaviors absent from any static dataset, which is what enabled the emergent long reasoning chains in DeepSeek-R1.</span>

<span style="font-size: 14px;">The motivation in DeepSeekMath was cost. Training a value model of comparable size to the policy roughly doubles memory and adds a second network to tune. For reasoning tasks where a verifier gives a clean correct or incorrect reward, the authors found a group baseline works as well as a learned critic at a fraction of the cost.</span>

<span style="font-size: 14px;">There is also a structural argument specific to language models. In token-level PPO the value function must predict the expected future reward at every partial generation, which is a hard credit-assignment problem when the only reward arrives at the final token. A critic trained on such sparse, terminal signals is noisy and slow to converge. GRPO sidesteps this by never estimating intermediate values at all: it compares whole completions against each other, which is a much easier signal to estimate from a handful of samples.</span>

---

## <span style="font-size: 16px;">The Group-Relative Advantage</span>

<span style="font-size: 14px;">For each prompt, GRPO samples a group of $G$ completions from the current policy, scores each with the reward function, and standardizes the rewards within that group. For sample $i$ belonging to group $g$:</span>

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \varepsilon}
$$

<span style="font-size: 14px;">where $\mu_g$ is the mean reward over the group, $\sigma_g$ is the population standard deviation over the group (dividing by $|g|$, not $|g| - 1$), and $\varepsilon$ is a small constant that stabilizes the division when every completion in the group earns the same reward and the variance is zero.</span>

<span style="font-size: 14px;">The advantage is shared by every token in completion $i$, since the reward is a single scalar for the whole response. This is the **outcome-supervision** form. The result is a z-score: completions that beat their group's average reward get positive advantage and are reinforced, while below-average completions get negative advantage and are suppressed.</span>

<span style="font-size: 14px;">The name "group relative" captures the core idea: an absolute reward is meaningless to the policy gradient on its own, what matters is how a completion ranks against the other attempts at the same prompt. A reward of 0.8 is good if its groupmates scored 0.2 but mediocre if they scored 0.95. By standardizing within the group, GRPO converts raw rewards into a relative signal that is automatically comparable across prompts of wildly different absolute difficulty, which is the same normalization a well-trained critic would have provided implicitly.</span>

---

## <span style="font-size: 16px;">Why a Group Baseline Works</span>

<span style="font-size: 14px;">A policy gradient is $\mathbb{E}[A_i \nabla_\theta \log\pi_\theta]$. The role of any baseline is to reduce the variance of this estimator without biasing it. The classical result is that subtracting any function of the state (not the action) leaves the gradient's expectation unchanged, because $\mathbb{E}_{a \sim \pi}[\nabla_\theta\log\pi_\theta] = 0$.</span>

* <span style="font-size: 14px;">**Mean subtraction** $r_i - \mu_g$ is a Monte Carlo estimate of the state-value baseline. The group mean approximates the expected reward for that prompt under the current policy, exactly what a value critic would predict at the prompt, but computed from samples instead of a learned network.</span>
* <span style="font-size: 14px;">**Standard-deviation division** rescales advantages to unit variance per group. This normalizes the gradient magnitude across prompts of very different difficulty, so an easy prompt where all completions score high does not dominate a hard prompt where rewards are small.</span>
* <span style="font-size: 14px;">**No bias from the critic:** a learned value function can be systematically wrong early in training, biasing the gradient. The empirical group mean is unbiased for the current policy by construction.</span>

<span style="font-size: 14px;">The variance-reduction intuition is concrete. Without a baseline, REINFORCE pushes up the log-probability of every sampled completion in proportion to its raw reward, so even a uniformly mediocre group gets reinforced as long as rewards are positive, and the gradient direction is dominated by the reward's mean rather than which completions were relatively better. Subtracting $\mu_g$ centers the rewards so only the **relative** ranking within the group drives the update: completions better than their peers go up, worse ones go down, and the absolute reward level washes out.</span>

<span style="font-size: 14px;">The standard-deviation division does introduce a small bias, because $\sigma_g$ depends on the sampled actions and is therefore not a pure state-only baseline. In practice this bias is minor and the variance reduction from scale-normalization outweighs it, but it is the reason RLOO, which keeps strict unbiasedness, omits the std term.</span>

---

## <span style="font-size: 16px;">The Full GRPO Objective</span>

<span style="font-size: 14px;">GRPO plugs this advantage into a PPO-style clipped surrogate, retaining the KL penalty to the reference. For a group of $G$ completions with token-length $|o_i|$:</span>

$$
\mathcal{L} = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\min\!\left(\rho_{i,t} A_i,\; \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_i\right) + \beta\,\text{KL}\!\left(\pi_\theta \,\|\, \pi_{ref}\right)
$$

<span style="font-size: 14px;">where $\rho_{i,t}$ is the per-token importance ratio. The clipping and KL penalty are inherited directly from PPO; the only structural change is that $A_i$ comes from group standardization rather than GAE over a critic. DeepSeek used the low-variance unbiased KL estimator $\pi_{ref}/\pi_\theta - \log(\pi_{ref}/\pi_\theta) - 1$ rather than the plain log-ratio.</span>

<span style="font-size: 14px;">A practical detail is the group size $G$. Too small a group makes $\mu_g$ and $\sigma_g$ noisy estimates, undermining the baseline; too large multiplies the sampling cost since every completion must be generated and scored. DeepSeekMath used groups in the range of 8 to 64 completions per prompt depending on the task, balancing baseline quality against generation throughput. Because all $G$ completions share the same prompt, they can be generated and scored in a single efficient batch.</span>

---

## <span style="font-size: 16px;">Population vs Sample Standard Deviation</span>

<span style="font-size: 14px;">The problem specifies the **population** standard deviation, dividing by $|g|$ rather than $|g| - 1$:</span>

$$
\sigma_g = \sqrt{\frac{1}{|g|}\sum_{i \in g}\left(r_i - \mu_g\right)^2}
$$

<span style="font-size: 14px;">The group is treated as the full population of samples being normalized, not a sample drawn from a larger set to estimate an unknown variance, so Bessel's correction $|g| - 1$ is not applied. Using the sample standard deviation by mistake inflates $\sigma_g$ for small groups and shrinks the advantages, subtly miscalibrating the gradient scale. The $\varepsilon$ guard matters because reasoning rewards are often binary: if all completions in a group are correct or all are wrong, $\sigma_g = 0$ and the division would blow up without it.</span>

<span style="font-size: 14px;">Note also that $\mu_g$ in the variance formula is the same group mean used in the numerator, so the computation is a single pass: compute the mean, then the mean of squared deviations, then the square root. Mixing a running-mean shortcut that uses a different denominator for the mean and the variance is a common source of off-by-a-factor errors in custom implementations.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take one group of $G = 4$ completions with rewards $r = [1.0, 0.0, 0.0, 1.0]$ and $\varepsilon = 10^{-8}$.</span>

<span style="font-size: 14px;">1. **Group mean**: $\mu_g = (1.0 + 0.0 + 0.0 + 1.0)/4 = 0.5$.</span>

<span style="font-size: 14px;">2. **Squared deviations**: each is $(1.0 - 0.5)^2 = 0.25$ or $(0.0 - 0.5)^2 = 0.25$, so all four are $0.25$.</span>

<span style="font-size: 14px;">3. **Population variance**: $\sigma_g^2 = (0.25 \times 4)/4 = 0.25$, so $\sigma_g = 0.5$. Note the division is by $|g| = 4$, not $3$; the sample std would give $\sqrt{1/3} \approx 0.577$ and the wrong advantages.</span>

<span style="font-size: 14px;">4. **Advantages**: $A = (r - 0.5)/(0.5 + 10^{-8})$, giving $[1.0, -1.0, -1.0, 1.0]$ after rounding to 4 decimals.</span>

<span style="font-size: 14px;">The two correct completions get advantage $+1$ and are reinforced, the two wrong ones get $-1$ and are suppressed, all without any value network. Output preserves the original sample order.</span>

<span style="font-size: 14px;">A second instructive case is a degenerate group $r = [1.0, 1.0, 1.0, 1.0]$ where every completion is correct. Then $\mu_g = 1.0$, $\sigma_g = 0$, and each advantage is $(1.0 - 1.0)/(0 + \varepsilon) = 0$. The whole group contributes zero gradient, which is exactly right: if every sampled answer is equally good, there is nothing to learn from comparing them. This is why the $\varepsilon$ guard cannot be omitted and why GRPO naturally focuses learning on prompts where the policy is still uncertain and produces a mix of outcomes.</span>

---

## <span style="font-size: 16px;">Connection to the Broader Pipeline</span>

<span style="font-size: 14px;">Reading the three RL fine-tuning approaches together clarifies the design space. All optimize the same KL-regularized reward objective; they differ in how they estimate the advantage and whether they sample on-policy.</span>

* <span style="font-size: 14px;">**PPO:** advantage from GAE over a learned critic. On-policy, most general, most expensive.</span>
* <span style="font-size: 14px;">**GRPO:** advantage from within-group reward standardization. On-policy, no critic, scale-normalized per prompt.</span>
* <span style="font-size: 14px;">**RLOO:** advantage from a leave-one-out group mean. On-policy, no critic, and strictly unbiased.</span>
* <span style="font-size: 14px;">**DPO:** no advantage and no sampling at all; here the implicit reward is the policy-reference log-ratio on a fixed preference dataset.</span>

<span style="font-size: 14px;">GRPO's sweet spot is verifiable-reward reasoning at scale, where rule-based correctness checks supply clean rewards and the cost savings from dropping the critic are large. Its reliance on multiple samples per prompt makes it data-hungry in generation but cheap in model memory, a tradeoff that pays off when generation is cheap relative to holding a second large network in memory.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

* <span style="font-size: 14px;">**Process supervision:** GRPO can assign per-step advantages when a process reward model scores intermediate reasoning steps, rather than a single outcome reward, giving denser credit assignment on long chains of thought.</span>
* <span style="font-size: 14px;">**DeepSeek-R1:** GRPO with rule-based verifiable rewards (correct answer, valid format) drove the emergence of long chain-of-thought reasoning, showing a critic-free method can scale to frontier reasoning training. Using verifiable rewards also removes the reward-hacking surface of a learned reward model, since a rule checker cannot be gamed the way a neural reward model can.</span>
* <span style="font-size: 14px;">**Relation to RLOO:** both drop the critic and use a group baseline. RLOO subtracts the mean of the **other** samples (leave-one-out) and skips the standard-deviation scaling, which keeps the estimator unbiased; GRPO's std division introduces a small bias but normalizes scale across prompts.</span>
* <span style="font-size: 14px;">**Std-normalization debate:** later analyses noted the standard-deviation division can over-weight prompts where the group barely varies, and some implementations drop it, reducing GRPO toward a mean-only baseline that behaves much like RLOO.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using sample instead of population std.** Dividing by $|g| - 1$ inflates the denominator for small groups and rescales every advantage, breaking the gradient calibration the method was tuned for. Always divide by $|g|$.</span>
* <span style="font-size: 14px;">**Omitting the epsilon on zero-variance groups.** With binary rewards a whole group is often all-correct or all-wrong, giving $\sigma_g = 0$. Without the $\varepsilon$ guard this produces division by zero and `NaN` advantages that corrupt the entire batch.</span>
* <span style="font-size: 14px;">**Standardizing across the wrong axis.** The mean and std must be computed per prompt group, not across the entire batch. Standardizing over the whole batch mixes prompts of different difficulty and destroys the within-prompt comparison that gives the baseline its meaning.</span>
* <span style="font-size: 14px;">**Forgetting the advantage is shared per completion.** Every token of a completion gets the same scalar advantage $A_i$ under outcome supervision. Recomputing or rescaling it per token, or shuffling the output order, breaks the correspondence between samples and their rewards.</span>

---