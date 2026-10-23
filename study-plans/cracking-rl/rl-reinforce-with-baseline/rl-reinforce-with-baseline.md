# <span style="font-size: 20px;">REINFORCE with Baseline</span>

<span style="font-size: 14px;">REINFORCE with a baseline (Williams, 1992) augments the Monte Carlo policy gradient by subtracting a state-dependent reference value $b(s_t)$ from each return $G_t$. The remarkable property is that any baseline that does not depend on the action leaves the gradient **unbiased** while substantially **reducing its variance**. This single idea, choosing $b(s_t) \approx V(s_t)$, is the bridge from raw REINFORCE to modern actor-critic and PPO.</span>

---

## <span style="font-size: 16px;">The Variance Problem It Solves</span>

<span style="font-size: 14px;">Plain REINFORCE weights each log-probability by the full return $G_t$, which carries the accumulated randomness of the entire remaining episode. Its gradient is unbiased but its variance scales with the magnitude and spread of the returns, so learning is noisy and slow. Critically, the absolute scale of $G_t$ is uninformative for credit assignment: if every action in a state yields a return of around $100$, pushing up the probability of all of them does nothing useful. What matters is whether an action did **better or worse than average** in that state.</span>

<span style="font-size: 14px;">A baseline encodes exactly this "average". By subtracting $b(s_t)$, the weight on the log-probability becomes $G_t - b(s_t)$, which is positive only when the action beat the baseline and negative when it underperformed. Centering the signal around a reference removes the large common offset that contributes only noise.</span>

---

## <span style="font-size: 16px;">Starting Point: The Policy Gradient Theorem</span>

<span style="font-size: 14px;">REINFORCE optimizes the expected discounted return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$ by ascending the policy gradient. The log-derivative trick turns this gradient into an expectation over trajectories that depends only on the policy, not on the environment dynamics:</span>

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t \right]
$$

<span style="font-size: 14px;">where $G_t = \sum_{k=t}^{T-1}\gamma^{k-t} r_k$ is the return-to-go. This unbiased estimator is correct but noisy. The baseline is the cleanest possible modification: it changes the weight from $G_t$ to $G_t - b(s_t)$ and, as shown below, preserves the expectation exactly while lowering the variance.</span>

---

## <span style="font-size: 16px;">The Baselined Policy Gradient</span>

<span style="font-size: 14px;">The policy gradient theorem with a baseline states:</span>

$$
\nabla_\theta J(\theta) = \mathbb{E}\!\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)\, \big( G_t - b(s_t) \big) \right]
$$

<span style="font-size: 14px;">The corresponding surrogate loss minimized by gradient descent, averaged over the trajectory, is:</span>

$$
L = -\frac{1}{T} \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t)\, \big( G_t - b_t \big)
$$

<span style="font-size: 14px;">The quantity $G_t - b_t$ is the **advantage** estimate $\hat{A}_t$. When the baseline is the state-value function $V(s_t)$, the advantage approximates $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$, the amount by which taking $a_t$ exceeds the value of merely being in state $s_t$.</span>

---

## <span style="font-size: 16px;">Why the Baseline Adds No Bias</span>

<span style="font-size: 14px;">The key is that subtracting any function of the state alone, $b(s_t)$, adds a term whose expectation is exactly zero. Consider the extra contribution it introduces:</span>

$$
\mathbb{E}_{a \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a|s)\, b(s) \right] = b(s) \sum_a \pi_\theta(a|s)\, \nabla_\theta \log \pi_\theta(a|s)
$$

<span style="font-size: 14px;">Using $\pi_\theta \nabla_\theta \log \pi_\theta = \nabla_\theta \pi_\theta$, the sum becomes $b(s)\, \nabla_\theta \sum_a \pi_\theta(a|s)$. Since probabilities always sum to one, $\sum_a \pi_\theta(a|s) = 1$, a constant, and its gradient is zero:</span>

$$
b(s)\, \nabla_\theta (1) = 0
$$

<span style="font-size: 14px;">So the expected gradient is unchanged. The derivation hinges entirely on the baseline being **independent of the action** $a$. If $b$ depended on $a$, it could not be pulled outside the sum over actions and the cancellation would fail, introducing bias.</span>

---

## <span style="font-size: 16px;">Why It Reduces Variance</span>

<span style="font-size: 14px;">Although the mean is unchanged, the variance is not. The per-step gradient term has variance proportional to $\mathbb{E}[(G_t - b_t)^2 \, \|\nabla_\theta \log \pi_\theta\|^2]$. Choosing $b_t$ to track $\mathbb{E}[G_t \mid s_t] = V(s_t)$ minimizes the spread of the weight $G_t - b_t$ around zero, shrinking this expectation. The optimal variance-minimizing baseline is technically a magnitude-weighted average of returns:</span>

$$
b^*(s) = \frac{\mathbb{E}\big[ \|\nabla_\theta \log \pi_\theta\|^2 \, G_t \big]}{\mathbb{E}\big[ \|\nabla_\theta \log \pi_\theta\|^2 \big]}
$$

<span style="font-size: 14px;">In practice $V(s_t)$ is an excellent and easily learned approximation to this optimum, which is why it is the standard choice. The intuition: if returns from a state cluster tightly around $V(s_t)$, then $G_t - V(s_t)$ is small and near zero-mean, so the gradient estimator is far quieter than one weighted by the full $G_t$.</span>

<span style="font-size: 14px;">It is worth stressing what the baseline does and does not do to variance. It cannot eliminate the **irreducible** variance arising from the stochasticity of the return given the state, $\text{Var}(G_t \mid s_t)$, because the advantage still contains that randomness. What it removes is the **across-state** variance, the part of the spread that comes from visiting states of very different value. A perfect baseline reduces the estimator to the action-relative signal alone, which is the smallest unbiased gradient estimate achievable without changing the estimator's structure. Further reduction requires bootstrapping (introducing bias) as in actor-critic and GAE.</span>

---

## <span style="font-size: 16px;">A Concrete Variance Argument</span>

<span style="font-size: 14px;">A simple scalar example makes the variance reduction tangible. Suppose from some state the return is a random variable $G$ with mean $100$ and standard deviation $5$, and the score $g = \nabla_\theta \log \pi_\theta$ is fixed. The plain estimator weight is $G$, with mean $100$ and standard deviation $5$. Choosing $b = 100 = \mathbb{E}[G]$, the baselined weight is $G - 100$, with mean $0$ and the same standard deviation $5$. The mean of the gradient is preserved up to the cancellation argument, but the **signal-to-noise ratio** improves dramatically: a fluctuation of $\pm 5$ around a baseline of $0$ is a far cleaner learning signal than the same fluctuation around an offset of $100$, where most of the weight is constant bias that the cancellation only removes in expectation, not per-sample.</span>

<span style="font-size: 14px;">This is why centering is the entire game. Per-sample, the large constant offset $b(s)$ multiplies a noisy score and inflates the variance of individual updates even though it averages out over infinitely many samples. Removing it makes each finite-batch update much more reliable.</span>

---

## <span style="font-size: 16px;">Learning the Baseline</span>

<span style="font-size: 14px;">The baseline $V_\phi(s)$ is typically a separate parameterized network (or a head sharing the policy's body) trained by **regression** to the observed returns:</span>

$$
L_V(\phi) = \frac{1}{T} \sum_{t=0}^{T-1} \big( V_\phi(s_t) - G_t \big)^2
$$

<span style="font-size: 14px;">The two objectives are optimized together: the policy moves with the baselined policy gradient while the value network minimizes squared error to the Monte Carlo targets $G_t$. A subtle but important rule is that the baseline must be **detached** when it appears inside the advantage for the policy loss; gradients of the policy loss should not flow into $V_\phi$, otherwise the value network would be pushed to make the advantage small rather than to predict returns accurately.</span>

<span style="font-size: 14px;">This construction is exactly an actor-critic when $V_\phi$ bootstraps; here, with full Monte Carlo returns as targets, it is the simplest such method and is sometimes called "vanilla policy gradient with a learned baseline".</span>

<span style="font-size: 14px;">A practical subtlety concerns the relative learning rates of actor and critic. If the critic learns too slowly, its predictions lag the policy and the advantages are poorly centered, giving little variance reduction early in training. If it learns too fast relative to the policy, it can overfit the returns of a rapidly changing policy. A common heuristic is to give the value head a learning rate comparable to or slightly higher than the policy, and to perform several value-regression steps per policy step when returns are cheap to recompute.</span>

---

## Worked Example ($T = 3$, $\gamma = 1$)

<span style="font-size: 14px;">Take rewards giving returns-to-go $G = [3, 2, 2]$ and a learned baseline $b = [2, 2, 1]$, so the advantages are $\hat{A} = G - b = [1, 0, 1]$. Let the action probabilities be $\pi(a_0|s_0) = 0.5$, $\pi(a_1|s_1) = 0.8$, $\pi(a_2|s_2) = 0.4$.</span>

<span style="font-size: 14px;">1. **Log-probabilities**: $\log 0.5 = -0.6931$, $\log 0.8 = -0.2231$, $\log 0.4 = -0.9163$.</span>

<span style="font-size: 14px;">2. **Weight by advantage**: $(-0.6931)(1) = -0.6931$, $(-0.2231)(0) = 0$, $(-0.9163)(1) = -0.9163$.</span>

<span style="font-size: 14px;">3. **Sum**: $-0.6931 + 0 - 0.9163 = -1.6094$.</span>

<span style="font-size: 14px;">4. **Negate and average over $T = 3$**: $L = -\frac{1}{3}(-1.6094) = 0.5365$.</span>

<span style="font-size: 14px;">Compare with plain REINFORCE on the same returns ($L \approx 1.4528$): the step at $t=1$ now contributes nothing because the action exactly matched its baseline, so no gradient is wasted reinforcing an average-quality action.</span>

---

## <span style="font-size: 16px;">Credit Assignment Interpretation</span>

<span style="font-size: 14px;">The advantage $G_t - V(s_t)$ answers a sharper question than the raw return. The return asks "was this trajectory good in absolute terms?", which conflates the quality of the action with the quality of the situation the agent happened to be in. The advantage asks "given that I was in state $s_t$, was choosing $a_t$ better than my policy's typical behavior here?". This isolates the contribution of the action from the value of the state, which is precisely the information a policy update should use.</span>

<span style="font-size: 14px;">Consider an agent in a high-value state where every reasonable action yields a large return. Plain REINFORCE reinforces all of them strongly because $G_t$ is large, even the mediocre ones. With a baseline of $V(s_t)$, only the actions that beat the state's expected value get a positive push, and the merely-average ones get near-zero weight. The agent stops wasting gradient on reinforcing good fortune and concentrates it on genuinely good decisions.</span>

<span style="font-size: 14px;">Symmetrically, in a low-value state where everything is bad, the unbaselined return suppresses every action's probability, which is counterproductive since the agent still must act. The advantage instead promotes the least-bad actions, those with $G_t > V(s_t)$, preserving useful relative preferences even when all outcomes are poor.</span>

---

## <span style="font-size: 16px;">Choices of Baseline in Practice</span>

<span style="font-size: 14px;">Several baselines are common, ordered roughly by sophistication:</span>

* <span style="font-size: 14px;">**Constant baseline.** A running average of all returns seen so far. Trivial to compute and already removes the bulk of the offset, but ignores that different states have different value, so it leaves state-dependent variance untouched.</span>
* <span style="font-size: 14px;">**Batch-mean baseline.** Subtract the mean return of the current batch, often combined with dividing by the batch standard deviation. This is the cheap normalization used as a default in many implementations and requires no learned parameters.</span>
* <span style="font-size: 14px;">**Learned state-value baseline $V_\phi(s)$.** A neural network predicting the expected return from each state. This is the principled choice that approximates the optimal baseline and is the standard in actor-critic methods.</span>

<span style="font-size: 14px;">Importantly, the baseline can be **time-dependent** as $b_t$ without harming unbiasedness, since the time index is part of the state in a finite-horizon problem. This is why the loss is written with a per-step $b_t$ rather than a single global constant.</span>

---

## <span style="font-size: 16px;">Relation to Advantage and Later Methods</span>

<span style="font-size: 14px;">Subtracting $V(s_t)$ turns the return into an advantage, and advantage-weighted policy gradients are the foundation of everything downstream. A2C replaces the Monte Carlo $G_t$ with a bootstrapped advantage; **GAE** (Schulman et al., 2016) generalizes the advantage into an exponentially-weighted sum of TD residuals with a tunable bias-variance knob; **PPO** (Schulman et al., 2017) uses a normalized advantage inside a clipped surrogate. In every case the baseline-subtraction principle, only the action-relative signal should drive the update, carries through unchanged.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using an action-dependent baseline.** The unbiasedness proof requires $b$ to depend only on the state. A baseline that depends on the chosen action breaks the cancellation $\nabla_\theta \sum_a \pi_\theta = 0$ and silently biases the gradient toward the wrong policy.</span>
* <span style="font-size: 14px;">**Backpropagating the policy loss into the value network.** The advantage $G_t - V_\phi(s_t)$ must use a detached $V_\phi$ in the policy term. If gradients leak, the critic learns to shrink the advantage instead of predicting returns, corrupting both objectives.</span>
* <span style="font-size: 14px;">**Confusing variance reduction with bias.** A poorly fit baseline still yields an unbiased gradient; it just reduces variance less. The baseline never needs to be accurate for correctness, only helpful for stability, so a slow-learning critic degrades speed, not direction.</span>
* <span style="font-size: 14px;">**Forgetting to detach the advantage in the policy loss.** Even when $V_\phi$ is correct, the scalar advantage weight must be treated as a constant for the policy gradient, exactly as $G_t$ is in plain REINFORCE, or the log-probability term picks up spurious gradient contributions.</span>

---