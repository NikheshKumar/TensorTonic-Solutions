# <span style="font-size: 20px;">REINFORCE Gradient</span>

<span style="font-size: 14px;">REINFORCE (Williams, 1992) is the foundational **Monte Carlo policy gradient** algorithm. It directly optimizes a parameterized stochastic policy $\pi_\theta(a|s)$ by following the gradient of expected return, using complete-episode sample returns as an unbiased estimate of how good each action was. It is the conceptual ancestor of every modern policy-gradient method, including A2C and PPO, and the loss it defines is the template every later method modifies.</span>

---

## <span style="font-size: 16px;">What It Optimizes</span>

<span style="font-size: 14px;">Reinforcement learning seeks a policy that maximizes the expected discounted return. Define the objective over trajectories $\tau = (s_0, a_0, s_1, a_1, \ldots)$ sampled from the policy:</span>

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T-1} \gamma^t r_t \right]
$$

<span style="font-size: 14px;">Unlike value-based methods such as Q-learning, REINFORCE does not learn a value function and act greedily with respect to it. Instead it represents the policy explicitly, often as a neural network outputting action logits for discrete actions or the parameters of a Gaussian for continuous actions, and adjusts $\theta$ so that high-return actions become more probable. This makes it natural for **continuous action spaces** and for genuinely **stochastic policies**, where an explicit action distribution is required and greedy value maximization is awkward or intractable.</span>

<span style="font-size: 14px;">The name REINFORCE is an acronym Williams gave for "REward Increment = Nonnegative Factor times Offset Reinforcement times Characteristic Eligibility", which exactly describes the update: each parameter moves by a learning rate times a reward signal times the gradient of the log-probability (the eligibility). The discount factor $\gamma \in [0, 1]$ trades off immediate against future reward and keeps the infinite-horizon return finite.</span>

---

## <span style="font-size: 16px;">The Policy Gradient Theorem</span>

<span style="font-size: 14px;">The central result is that the gradient of the expected return can be written as an expectation that does not require differentiating through the environment dynamics. Start from $J(\theta) = \int p_\theta(\tau)\, R(\tau)\, d\tau$ where $R(\tau)$ is the trajectory return. The trajectory probability factorizes as:</span>

$$
p_\theta(\tau) = p(s_0)\prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t,a_t)
$$

<span style="font-size: 14px;">Differentiating $J$ and applying the **log-derivative trick** $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\, \nabla_\theta \log p_\theta(\tau)$ converts the gradient back into an expectation over trajectories:</span>

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \right]
$$

<span style="font-size: 14px;">When the log is expanded, the initial-state term $\log p(s_0)$ and every transition term $\log p(s_{t+1}|s_t,a_t)$ do not depend on $\theta$, so their gradients vanish. Only the policy log-probabilities survive:</span>

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

<span style="font-size: 14px;">This is what makes REINFORCE **model-free**: the agent never needs to know or differentiate the dynamics. Substituting back, and using causality (an action at time $t$ can only influence rewards at $t$ and later, so earlier rewards can be dropped), the gradient becomes the return-to-go weighted form:</span>

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[ \sum_{t=0}^{T-1} \gamma^t\,\nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t \right]
$$

<span style="font-size: 14px;">The **return-to-go** discounts rewards relative to time $t$:</span>

$$
G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k
$$

<span style="font-size: 14px;">Here, $k$ indexes rewards from the current timestep through the end of the trajectory. The outer factor $\gamma^t$ in the gradient discounts that contribution from the episode's start: together, $\gamma^t\gamma^{k-t}=\gamma^k$, matching the objective. It cannot generally be omitted when $\gamma<1$.</span>

---

## <span style="font-size: 16px;">The Score Function and Eligibility</span>

<span style="font-size: 14px;">The term $\nabla_\theta \log \pi_\theta(a|s)$ is the **score function** of the policy, and Williams called it the **characteristic eligibility** of the action. It points in the direction of parameter space that most increases the log-probability of the chosen action, regardless of reward. REINFORCE simply scales this direction by the observed return: good outcomes amplify the eligibility, bad outcomes reverse it.</span>

<span style="font-size: 14px;">For a **softmax policy** over discrete actions with logits $z = f_\theta(s)$, the score has a clean closed form. With $\pi_\theta(a|s) = \text{softmax}(z)_a$, the gradient with respect to the logits is:</span>

$$
\nabla_z \log \pi_\theta(a|s) = e_a - \pi_\theta(\cdot|s)
$$

<span style="font-size: 14px;">where $e_a$ is the one-hot vector for the taken action and $\pi_\theta(\cdot|s)$ is the full probability vector. This is the same as the gradient of a cross-entropy loss with the taken action as the label, which is why REINFORCE for discrete actions is often implemented as a reward-weighted cross-entropy. For a **Gaussian policy** $a \sim \mathcal{N}(\mu_\theta(s), \sigma^2)$, the score with respect to the mean is $(a - \mu_\theta(s))/\sigma^2$, pushing the mean toward actions that earned positive return.</span>

---

## <span style="font-size: 16px;">From Gradient to Loss</span>

<span style="font-size: 14px;">Automatic-differentiation frameworks minimize a scalar **surrogate loss**. This coding task evaluates the following unweighted mean over the $T$ supplied log-probabilities and returns:</span>

$$
L = -\frac{1}{T}\sum_{t=0}^{T-1} \log\pi_\theta(a_t|s_t)\, G_t
$$

<span style="font-size: 14px;">For fixed trajectory length $T$ and $\gamma=1$, the expected gradient of $L$ is $-\nabla_\theta J/T$, so averaging only rescales the step. For the discounted start-state objective above, a trajectory-gradient estimator also needs the outer $\gamma^t$ weights. This exercise intentionally computes the stated unweighted mean from supplied returns; no additional discounting or gradient calculation is required. The returns $G_t$ are treated as **constants** during backpropagation: they multiply the log-probabilities as fixed weights, so gradients flow only through $\log\pi_\theta(a_t|s_t)$. This surrogate is not a true loss in the supervised sense, since its value at the optimum is not meaningful; only its gradient matters.</span>

<span style="font-size: 14px;">The interpretation is direct. When $G_t > 0$, the update increases $\log\pi_\theta(a_t|s_t)$, raising the probability of repeating that action in that state. When $G_t < 0$, it decreases it. The magnitude of the push scales with the size of the return, so high-reward trajectories dominate the update. This is **trial and error made differentiable**: sample behavior, observe outcomes, and shift probability mass toward whatever paid off.</span>

---

## <span style="font-size: 16px;">Why the Estimate Is Unbiased</span>

<span style="font-size: 14px;">REINFORCE uses a single sampled return $G_t$ in place of the true action-value $Q^{\pi}(s_t, a_t)$. By definition $Q^{\pi}(s_t, a_t) = \mathbb{E}[G_t \mid s_t, a_t]$, so the observed return is an unbiased Monte Carlo estimate of that quantity. With the outer $\gamma^t$ weights in the trajectory formula above, the gradient estimate is **unbiased**: its expectation over sampled trajectories equals the true policy gradient $\nabla_\theta J$. No bootstrapping, no learned approximation, and no reliance on a model are involved, which is why Williams described it as a statistical gradient-following algorithm.</span>

<span style="font-size: 14px;">Unbiasedness is a strong theoretical guarantee: with a small enough learning rate and enough samples, REINFORCE converges to a local optimum of $J$. The catch is entirely in the variance of that estimate, not its correctness.</span>

---

## <span style="font-size: 16px;">The High-Variance Problem</span>

<span style="font-size: 14px;">Unbiasedness comes at the cost of **high variance**. A full-episode return $G_t$ aggregates the randomness of every action choice, every reward, and every state transition from time $t$ onward. Two trajectories generated by the very same policy can produce wildly different returns, so the gradient estimate is noisy. The variance grows with the horizon $T$, with the stochasticity of the environment, and with the absolute scale of the rewards. Noisy gradients mean small, cautious learning rates, slow convergence, and a real risk of the policy collapsing onto a single deterministic action before it has explored enough.</span>

<span style="font-size: 14px;">A second structural issue is that REINFORCE is **on-policy and Monte Carlo**: it must wait until an episode terminates to compute $G_t$, so it cannot bootstrap from partial trajectories and cannot reuse old data collected under a previous policy. Each gradient step typically consumes fresh trajectories, making it sample-inefficient.</span>

<span style="font-size: 14px;">These weaknesses motivate the entire lineage of later methods. Subtracting a **baseline** $b(s)$ reduces variance without adding bias because the baseline does not depend on the action. Replacing $G_t$ with a learned **advantage** $A(s,a)$ centers the signal around the value of the state. **GAE** interpolates between high-variance Monte Carlo returns and low-variance bootstrapped TD targets. **PPO** takes stable, clipped steps that reuse data over several epochs. Each can be read as a direct answer to the variance and inefficiency of plain REINFORCE.</span>

---

## Worked Example ($T = 3$, $\gamma = 1$)

<span style="font-size: 14px;">Suppose a 3-step episode with rewards $r = [1, 0, 2]$ and undiscounted returns-to-go $G_0 = 1+0+2 = 3$, $G_1 = 0+2 = 2$, $G_2 = 2$. Let the policy probabilities of the actions actually taken be $\pi(a_0|s_0) = 0.5$, $\pi(a_1|s_1) = 0.8$, $\pi(a_2|s_2) = 0.4$.</span>

<span style="font-size: 14px;">1. **Log-probabilities**: $\log 0.5 = -0.6931$, $\log 0.8 = -0.2231$, $\log 0.4 = -0.9163$.</span>

<span style="font-size: 14px;">2. **Weight by returns**: $(-0.6931)(3) = -2.0794$, $(-0.2231)(2) = -0.4463$, $(-0.9163)(2) = -1.8326$.</span>

<span style="font-size: 14px;">3. **Sum**: $-2.0794 - 0.4463 - 1.8326 = -4.3583$.</span>

<span style="font-size: 14px;">4. **Negate and average over $T = 3$**: $L = -\frac{1}{3}(-4.3583) = 1.4528$.</span>

<span style="font-size: 14px;">The action at $t = 2$ had the lowest probability ($0.4$) and a positive return, so it contributes the largest per-step gradient signal toward increasing its log-probability. The action at $t = 1$ was already likely ($0.8$) with a moderate return, so its contribution is smallest.</span>

---

## <span style="font-size: 16px;">Policy-Based vs Value-Based Learning</span>

<span style="font-size: 14px;">REINFORCE sits at the head of the **policy-based** family, in contrast to **value-based** methods like Q-learning and DQN that learn $Q(s,a)$ and derive a policy implicitly. The policy-based approach has several structural advantages that the paper and later work emphasize:</span>

* <span style="font-size: 14px;">**Native stochastic policies.** A softmax or Gaussian policy can represent genuinely random optimal behavior, which value-greedy methods cannot. This matters in partially observable settings and in games where mixed strategies are optimal.</span>
* <span style="font-size: 14px;">**Continuous actions without discretization.** Optimizing a Gaussian's mean and variance avoids the intractable $\arg\max_a Q(s,a)$ over a continuous space that plagues value-based control.</span>
* <span style="font-size: 14px;">**Smooth policy improvement.** Probabilities change gradually with $\theta$, so small parameter updates produce small behavior changes, whereas a greedy $\arg\max$ can flip discontinuously and destabilize learning.</span>

<span style="font-size: 14px;">The trade-off is variance and sample efficiency: value-based bootstrapping reuses data and has lower-variance targets, while Monte Carlo policy gradients are unbiased but noisy. Actor-critic methods later combine both, using a learned value function to reduce the variance of the policy gradient.</span>

---

## <span style="font-size: 16px;">Practical Variance Reduction</span>

<span style="font-size: 14px;">Two cheap tricks are standard even before introducing a learned baseline. **Return normalization** subtracts the batch mean and divides by the batch standard deviation of the $G_t$ values, which stabilizes the scale of gradients across episodes of differing magnitude. **Reward-to-go with discounting** ($\gamma < 1$) shrinks the influence of distant, weakly-related rewards, lowering variance at the cost of a small bias toward myopic behavior. An **entropy bonus** on the policy is often added to discourage premature convergence to a deterministic policy, preserving exploration during the high-variance early phase of training.</span>

<span style="font-size: 14px;">Batching multiple episodes before each update is the simplest variance reducer of all: averaging the gradient over $N$ independent trajectories divides its variance by $N$. Because REINFORCE is on-policy, those trajectories must come from the current $\theta$, so larger batches trade wall-clock sampling time for gradient quality. In practice REINFORCE converges to a **local** optimum of $J$, not necessarily the global one, since the objective is non-convex in $\theta$. Convergence guarantees follow from stochastic approximation theory: with an unbiased gradient estimate and a learning rate schedule satisfying $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, the iterates converge to a stationary point.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Letting gradients flow through the returns.** $G_t$ must be detached and treated as a constant scalar weight. If returns are left attached to the computation graph, backpropagation produces a meaningless gradient that does not correspond to the policy gradient theorem.</span>
* <span style="font-size: 14px;">**Computing returns-to-go incorrectly.** $G_t$ is the discounted sum of rewards from $t$ onward, not the full-episode return assigned to every step. Using a single total return for all timesteps weakens credit assignment, and accumulating in the wrong direction is a frequent off-by-one bug.</span>
* <span style="font-size: 14px;">**Dropping the negative sign.** The loss is the negative weighted log-probability because optimizers minimize. Omitting the minus performs gradient descent on $J$ instead of ascent, driving the policy toward low-return actions.</span>
* <span style="font-size: 14px;">**Expecting stable learning without variance reduction.** Plain REINFORCE is correct but noisy. Without a baseline or reward normalization, training on anything beyond toy problems is slow and can stall, which is exactly why baselines and advantages exist.</span>

---