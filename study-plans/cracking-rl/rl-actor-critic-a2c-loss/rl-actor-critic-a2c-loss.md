# <span style="font-size: 20px;">Actor-Critic (A2C) Loss with Entropy</span>

<span style="font-size: 14px;">Advantage Actor-Critic (A2C) is the synchronous variant of the A3C algorithm (Mnih et al., 2016) that jointly trains a **policy** (the actor) and a **value function** (the critic) with a single composite objective. The actor follows a variance-reduced policy gradient weighted by the critic's advantage, the critic regresses toward bootstrapped returns, and an entropy bonus sustains exploration. It is the direct predecessor of PPO and the simplest method that combines the policy-gradient and value-learning families.</span>

---

## <span style="font-size: 16px;">Actor and Critic</span>

<span style="font-size: 14px;">A2C maintains two function approximators, usually two heads on a shared network body:</span>

* <span style="font-size: 14px;">**Actor** $\pi_\theta(a|s)$: a stochastic policy that outputs action probabilities (softmax logits for discrete actions, or Gaussian parameters for continuous ones). It is the thing being optimized to maximize return.</span>
* <span style="font-size: 14px;">**Critic** $V_\phi(s)$: a state-value estimator that predicts the expected return from a state. It does not choose actions; its sole purpose is to provide a low-variance baseline and bootstrap targets for the actor.</span>

<span style="font-size: 14px;">The name "advantage actor-critic" comes from using the critic to compute the **advantage** $A_t = Q(s_t, a_t) - V(s_t)$, the centered learning signal that tells the actor whether an action beat the state's expected value. Sharing a body lets representation learning benefit both heads, but the two losses must be balanced carefully so neither dominates the shared parameters.</span>

---

## <span style="font-size: 16px;">From Policy Gradient to Actor-Critic</span>

<span style="font-size: 14px;">A2C descends directly from the policy gradient theorem. The general form of the gradient is $\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\, \Psi_t]$, where $\Psi_t$ is some measure of how good action $a_t$ was. Schulman et al. catalogue the valid choices for $\Psi_t$: the total return, the return-to-go $G_t$, the return minus a baseline, the action-value $Q(s_t,a_t)$, the advantage $A(s_t,a_t)$, or the TD residual. All give the same expected gradient; they differ only in variance.</span>

<span style="font-size: 14px;">REINFORCE picks $\Psi_t = G_t$ (high variance). REINFORCE with a baseline picks $\Psi_t = G_t - b(s_t)$. A2C goes further by choosing $\Psi_t = A_t$, the **advantage**, and crucially by **learning** the value function that defines it rather than using a fixed or hand-tuned baseline. The critic is what turns a baselined policy gradient into a true actor-critic: the baseline is now a trained, state-aware estimator that also supplies bootstrap targets, enabling learning before an episode ends.</span>

---

## <span style="font-size: 16px;">The Policy (Actor) Loss</span>

<span style="font-size: 14px;">The actor minimizes the negative advantage-weighted log-probability, exactly the baselined policy gradient surrogate:</span>

$$
\mathcal{L}_{\pi} = -\frac{1}{T}\sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t)\, A_t
$$

<span style="font-size: 14px;">This descends the negative of the policy gradient $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t)\, A_t]$. The advantage $A_t$ is a **detached constant** in this term: gradients flow only through the log-probability, never through the critic that produced $A_t$. When $A_t > 0$ the action beat expectation and its probability is increased; when $A_t < 0$ it is decreased. The advantage is commonly the GAE estimate, but the simplest A2C uses the one-step advantage $A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ or the Monte Carlo form $G_t - V_\phi(s_t)$.</span>

---

## <span style="font-size: 16px;">The Value (Critic) Loss</span>

<span style="font-size: 14px;">The critic is trained by regression to the target return with a squared error:</span>

$$
\mathcal{L}_{V} = \frac{1}{T}\sum_{t=0}^{T-1} \big( G_t - V_\phi(s_t) \big)^2
$$

<span style="font-size: 14px;">Here $G_t$ is the bootstrapped target return, for example the $n$-step return $r_t + \gamma r_{t+1} + \ldots + \gamma^n V_\phi(s_{t+n})$ or the GAE-implied return $A_t^{GAE} + V_\phi(s_t)$. Minimizing this drives $V_\phi$ toward the true expected return, which in turn makes the advantage a better-centered, lower-variance signal for the actor. Unlike the actor loss, gradients here **do** flow through $V_\phi(s_t)$; the target $G_t$ is detached so the critic chases the target rather than the target chasing the critic.</span>

---

## <span style="font-size: 16px;">The Entropy Bonus</span>

<span style="font-size: 14px;">A pure policy gradient tends to collapse prematurely onto a single action, killing exploration before the agent has discovered the best behavior. A2C counters this with an **entropy bonus** that rewards a policy for staying uncertain. The per-step entropy of a discrete policy is:</span>

$$
H(\pi(\cdot|s_t)) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)
$$

<span style="font-size: 14px;">and $\bar{H} = \frac{1}{T}\sum_t H(\pi(\cdot|s_t))$ is the mean per-step entropy. Entropy is maximized by a uniform distribution and minimized (zero) by a deterministic one. Because the goal is to **maximize** entropy, it enters the minimized total loss with a negative sign, so reducing the loss encourages higher entropy. Unlike the advantage, the entropy term is differentiable through the policy and contributes a real gradient that pushes probabilities toward uniformity, fading in influence as $c_e$ is small.</span>

<span style="font-size: 14px;">The entropy bonus addresses a structural failure mode of policy gradients called **premature convergence** or policy collapse. Early in training the advantage estimates are noisy, and a run of spuriously positive advantages for one action can rapidly inflate its probability toward $1$. Once a discrete action has probability near $1$, the policy almost never samples alternatives, so it can never discover that another action is better; the gradient on the unexplored actions vanishes because they are never tried. Entropy regularization keeps a floor of randomness in the policy so exploration continues until the evidence genuinely favors commitment.</span>

<span style="font-size: 14px;">For continuous Gaussian policies the same role is played by the entropy of the action distribution, $H = \frac{1}{2}\log(2\pi e\, \sigma^2)$ per dimension, which grows with the policy's standard deviation $\sigma$. Maximizing it discourages the policy from shrinking $\sigma$ to zero too early, preserving exploratory noise in the continuous action.</span>

---

## <span style="font-size: 16px;">The Combined Objective</span>

<span style="font-size: 14px;">The three terms are summed into a single scalar loss optimized in one backward pass:</span>

$$
\mathcal{L} = \mathcal{L}_{\pi} + c_v\, \mathcal{L}_{V} - c_e\, \bar{H}
$$

<span style="font-size: 14px;">The coefficients balance the objectives:</span>

* <span style="font-size: 14px;">**$c_v$ (value coefficient)**: scales the critic loss, typically $0.5$. With a shared body, too large a $c_v$ lets value regression dominate the representation and starves the actor; too small leaves the critic inaccurate and the advantages noisy.</span>
* <span style="font-size: 14px;">**$c_e$ (entropy coefficient)**: scales the exploration bonus, typically a small value such as $0.01$. Larger values keep the policy more random for longer; setting it to zero often causes premature convergence to a suboptimal deterministic policy.</span>

<span style="font-size: 14px;">Optimizing the sum jointly, rather than alternating, is what makes A2C a single coherent algorithm. The shared optimizer step updates actor and critic parameters together, and the synchronous design (multiple parallel environment workers whose gradients are averaged) is what distinguishes A2C from the asynchronous A3C.</span>

---

## Worked Example ($T = 2$, $c_v = 0.5$, $c_e = 0.01$)

<span style="font-size: 14px;">Two steps with log-probabilities $\log\pi = [-0.7, -0.4]$, advantages $A = [1.0, -0.5]$, returns $G = [2.0, 1.0]$, critic values $V = [1.5, 1.2]$, and per-step entropies $H = [0.6, 0.5]$.</span>

<span style="font-size: 14px;">1. **Policy loss**: $\mathcal{L}_\pi = -\frac{1}{2}\big[(-0.7)(1.0) + (-0.4)(-0.5)\big] = -\frac{1}{2}(-0.7 + 0.2) = 0.25$.</span>

<span style="font-size: 14px;">2. **Value loss**: $\mathcal{L}_V = \frac{1}{2}\big[(2.0-1.5)^2 + (1.0-1.2)^2\big] = \frac{1}{2}(0.25 + 0.04) = 0.145$.</span>

<span style="font-size: 14px;">3. **Mean entropy**: $\bar{H} = \frac{1}{2}(0.6 + 0.5) = 0.55$.</span>

<span style="font-size: 14px;">4. **Total**: $\mathcal{L} = 0.25 + 0.5(0.145) - 0.01(0.55) = 0.25 + 0.0725 - 0.0055 = 0.3170$.</span>

---

## <span style="font-size: 16px;">Choosing the Advantage Estimator</span>

<span style="font-size: 14px;">The advantage that weights the actor loss can be computed several ways, trading bias against variance exactly as discussed in GAE:</span>

* <span style="font-size: 14px;">**One-step (TD) advantage** $A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$. Lowest variance, biased by the critic's error. This is the textbook A2C advantage and allows fully online updates.</span>
* <span style="font-size: 14px;">**n-step advantage** $A_t = \sum_{l=0}^{n-1}\gamma^l r_{t+l} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)$. A middle ground: more observed reward, less bootstrapping, moderate variance.</span>
* <span style="font-size: 14px;">**GAE** $A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$. The exponentially-weighted blend, now the default in nearly all implementations.</span>

<span style="font-size: 14px;">The value target $G_t$ in the critic loss should be consistent with the advantage choice: with GAE advantages, the target is $G_t = A_t^{GAE} + V_\phi(s_t)$, so a single rollout pass produces both. Mixing an inconsistent advantage and target (for example a one-step advantage with a Monte Carlo value target) is a common and subtle source of instability.</span>

---

## <span style="font-size: 16px;">Shared vs Separate Networks</span>

<span style="font-size: 14px;">The actor and critic can either share a feature-extraction body with two output heads or be fully separate networks. Sharing is parameter-efficient and lets the two tasks regularize a common representation, which often speeds learning on pixel-based inputs where the convolutional encoder is expensive. The risk is **gradient interference**: the value loss and policy loss compete for the shared parameters, so the coefficient $c_v$ effectively controls how much the critic objective reshapes the features the actor depends on.</span>

<span style="font-size: 14px;">Separate networks avoid this interference at the cost of more parameters and no shared representation learning. Many high-performing PPO setups for continuous control use separate actor and critic networks for precisely this reason, while pixel-based agents typically share the encoder. The decision interacts with $c_v$: separate networks tolerate a wider range of value coefficients because there is no competition for shared weights.</span>

---

## <span style="font-size: 16px;">Why Combine Them</span>

<span style="font-size: 14px;">A2C unifies the strengths of both reinforcement learning families. From policy gradients it inherits native support for stochastic and continuous policies and smooth improvement. From value learning it inherits the critic, which provides a learned baseline that reduces variance far more effectively than any return-based statistic, and bootstrapped targets that allow learning from incomplete episodes rather than waiting for termination as REINFORCE must. The advantage formulation is the hinge: it is simultaneously the variance-reduced actor signal and a quantity defined entirely in terms of the critic's outputs.</span>

<span style="font-size: 14px;">The mutual dependence between actor and critic is worth naming. The critic supplies the advantage that shapes the actor; the actor's improving policy changes the distribution of states visited, which changes what the critic must learn to predict. This coupled, non-stationary learning problem is harder than either task alone, which is why careful coefficient balancing, advantage normalization, and an exploration bonus all matter so much in practice. When the two components stay roughly in step, the result is a method far more sample-efficient and stable than REINFORCE.</span>

<span style="font-size: 14px;">The paper emphasizes that running many actors in parallel decorrelates the data, which stabilizes training without a replay buffer. A2C's synchronous gradient averaging captures the same decorrelation benefit while being simpler to implement on a single machine with vectorized environments, which is why it became the common baseline.</span>

<span style="font-size: 14px;">A2C is also the conceptual scaffold for PPO. PPO keeps the same three-part loss (advantage-weighted actor term, value regression, entropy bonus) and the same GAE advantages, but replaces the plain log-probability term with a clipped probability-ratio surrogate that permits multiple optimization epochs on each batch of collected data. In that sense, understanding the A2C composite loss is most of the way to understanding PPO: the difference is confined entirely to the form of the actor term, while the critic loss and entropy bonus carry over unchanged. The same coefficients $c_v$ and $c_e$ appear in PPO with the same roles and similar default values.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Letting actor gradients flow into the advantage.** $A_t$ must be detached in $\mathcal{L}_\pi$. If it is left attached to the critic graph, the actor loss starts training the critic in the wrong direction, corrupting both heads, especially with a shared body.</span>
* <span style="font-size: 14px;">**Wrong sign on the entropy term.** Entropy is added to be maximized, so it is subtracted in the minimized total loss. Flipping the sign drives the policy toward determinism and collapses exploration, often producing fast but badly suboptimal convergence.</span>
* <span style="font-size: 14px;">**Imbalanced coefficients on a shared trunk.** A large $c_v$ lets value regression dominate the shared features and degrades the policy; too small a $c_e$ removes exploration. These interact, so tuning one in isolation can be misleading.</span>
* <span style="font-size: 14px;">**Not detaching the value target.** The critic regresses $V_\phi(s_t)$ toward $G_t$; the target must be detached. If $G_t$ contains an attached $V_\phi(s_{t+n})$ that receives gradient, the critic chases a moving target it is itself shifting, causing instability.</span>

---