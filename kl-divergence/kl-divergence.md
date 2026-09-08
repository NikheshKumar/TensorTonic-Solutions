## What Is KL Divergence?

Kullback-Leibler (KL) divergence measures how one probability distribution differs from another. It answers the question: "If I use distribution $Q$ to approximate distribution $P$, how much information do I lose?"

$$
D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

Where:
- $P$ is the "true" or reference distribution
- $Q$ is the "approximate" or model distribution
- The sum is over all possible outcomes $x$

---

## Interpreting the Formula

Breaking down $P(x) \log \frac{P(x)}{Q(x)}$:

**The ratio $\frac{P(x)}{Q(x)}$:**
- If $P(x) > Q(x)$: the ratio is $> 1$, log is positive (we underestimate this outcome)
- If $P(x) = Q(x)$: the ratio is 1, log is 0 (perfect match)
- If $P(x) < Q(x)$: the ratio is $< 1$, log is negative (we overestimate this outcome)

**Weighting by $P(x)$:**
- The error at each point is weighted by the true probability
- Errors on high-probability outcomes matter more
- Errors on low-probability outcomes contribute little

---

## Key Properties of KL Divergence

**Non-negativity:**

$$
D_{KL}(P || Q) \geq 0
$$
With equality if and only if $P = Q$ everywhere.

**Asymmetry:**

$$
D_{KL}(P || Q) \neq D_{KL}(Q || P)
$$
KL divergence is NOT a distance metric. The order matters.

**Unbounded:**
- If $Q(x) = 0$ where $P(x) > 0$: $\log \frac{P(x)}{0} = \infty$
- KL divergence can be infinite if $Q$ assigns zero probability to an outcome that $P$ says is possible

---

## Numerical Example

True distribution $P$: [0.7, 0.2, 0.1]
Model distribution $Q$: [0.5, 0.3, 0.2]

$$
D_{KL}(P || Q) = 0.7 \log \frac{0.7}{0.5} + 0.2 \log \frac{0.2}{0.3} + 0.1 \log \frac{0.1}{0.2}
$$

Computing each term:
- $0.7 \log(1.4) = 0.7 \times 0.336 = 0.235$
- $0.2 \log(0.667) = 0.2 \times (-0.405) = -0.081$
- $0.1 \log(0.5) = 0.1 \times (-0.693) = -0.069$

Total: $0.235 - 0.081 - 0.069 = 0.085$

This is in nats (natural log). For bits, use $\log_2$:
$D_{KL} \approx 0.085 / \ln(2) \approx 0.123$ bits

---

## Forward vs. Reverse KL

**Forward KL: $D_{KL}(P || Q)$**
- Minimize this to make $Q$ "cover" $P$
- Penalizes $Q(x) = 0$ where $P(x) > 0$ (infinite penalty)
- Results in a $Q$ that is "mean-seeking" or "mode-covering"
- $Q$ tends to be broader than necessary

**Reverse KL: $D_{KL}(Q || P)$**
- Minimize this to make $Q$ "fit inside" $P$
- Penalizes $Q(x) > 0$ where $P(x) = 0$
- Results in a $Q$ that is "mode-seeking"
- $Q$ tends to collapse to a single mode of $P$

This distinction is crucial in variational inference and generative modeling.

---

## Continuous Distributions

For continuous probability densities $p$ and $q$:

$$
D_{KL}(p || q) = \int p(x) \log \frac{p(x)}{q(x)} dx
$$

**Special case: two Gaussians**

For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$
D_{KL}(p || q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

This closed-form expression is used extensively in variational autoencoders (VAEs).

---

## KL Divergence and Cross-Entropy

KL divergence is related to cross-entropy and entropy:

$$
D_{KL}(P || Q) = H(P, Q) - H(P)
$$

Where:
- $H(P, Q) = -\sum_x P(x) \log Q(x)$ is cross-entropy
- $H(P) = -\sum_x P(x) \log P(x)$ is entropy of $P$

Since $H(P)$ is constant with respect to $Q$:
- Minimizing KL divergence $\equiv$ minimizing cross-entropy
- This is why cross-entropy loss trains the model to match the data distribution

---

## The Gradient

For a parametric model $Q_\theta$:

$$
\frac{\partial D_{KL}}{\partial \theta} = -\sum_x P(x) \frac{1}{Q_\theta(x)} \frac{\partial Q_\theta(x)}{\partial \theta}
$$

This shows:
- Gradient is weighted by $P(x)/Q_\theta(x)$
- Large gradient where $Q$ underestimates $P$
- The model is pushed to increase probability where it is currently too low

---

## Where KL Divergence Is Used

**Variational Autoencoders (VAEs):**
- The latent distribution $q(z|x)$ should match the prior $p(z)$
- KL divergence term in the ELBO: $D_{KL}(q(z|x) || p(z))$

**Knowledge distillation:**
- Student network should match teacher's soft predictions
- Minimize KL divergence between student and teacher outputs

**Reinforcement learning (policy optimization):**
- Constrain policy updates: $D_{KL}(\pi_{\text{old}} || \pi_{\text{new}}) < \delta$
- Used in TRPO, PPO

**Information theory:**
- Measuring information loss in compression
- Quantifying surprise or unexpectedness

**Language models:**
- Training: cross-entropy loss (equivalent to forward KL)
- Evaluation: perplexity (exponentiated cross-entropy)

---

## Numerical Stability

Computing KL divergence directly can cause issues:
- Division by zero if $Q(x) = 0$
- Log of zero if either distribution is zero

Solutions:
- Add small $\epsilon$ to both distributions before computing
- Use log-space computation: $\log P(x) - \log Q(x)$
- Use library functions that handle edge cases