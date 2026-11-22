<span style="font-size: 14px;">The Kullback-Leibler divergence is one of the most important quantities in machine learning. It measures how one probability distribution differs from another. It appears as the loss function in variational autoencoders, the objective in knowledge distillation, the regularizer in policy optimization, and the foundation of information-theoretic model selection. Understanding how to compute KL divergence - both for discrete distributions and continuous ones - is essential for anyone working with probabilistic models.</span>

## <span style="font-size: 14px;">Definition: Discrete KL Divergence</span>

<span style="font-size: 14px;">For two discrete probability distributions $P$ and $Q$ defined over the same set of outcomes $\{x_1, x_2, \ldots, x_k\}$, the KL divergence from $Q$ to $P$ (read: "KL divergence of $P$ from $Q$") is:</span>

$$
D_{\text{KL}}(P \| Q) = \sum_{i=1}^{k} P(x_i) \ln \frac{P(x_i)}{Q(x_i)}
$$

<span style="font-size: 14px;">This can be rewritten as:</span>

$$
D_{\text{KL}}(P \| Q) = \sum_{i=1}^{k} P(x_i) [\ln P(x_i) - \ln Q(x_i)] = -H(P) - \sum_{i=1}^{k} P(x_i) \ln Q(x_i)
$$

<span style="font-size: 14px;">where $H(P) = -\sum P(x_i) \ln P(x_i)$ is the entropy of $P$. The second term $-\sum P(x_i) \ln Q(x_i)$ is the cross-entropy of $P$ and $Q$. Thus KL divergence equals cross-entropy minus entropy: $D_{\text{KL}}(P \| Q) = H(P, Q) - H(P)$.</span>

### <span style="font-size: 14px;">Properties</span>

<span style="font-size: 14px;">1. Non-negativity (Gibbs' inequality): $D_{\text{KL}}(P \| Q) \geq 0$, with equality if and only if $P = Q$.</span>

<span style="font-size: 14px;">2. Asymmetry: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$ in general. This means KL divergence is not a true distance metric.</span>

<span style="font-size: 14px;">3. Not a metric: KL divergence does not satisfy the triangle inequality, nor is it symmetric. It is a divergence, not a distance.</span>

<span style="font-size: 14px;">4. Relationship to likelihood: $D_{\text{KL}}(P \| Q)$ measures the expected number of extra nats needed to encode samples from $P$ using a code optimized for $Q$.</span>

### <span style="font-size: 14px;">Handling Edge Cases</span>

<span style="font-size: 14px;">The convention for edge cases in KL divergence computation is:</span>

<span style="font-size: 14px;">If $P(x_i) = 0$: the term contributes 0 to the sum, regardless of $Q(x_i)$. This follows from the convention $0 \cdot \ln(0/q) = 0$, which is justified by the limit $\lim_{p \to 0^+} p \ln(p/q) = 0$.</span>

<span style="font-size: 14px;">If $P(x_i) > 0$ and $Q(x_i) = 0$: the term is $P(x_i) \cdot \ln(P(x_i)/0) = +\infty$. If $P$ assigns positive probability to an event that $Q$ considers impossible, the KL divergence is infinite. This makes intuitive sense: no finite amount of information can reconcile a model ($Q$) that says an event is impossible with data ($P$) showing it happens.</span>

## <span style="font-size: 14px;">Definition: Continuous KL Divergence</span>

<span style="font-size: 14px;">For two continuous distributions with density functions $p(x)$ and $q(x)$:</span>

$$
D_{\text{KL}}(p \| q) = \int_{-\infty}^{\infty} p(x) \ln \frac{p(x)}{q(x)} \, dx
$$

<span style="font-size: 14px;">This integral requires that $p(x) > 0$ implies $q(x) > 0$ (absolute continuity of $P$ with respect to $Q$). When both are Gaussian distributions, the integral has a closed-form solution.</span>

## <span style="font-size: 14px;">KL Divergence Between Two Gaussians</span>

<span style="font-size: 14px;">For $p(x) = N(\mu_1, \sigma_1^2)$ and $q(x) = N(\mu_2, \sigma_2^2)$:</span>

$$
D_{\text{KL}}(p \| q) = \ln \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

### <span style="font-size: 14px;">Derivation</span>

<span style="font-size: 14px;">Starting from the definition:</span>

$$
D_{\text{KL}} = \int p(x) \ln \frac{p(x)}{q(x)} dx = \int p(x) [\ln p(x) - \ln q(x)] dx
$$

<span style="font-size: 14px;">For a Gaussian $N(\mu, \sigma^2)$, the log-density is:</span>

$$
\ln N(x; \mu, \sigma^2) = -\frac{1}{2}\ln(2\pi) - \ln\sigma - \frac{(x-\mu)^2}{2\sigma^2}
$$

<span style="font-size: 14px;">Therefore:</span>

$$
\ln p(x) - \ln q(x) = -\ln\sigma_1 + \ln\sigma_2 - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

<span style="font-size: 14px;">Taking the expectation under $p(x) = N(\mu_1, \sigma_1^2)$:</span>

$$
E_p[\ln p - \ln q] = \ln\frac{\sigma_2}{\sigma_1} - \frac{1}{2} + \frac{E_p[(x-\mu_2)^2]}{2\sigma_2^2}
$$

<span style="font-size: 14px;">Using $E_p[(x-\mu_2)^2] = E_p[(x-\mu_1+\mu_1-\mu_2)^2] = \text{Var}(X) + (\mu_1-\mu_2)^2 = \sigma_1^2 + (\mu_1-\mu_2)^2$:</span>

$$
D_{\text{KL}} = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

### <span style="font-size: 14px;">Special Cases</span>

<span style="font-size: 14px;">When $\mu_1 = \mu_2$ (same mean): $D_{\text{KL}} = \ln(\sigma_2/\sigma_1) + \sigma_1^2/(2\sigma_2^2) - 1/2$. The divergence depends only on the ratio of standard deviations.</span>

<span style="font-size: 14px;">When $\sigma_1 = \sigma_2$ (same variance): $D_{\text{KL}} = (\mu_1-\mu_2)^2/(2\sigma^2)$. The divergence depends only on the squared difference in means, normalized by the variance. This is half the squared Mahalanobis distance.</span>

<span style="font-size: 14px;">When $p = q$ (identical): $D_{\text{KL}} = \ln(1) + 1/2 - 1/2 = 0$, as expected.</span>

## <span style="font-size: 14px;">Numerical Computation of Continuous KL</span>

<span style="font-size: 14px;">Even when a closed-form exists, it is valuable to verify it via numerical integration. The integrand is:</span>

$$
g(x) = p(x) \ln \frac{p(x)}{q(x)}
$$

<span style="font-size: 14px;">We integrate this over a range that covers the support of $p$. A natural choice is $[\mu_1 - 5\sigma_1, \mu_1 + 5\sigma_1]$, but if $q$ has a very different mean or variance, we may need a wider range. A safe choice is:</span>

$$
[a, b] = [\min(\mu_1, \mu_2) - 5\max(\sigma_1, \sigma_2), \; \max(\mu_1, \mu_2) + 5\max(\sigma_1, \sigma_2)]
$$

<span style="font-size: 14px;">Since the integrand is weighted by $p(x)$, which decays exponentially outside $\mu_1 \pm 5\sigma_1$, the contribution from the extended range is negligible. But the wider range ensures we capture any behavior near the tails of $q$ that might affect the logarithm.</span>

<span style="font-size: 14px;">We use the trapezoidal rule or Simpson's rule to evaluate the integral numerically. The result should agree with the closed-form formula to several decimal places for moderate $n$ (e.g., $n = 1000$).</span>

## <span style="font-size: 14px;">KL Divergence in Machine Learning</span>

### <span style="font-size: 14px;">Variational Autoencoders (VAEs)</span>

<span style="font-size: 14px;">The VAE training objective maximizes the evidence lower bound (ELBO):</span>

$$
\text{ELBO} = E_{q_\phi(z|x)}[\ln p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

<span style="font-size: 14px;">The KL term acts as a regularizer, pushing the approximate posterior $q_\phi(z|x)$ toward the prior $p(z)$. When both are Gaussian, the KL term has the exact closed form derived above. This is why the Gaussian VAE is so popular: the KL regularizer can be computed analytically, avoiding the need for numerical integration or sampling.</span>

### <span style="font-size: 14px;">Knowledge Distillation</span>

<span style="font-size: 14px;">In knowledge distillation, a small "student" model learns to mimic a large "teacher" model by minimizing the KL divergence between their output distributions:</span>

$$
\mathcal{L}_{\text{distill}} = D_{\text{KL}}(P_{\text{teacher}} \| P_{\text{student}})
$$

<span style="font-size: 14px;">where $P_{\text{teacher}}$ and $P_{\text{student}}$ are softmax distributions over classes. This is a discrete KL divergence computation, exactly as in Part A of this problem.</span>

### <span style="font-size: 14px;">Reinforcement Learning: PPO and TRPO</span>

<span style="font-size: 14px;">Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO) use KL divergence to constrain how much the policy changes between updates:</span>

$$
D_{\text{KL}}(\pi_{\text{old}} \| \pi_{\text{new}}) \leq \delta
$$

<span style="font-size: 14px;">This constraint prevents catastrophically large policy updates that could destabilize training. For Gaussian policies (common in continuous control), the KL divergence again has the closed form derived above.</span>

### <span style="font-size: 14px;">Information Theory and Model Selection</span>

<span style="font-size: 14px;">KL divergence connects to the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) through its role as a measure of information loss. When we approximate a true distribution $P$ with a model $Q$, the KL divergence $D_{\text{KL}}(P \| Q)$ quantifies exactly how much information is lost. AIC can be derived as an estimate of the expected KL divergence between the true distribution and the fitted model.</span>

## <span style="font-size: 14px;">Forward vs. Reverse KL</span>

<span style="font-size: 14px;">The asymmetry of KL divergence has important practical implications:</span>

<span style="font-size: 14px;">Forward KL, $D_{\text{KL}}(P \| Q)$, is "mean-seeking": minimizing it over $Q$ encourages $Q$ to cover all modes of $P$, even at the cost of placing probability mass where $P$ is small. This is because $D_{\text{KL}}(P \| Q) = \infty$ whenever $P(x) > 0$ and $Q(x) = 0$.</span>

<span style="font-size: 14px;">Reverse KL, $D_{\text{KL}}(Q \| P)$, is "mode-seeking": minimizing it over $Q$ encourages $Q$ to concentrate on a single mode of $P$, avoiding regions where $P$ is small. This is because $D_{\text{KL}}(Q \| P) = \infty$ whenever $Q(x) > 0$ and $P(x) = 0$.</span>

<span style="font-size: 14px;">In variational inference, the choice between forward and reverse KL determines the behavior of the approximation. The standard variational Bayes approach minimizes reverse KL (equivalent to maximizing the ELBO), which tends to underestimate uncertainty by focusing on the dominant mode.</span>

## <span style="font-size: 14px;">Numerical Stability</span>

<span style="font-size: 14px;">Computing KL divergence requires evaluating $\ln(P/Q)$, which involves logarithms of potentially very small numbers. Several strategies improve numerical stability:</span>

<span style="font-size: 14px;">1. Use $\ln P - \ln Q$ instead of $\ln(P/Q)$ to avoid the intermediate ratio, which could overflow or underflow.</span>

<span style="font-size: 14px;">2. For discrete distributions, skip terms where $P(x_i) = 0$ (they contribute 0 regardless).</span>

<span style="font-size: 14px;">3. For continuous distributions, compute the log-density difference directly from the parametric form rather than computing densities first and then taking their ratio.</span>

<span style="font-size: 14px;">4. For Gaussians, the log-density difference $\ln p(x) - \ln q(x)$ can be computed without evaluating exponentials:</span>

$$
\ln p(x) - \ln q(x) = \ln\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}
$$

<span style="font-size: 14px;">This avoids the catastrophic cancellation that would occur when subtracting two large, nearly equal log-density values.</span>

## <span style="font-size: 14px;">Connection to Cross-Entropy Loss</span>

<span style="font-size: 14px;">The cross-entropy loss used in classification is closely related to KL divergence. For a single sample with true label $y$ (one-hot) and predicted probabilities $\hat{y}$:</span>

$$
\mathcal{L}_{\text{CE}} = -\sum_i y_i \ln \hat{y}_i = H(y, \hat{y})
$$

<span style="font-size: 14px;">Since the true distribution has zero entropy (it is a delta), minimizing the cross-entropy is equivalent to minimizing $D_{\text{KL}}(y \| \hat{y})$. The two objectives differ by a constant (the entropy of $y$), which does not affect optimization.</span>

## <span style="font-size: 14px;">Interpreting KL Divergence Values</span>

<span style="font-size: 14px;">When computing discrete KL divergence, each term $P(x_i) \ln(P(x_i)/Q(x_i))$ contributes positively when $P$ assigns more weight than $Q$ to outcome $x_i$ (since the log ratio is positive), and negatively when $P$ assigns less weight. The total divergence is always non-negative (Gibbs' inequality), and it is zero only when the distributions are identical. When two distributions assign their probability mass in opposite orders (e.g., $P$ concentrates on the first outcome while $Q$ concentrates on the last), the divergence is large because $P$ is "surprised" by $Q$'s assignments.</span>

<span style="font-size: 14px;">An important point to verify in practice: $D_{\text{KL}}(P \| Q) \neq D_{\text{KL}}(Q \| P)$. Computing both directions on the same pair of distributions produces different values, illustrating the asymmetry of KL divergence.</span>

## <span style="font-size: 14px;">Verifying the Gaussian KL Formula Numerically</span>

<span style="font-size: 14px;">The closed-form Gaussian KL formula $D_{\text{KL}} = \ln(\sigma_2/\sigma_1) + (\sigma_1^2 + (\mu_1-\mu_2)^2)/(2\sigma_2^2) - 1/2$ can be verified by numerical integration. The integrand $p(x) \ln(p(x)/q(x))$ is a smooth function that peaks near the mean of $p$ and decays to zero in both tails. When the two distributions differ in both mean and variance, the integrand reflects both the location shift (through the $(\mu_1-\mu_2)^2$ term) and the scale mismatch (through the $\ln(\sigma_2/\sigma_1)$ and $\sigma_1^2/\sigma_2^2$ terms). With sufficient quadrature points, the numerical result matches the closed form to many decimal places, confirming the derivation.</span>

## <span style="font-size: 14px;">Proof of Non-Negativity</span>

<span style="font-size: 14px;">The non-negativity of KL divergence, $D_{\text{KL}}(P \| Q) \geq 0$, follows from Jensen's inequality. Since $-\ln$ is a convex function:</span>

$$
\begin{aligned}
D_{\text{KL}}(P \| Q) &= \sum P(x_i) \ln\frac{P(x_i)}{Q(x_i)} \\
&= -\sum P(x_i) \ln\frac{Q(x_i)}{P(x_i)} \\
&\geq -\ln\left(\sum P(x_i) \frac{Q(x_i)}{P(x_i)}\right) = -\ln\left(\sum Q(x_i)\right) \\
&= -\ln 1 = 0
\end{aligned}
$$

<span style="font-size: 14px;">Equality holds if and only if $Q(x_i)/P(x_i)$ is constant for all $i$ where $P(x_i) > 0$, which requires $P = Q$ (since both distributions sum to 1). This elegant proof using Jensen's inequality connects KL divergence to the fundamental convexity properties of the logarithm.</span>

## <span style="font-size: 14px;">Relationship to Other Divergences</span>

<span style="font-size: 14px;">KL divergence is a member of the family of $f$-divergences, which have the general form $D_f(P \| Q) = \sum Q(x_i) f(P(x_i)/Q(x_i))$ for a convex function $f$ with $f(1) = 0$. KL divergence corresponds to $f(t) = t \ln t$. Other notable $f$-divergences include the total variation distance ($f(t) = |t-1|/2$), the chi-squared divergence ($f(t) = (t-1)^2$), and the Hellinger distance ($f(t) = (\sqrt{t}-1)^2$).</span>

<span style="font-size: 14px;">The Jensen-Shannon divergence (JSD) is a symmetrized and smoothed version of KL divergence:</span>

$$
\text{JSD}(P \| Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M), \quad M = \frac{P + Q}{2}
$$

<span style="font-size: 14px;">JSD is symmetric, always finite (even when KL is infinite), and its square root is a true metric. It is used in the original GAN training objective, where the generator minimizes the JSD between the data distribution and the generated distribution.</span>