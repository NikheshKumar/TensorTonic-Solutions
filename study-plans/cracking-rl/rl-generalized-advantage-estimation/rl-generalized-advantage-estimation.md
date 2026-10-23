# <span style="font-size: 20px;">Generalized Advantage Estimation (GAE)</span>

<span style="font-size: 14px;">Generalized Advantage Estimation (Schulman et al., 2016) is the standard method for estimating the advantage function in modern policy-gradient algorithms. It blends multi-step temporal-difference residuals into a single estimate using an exponential weighting controlled by a parameter $\lambda \in [0, 1]$, giving the practitioner a smooth **bias-variance knob** that interpolates between low-variance bootstrapped TD and high-variance Monte Carlo returns. It is the advantage estimator inside virtually every PPO implementation.</span>

---

## <span style="font-size: 16px;">The Problem GAE Solves</span>

<span style="font-size: 14px;">Policy gradients weight each log-probability by an advantage $A(s_t, a_t) = Q(s_t,a_t) - V(s_t)$, the amount by which an action beats the state's expected value. The advantage is never observed directly and must be estimated, and the choice of estimator is the dominant factor in a policy gradient method's stability and speed.</span>

<span style="font-size: 14px;">The two extremes both have flaws. The **Monte Carlo** advantage $G_t - V(s_t)$ uses the full sampled return, which is unbiased but high variance because $G_t$ accumulates the noise of the entire remaining episode. The **one-step TD** advantage $r_t + \gamma V(s_{t+1}) - V(s_t)$ replaces the return after one step with the bootstrapped value estimate, which has low variance but is biased whenever $V$ is imperfect. GAE constructs a continuum between these two and lets $\lambda$ select where to sit.</span>

---

## <span style="font-size: 16px;">The TD Residual</span>

<span style="font-size: 14px;">The building block is the **one-step TD residual** (also called the TD error):</span>

$$
\delta_t = r_t + \gamma\, V(s_{t+1}) - V(s_t)
$$

<span style="font-size: 14px;">This measures the difference between the one-step bootstrapped estimate of the return, $r_t + \gamma V(s_{t+1})$, and the current value prediction $V(s_t)$. If the value function were exact, $\mathbb{E}[\delta_t] = 0$ at every step. The residual is itself a low-variance, biased estimate of the advantage: it is the advantage estimate of the $k=1$ step horizon. Longer-horizon advantage estimates can be built by summing discounted residuals, and GAE is the exponentially-weighted average of all of them.</span>

---

## <span style="font-size: 16px;">k-Step Advantage Estimators</span>

<span style="font-size: 14px;">Define the $k$-step advantage estimator $\hat{A}_t^{(k)}$ as the sum of the first $k$ discounted TD residuals:</span>

$$
\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} (\gamma)^l \delta_{t+l}
$$

<span style="font-size: 14px;">Equivalently, $\hat{A}_t^{(k)} = -V(s_t) + r_t + \gamma r_{t+1} + \ldots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})$, which is a $k$-step return minus the baseline. As $k$ increases, the estimator relies less on the bootstrapped value $V$ and more on actually-observed rewards, trading lower bias for higher variance. At $k=1$ it is the pure TD residual (low variance, high bias); as $k \to \infty$ it becomes the full Monte Carlo advantage (zero bias, high variance).</span>

---

## <span style="font-size: 16px;">The GAE Formula</span>

<span style="font-size: 14px;">Rather than commit to a single $k$, GAE takes an **exponentially-weighted average** of all $k$-step estimators with weights $(1-\lambda)\lambda^{k-1}$:</span>

$$
\hat{A}_t^{GAE(\gamma,\lambda)} = (1 - \lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}
$$

<span style="font-size: 14px;">After simplification this telescopes into a strikingly simple form: an exponentially-discounted sum of the future TD residuals:</span>

$$
\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}
$$

<span style="font-size: 14px;">This is the central result of the paper. The advantage at time $t$ is just the future stream of TD errors, geometrically discounted by the combined factor $\gamma\lambda$. The discount $\gamma$ handles the usual time-value of reward; the new factor $\lambda$ additionally downweights residuals further in the future, controlling how much the estimate trusts long bootstrapped chains.</span>

---

## <span style="font-size: 16px;">Deriving the Telescoping Sum</span>

<span style="font-size: 14px;">The collapse from a weighted average of $k$-step estimators to a single discounted residual stream is worth seeing in full, because it explains why GAE is so cheap to compute. Write the $k$-step estimator as $\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}$ and substitute into the exponentially-weighted average:</span>

$$
\hat{A}_t^{GAE} = (1-\lambda)\sum_{k=1}^{\infty} \lambda^{k-1} \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}
$$

<span style="font-size: 14px;">Swapping the order of summation, a given residual $\gamma^l \delta_{t+l}$ appears in every $k$-step estimator with $k > l$, that is for $k = l+1, l+2, \ldots$. Collecting its total weight gives a geometric series:</span>

$$
(1-\lambda)\sum_{k=l+1}^{\infty} \lambda^{k-1} = (1-\lambda)\,\lambda^{l}\sum_{j=0}^{\infty}\lambda^{j} = (1-\lambda)\,\lambda^{l}\,\frac{1}{1-\lambda} = \lambda^{l}
$$

<span style="font-size: 14px;">So residual $\delta_{t+l}$ ends up with combined weight $\gamma^l \lambda^l = (\gamma\lambda)^l$, yielding $\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$. The normalization constant $(1-\lambda)$ exists precisely so the infinite set of estimator weights sums to one, keeping GAE an honest weighted average rather than an arbitrary rescaling.</span>

---

## <span style="font-size: 16px;">The Recursive Backward Form</span>

<span style="font-size: 14px;">The infinite sum is computed efficiently with a single backward pass using the recursion:</span>

$$
A_t^{GAE} = \delta_t + \gamma\lambda\, A_{t+1}^{GAE}
$$

<span style="font-size: 14px;">with terminal condition $A_T^{GAE} = 0$ and the bootstrap value $V_T \equiv \texttt{last\_value}$. The procedure is:</span>

<span style="font-size: 14px;">1. **Compute each residual**: for $t$ from $0$ to $T-1$, $\delta_t = r_t + \gamma V_{t+1} - V_t$, where $V_T$ is the bootstrap value of the state following the last collected step.</span>

<span style="font-size: 14px;">2. **Initialize**: set the running advantage $A = 0$ for the step after the horizon.</span>

<span style="font-size: 14px;">3. **Iterate backward**: for $t$ from $T-1$ down to $0$, set $A_t = \delta_t + \gamma\lambda\, A$, then update the running value $A \leftarrow A_t$.</span>

<span style="font-size: 14px;">This runs in $O(T)$ time and accumulates the discounted residual stream exactly, because unrolling the recursion reproduces $\sum_l (\gamma\lambda)^l \delta_{t+l}$.</span>

---

## <span style="font-size: 16px;">The Bias-Variance Knob</span>

<span style="font-size: 14px;">The parameter $\lambda$ interpolates the two extremes:</span>

* <span style="font-size: 14px;">**$\lambda = 0$**: the sum collapses to $A_t = \delta_t = r_t + \gamma V_{t+1} - V_t$, the pure one-step TD advantage. Lowest variance, highest bias. The estimate trusts the value function fully after one step.</span>
* <span style="font-size: 14px;">**$\lambda = 1$**: the factor becomes $\gamma^l$ and the sum equals $\sum_l \gamma^l \delta_{t+l} = G_t - V(s_t)$, the Monte Carlo advantage. Highest variance, zero bias (given correct discounting). The estimate ignores intermediate value estimates entirely.</span>
* <span style="font-size: 14px;">**$0 < \lambda < 1$**: a smooth blend. Each additional step of bootstrapping is downweighted by $\lambda$, so nearby residuals (more reliable, less compounded value error) dominate while distant residuals contribute progressively less.</span>

<span style="font-size: 14px;">Typical values are $\lambda \in [0.9, 0.99]$, with $\gamma$ around $0.99$. The paper reports that this range gives the best empirical trade-off, accepting a small bias from bootstrapping in exchange for a large reduction in variance relative to pure Monte Carlo, which is what makes high-dimensional continuous control tractable.</span>

<span style="font-size: 14px;">Intuitively, $\lambda$ answers the question "how far ahead do I trust the actually-observed rewards before I fall back on my value estimate?". A high $\lambda$ says the value function is untrustworthy, so look deep into the real reward stream. A low $\lambda$ says the value function is good, so bootstrap aggressively after a step or two. Early in training $V$ is poor and a higher $\lambda$ can help, while a well-trained $V$ tolerates a lower $\lambda$; in practice a single fixed value in the standard range works well and is rarely annealed.</span>

---

## Worked Example ($T = 3$, $\gamma = 0.99$, $\lambda = 0.95$)

<span style="font-size: 14px;">Let rewards $r = [1, 1, 1]$, values $V = [0.5, 0.6, 0.7]$, and bootstrap $V_3 = \texttt{last\_value} = 0.0$.</span>

<span style="font-size: 14px;">1. **Residuals**: $\delta_2 = 1 + 0.99(0.0) - 0.7 = 0.3$; $\delta_1 = 1 + 0.99(0.7) - 0.6 = 1.093$; $\delta_0 = 1 + 0.99(0.6) - 0.5 = 1.094$.</span>

<span style="font-size: 14px;">2. **Backward pass**, with $\gamma\lambda = 0.99 \times 0.95 = 0.9405$. Start $A = 0$.</span>

<span style="font-size: 14px;">3. **$t=2$**: $A_2 = 0.3 + 0.9405(0) = 0.3$.</span>

<span style="font-size: 14px;">4. **$t=1$**: $A_1 = 1.093 + 0.9405(0.3) = 1.3752$.</span>

<span style="font-size: 14px;">5. **$t=0$**: $A_0 = 1.094 + 0.9405(1.3752) = 2.3873$.</span>

<span style="font-size: 14px;">The advantages decay as the horizon shrinks, and earlier steps accumulate more discounted future residuals.</span>

---

## <span style="font-size: 16px;">Connection to TD(λ) and Eligibility Traces</span>

<span style="font-size: 14px;">GAE is the policy-gradient analogue of **TD($\lambda$)** from value-based reinforcement learning. The $\lambda$-return in TD($\lambda$) is an exponentially-weighted average of $n$-step returns, and GAE applies the identical weighting scheme to $n$-step **advantages** instead. The backward recursion mirrors the **eligibility trace** view of TD($\lambda$): each TD residual $\delta_t$ propagates backward through time, decaying by $\gamma\lambda$ at each step, assigning credit to the states and actions that preceded it. The paper makes this connection explicit, presenting GAE as a principled extension of the $\lambda$-return idea to the actor's gradient.</span>

<span style="font-size: 14px;">A second discount interpretation in the paper treats $\gamma$ not only as the problem's reward discount but also as a variance-reduction parameter in its own right: a smaller effective horizon ($\gamma$ closer to zero) reduces variance at the cost of ignoring long-term reward. GAE separates the two roles, letting $\gamma$ encode the true objective while $\lambda$ independently tunes the estimator's bias and variance, which is why both factors appear together as $\gamma\lambda$ in the accumulation but $\gamma$ alone appears in the residual.</span>

---

## <span style="font-size: 16px;">Why Bootstrapping Introduces Bias</span>

<span style="font-size: 14px;">The bias in GAE for $\lambda < 1$ comes entirely from the value function $V$ being imperfect. Each TD residual $\delta_t = r_t + \gamma V_{t+1} - V_t$ substitutes the learned estimate $V_{t+1}$ for the true expected future return. If $V$ is systematically off, that error propagates into every residual and hence into the advantage. The further out a residual sits, the more compounded value error it can carry, which is exactly why $\lambda$ downweights distant residuals: it limits how much the estimate leans on long chains of potentially-wrong bootstrapped values.</span>

<span style="font-size: 14px;">In the limit $\lambda = 1$ no bootstrapping survives between consecutive steps, the telescoping sum reduces to $G_t - V(s_t)$, and the only $V$ that appears is the single subtracted baseline $V(s_t)$, which (as in REINFORCE with a baseline) does not bias the estimate. This is why $\lambda = 1$ recovers an unbiased Monte Carlo advantage regardless of how poor $V$ is, paying for it with maximal variance.</span>

---

## <span style="font-size: 16px;">Role in PPO and Value Targets</span>

<span style="font-size: 14px;">GAE produces the advantage that weights the policy gradient or the PPO clipped surrogate. The same computation also yields the **value targets**: the return-to-go used to regress the critic is recovered as $\hat{R}_t = A_t^{GAE} + V(s_t)$, so a single backward pass supplies both the actor's advantage and the critic's regression target. Advantages are typically **normalized** to zero mean and unit variance across the batch before entering the policy loss, which further stabilizes the gradient scale, a step almost always paired with GAE in practice.</span>

<span style="font-size: 14px;">Defining the value target as $\hat{R}_t = A_t^{GAE} + V(s_t)$ rather than the raw Monte Carlo return is a deliberate and useful choice. It makes the critic's target consistent with the same $\lambda$-weighting used for the actor, so the value function is trained toward the $\lambda$-return rather than the full return. Because the current $V(s_t)$ is detached and added back, this target is sometimes called the **TD($\lambda$) target**, and it inherits GAE's reduced variance: lower-variance targets give a more stable critic, which in turn produces more accurate residuals on the next iteration, a virtuous loop. PPO implementations therefore store the GAE advantages and the implied returns together at rollout time and reuse both across the multiple optimization epochs performed on each batch of data.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Mishandling the bootstrap value at truncation.** When an episode is cut off rather than terminated, $V_T$ must be the value of the next state ($\texttt{last\_value}$). When the episode genuinely terminates, that value is zero. Using the wrong one biases every advantage in the trajectory through the recursion.</span>
* <span style="font-size: 14px;">**Iterating forward instead of backward.** The recursion $A_t = \delta_t + \gamma\lambda A_{t+1}$ depends on the future advantage, so it must be computed from $t = T-1$ down to $0$. A forward loop produces nonsense because $A_{t+1}$ is not yet available.</span>
* <span style="font-size: 14px;">**Confusing $\gamma$ and $\gamma\lambda$ in the recursion.** The residual $\delta_t$ uses $\gamma$ alone for bootstrapping $V_{t+1}$, but the accumulation across steps uses the combined factor $\gamma\lambda$. Swapping these silently changes the bias-variance trade-off and breaks the equivalence with the $k$-step derivation.</span>
* <span style="font-size: 14px;">**Resetting the running advantage across episode boundaries.** In batched rollouts spanning multiple episodes, the running $A$ must be reset to zero at each terminal step, otherwise advantages leak across unrelated episodes through the recursion.</span>

---