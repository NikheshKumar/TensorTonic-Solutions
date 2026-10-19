# <span style="font-size: 20px;">Monte Carlo Policy Evaluation</span>

<span style="font-size: 14px;">Monte Carlo (MC) policy evaluation estimates the **state-value function** $V^\pi(s)$ of a fixed policy $\pi$ by averaging the actual returns observed after visiting each state. It is model-free: it needs no knowledge of the transition probabilities or reward function, only the ability to sample complete episodes by following $\pi$.</span>

---

## <span style="font-size: 16px;">What the Value Function Is</span>

<span style="font-size: 14px;">The value of a state under a policy is the expected discounted return obtained by starting in that state and acting according to $\pi$ thereafter:</span>

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[ G_t \mid S_t = s \right], \quad G_t = \sum_{k=0}^{T-t-1} \gamma^k \, r_{t+k}
$$

<span style="font-size: 14px;">Here $G_t$ is the **return** from timestep $t$: the discounted sum of all rewards received until the episode terminates at $T$. The discount $\gamma \in [0,1]$ trades off immediate against future reward. With $\gamma = 0$ the value is just the next reward; as $\gamma \to 1$ the agent weighs distant rewards almost equally.</span>

<span style="font-size: 14px;">MC replaces the expectation $\mathbb{E}_\pi[\cdot]$ with a **sample average**. The law of large numbers guarantees that as the number of observed returns grows, the sample mean converges to the true expectation. This is the entire conceptual basis of the method: do not compute the expectation analytically, estimate it by sampling.</span>

---

## <span style="font-size: 16px;">Sampling, Not Bootstrapping</span>

<span style="font-size: 14px;">MC is the purest **sampling** method in reinforcement learning. To estimate $V^\pi(s)$ it uses the full return $G_t$, a real number measured from data, all the way to episode termination. It never substitutes its own current value estimate of any other state into the target. This is the defining contrast with temporal-difference (TD) methods, which **bootstrap** by using $V(s')$ as part of the target.</span>

<span style="font-size: 14px;">The consequences of using full returns are central to understanding MC:</span>

* <span style="font-size: 14px;">**No model required.** Because $G_t$ is observed directly, transition dynamics and reward functions are never touched. MC learns purely from experience.</span>
* <span style="font-size: 14px;">**Episodic only.** A complete return requires the episode to end. MC in its basic form cannot be applied to continuing (non-terminating) tasks, because $G_t$ would never be fully observed.</span>
* <span style="font-size: 14px;">**Each estimate is independent of others.** Updating $V(s)$ never depends on the accuracy of $V(s')$. Errors do not propagate between states the way they do under bootstrapping.</span>

---

## <span style="font-size: 16px;">First-Visit vs Every-Visit</span>

<span style="font-size: 14px;">Within a single episode a state may be visited multiple times. The two MC variants differ in which visits contribute a return to the running average:</span>

* <span style="font-size: 14px;">**First-visit MC** averages the returns following only the **first** time $s$ is entered in each episode. Later visits to $s$ in the same episode are ignored. This makes each contributing return an independent and identically distributed sample of $V^\pi(s)$, which simplifies the statistical analysis: the estimate is provably unbiased and converges as $1/\sqrt{n}$ in the number of episodes.</span>
* <span style="font-size: 14px;">**Every-visit MC** averages the returns following **every** occurrence of $s$. Returns from the same episode are correlated (they share trailing rewards), so the classic unbiasedness argument does not apply directly, yet every-visit MC also converges to $V^\pi(s)$ and often uses data more efficiently.</span>

<span style="font-size: 14px;">The first-visit estimate for state $s$ is the average over all (episode, first-visit timestep) pairs $\mathcal{F}(s)$:</span>

$$
V(s) = \frac{1}{|\mathcal{F}(s)|} \sum_{(\text{ep}, t) \in \mathcal{F}(s)} G_t^{(\text{ep})}
$$

<span style="font-size: 14px;">A state that never appears in any episode has $|\mathcal{F}(s)| = 0$ and its value is left at the default $0.0$.</span>

---

## <span style="font-size: 16px;">The Algorithm</span>

<span style="font-size: 14px;">First-visit MC processes one episode at a time. The standard trick is to compute returns **backwards** from the end of the episode, accumulating $G$ so each return costs constant work:</span>

<span style="font-size: 14px;">1. **Initialize**: for every state keep a running sum of returns $\text{Sum}(s) = 0$ and a count $N(s) = 0$.</span>

<span style="font-size: 14px;">2. **For each episode**, set $G \leftarrow 0$ and iterate from the last timestep $t = T-1$ down to $0$, updating $G \leftarrow r_t + \gamma G$ at each step. After this update $G$ equals $G_t$, the return from timestep $t$.</span>

<span style="font-size: 14px;">3. **First-visit check**: if state $S_t$ does not appear at any earlier timestep $0, 1, \ldots, t-1$ in this episode, it is a first visit. Add $G_t$ to $\text{Sum}(S_t)$ and increment $N(S_t)$.</span>

<span style="font-size: 14px;">4. **Final estimate**: $V(s) = \text{Sum}(s) / N(s)$ for each state with $N(s) > 0$, else $0.0$.</span>

<span style="font-size: 14px;">The backward accumulation is what makes the algorithm efficient: computing each $G_t$ from scratch would be quadratic in episode length, but the recurrence $G_t = r_t + \gamma G_{t+1}$ makes the whole pass linear.</span>

---

## <span style="font-size: 16px;">Incremental Update Form</span>

<span style="font-size: 14px;">Storing all returns is unnecessary. The same average can be maintained incrementally, which is the form most implementations and textbooks use:</span>

$$
V(s) \leftarrow V(s) + \frac{1}{N(s)} \left[ G_t - V(s) \right]
$$

<span style="font-size: 14px;">This is a running mean: the new return $G_t$ nudges the estimate toward itself by a step size $1/N(s)$ that shrinks as more samples arrive. Replacing $1/N(s)$ with a fixed step size $\alpha$ gives a **constant-$\alpha$ MC** update that tracks a non-stationary target by forgetting old returns, at the cost of remaining a moving (no longer exact) average.</span>

---

## Worked Example ($\gamma = 1$, one episode)

<span style="font-size: 14px;">Consider a three-state chain with states $\{0, 1, 2\}$ and a single episode generated by $\pi$:</span>

<span style="font-size: 14px;">states $= [0, 1, 0, 2]$, rewards $= [1, 0, 2, 3]$, terminating after the reward $3$.</span>

<span style="font-size: 14px;">Compute returns backwards with $\gamma = 1$, so $G_t = r_t + G_{t+1}$:</span>

* <span style="font-size: 14px;">$t = 3$, state $2$: $G_3 = 3$</span>
* <span style="font-size: 14px;">$t = 2$, state $0$: $G_2 = 2 + 3 = 5$</span>
* <span style="font-size: 14px;">$t = 1$, state $1$: $G_1 = 0 + 5 = 5$</span>
* <span style="font-size: 14px;">$t = 0$, state $0$: $G_0 = 1 + 5 = 6$</span>

<span style="font-size: 14px;">Now apply the first-visit rule. State $0$ appears at $t = 0$ and $t = 2$; only $t = 0$ counts, contributing $G_0 = 6$. State $1$ first appears at $t = 1$, contributing $G_1 = 5$. State $2$ first appears at $t = 3$, contributing $G_3 = 3$. With one episode each average equals its single sample:</span>

* <span style="font-size: 14px;">$V(0) = 6.0$ (the later visit's return $5$ is discarded)</span>
* <span style="font-size: 14px;">$V(1) = 5.0$</span>
* <span style="font-size: 14px;">$V(2) = 3.0$</span>

<span style="font-size: 14px;">An every-visit estimate would instead set $V(0) = (6 + 5)/2 = 5.5$, illustrating concretely how the two variants diverge on a single episode while both converging to the same limit over many.</span>

---

## <span style="font-size: 16px;">Why Sampling Works: Convergence</span>

<span style="font-size: 14px;">Each first-visit return for a state $s$ is an independent draw from the distribution of returns whose mean is exactly $V^\pi(s)$. Writing the $n$-th such return as $G^{(n)}$, the estimate after $n$ episodes is</span>

$$
V_n(s) = \frac{1}{n} \sum_{i=1}^{n} G^{(i)}
$$

<span style="font-size: 14px;">By the strong law of large numbers $V_n(s) \to V^\pi(s)$ almost surely as $n \to \infty$. The standard error of the mean shrinks like $\sigma_G / \sqrt{n}$, where $\sigma_G$ is the standard deviation of the return distribution. Two facts follow directly. First, convergence is guaranteed but slow: halving the error requires four times as many episodes. Second, the rate is governed entirely by $\sigma_G$, so any environment that produces high-variance returns will demand many more episodes for a given accuracy.</span>

<span style="font-size: 14px;">This $1/\sqrt{n}$ behaviour is the same as estimating the mean of any random variable by sampling, which is exactly what MC is doing. There is no special reinforcement-learning machinery in the convergence argument; the only RL-specific content is the definition of the return as the quantity being averaged.</span>

---

## <span style="font-size: 16px;">Relation to Dynamic Programming and TD</span>

<span style="font-size: 14px;">It is useful to place MC against the two neighbouring families of value-estimation methods, because their differences are exactly the design axes this problem highlights.</span>

* <span style="font-size: 14px;">**Dynamic programming** computes $V^\pi$ by solving the Bellman expectation equation $V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[r + \gamma V^\pi(s')]$. It bootstraps fully and requires a known model $P$. No sampling is involved.</span>
* <span style="font-size: 14px;">**Temporal difference TD($0$)** samples one step and bootstraps the rest: target $r + \gamma V(s')$. It needs no model and updates online, after every step.</span>
* <span style="font-size: 14px;">**Monte Carlo** samples the entire return and bootstraps nothing: target $G_t$. It needs no model but updates only after the episode ends.</span>

<span style="font-size: 14px;">MC and TD therefore sit at opposite ends of a spectrum. MC has zero bootstrapping and maximal sampling depth; TD($0$) has maximal bootstrapping and minimal sampling depth. TD($\lambda$) interpolates continuously between them by mixing $n$-step returns with geometric weights $\lambda$, recovering MC at $\lambda = 1$ and TD($0$) at $\lambda = 0$.</span>

---

## <span style="font-size: 16px;">On-Policy Evaluation</span>

<span style="font-size: 14px;">The plain MC estimate above is **on-policy**: the episodes must be generated by the very policy $\pi$ whose value is being estimated. The returns are samples from $\pi$'s own trajectory distribution, so their mean is $V^\pi$ by construction.</span>

<span style="font-size: 14px;">Estimating $V^\pi$ from data collected under a different behaviour policy $b$ requires **off-policy** MC with importance-sampling weights $\rho = \prod_t \frac{\pi(a_t|s_t)}{b(a_t|s_t)}$ applied to each return. This reweights samples from $b$'s distribution to look like samples from $\pi$'s, at the cost of additional variance from the weight product. The problem here is the on-policy case, where every weight is $1$ and no correction is needed.</span>

---

## <span style="font-size: 16px;">Bias and Variance</span>

<span style="font-size: 14px;">MC's statistical signature is **unbiased but high variance**. Because the target $G_t$ is an actual return, $\mathbb{E}[G_t \mid S_t = s] = V^\pi(s)$ exactly, with no approximation introduced by the estimator. The first-visit average inherits this unbiasedness from the start.</span>

<span style="font-size: 14px;">The cost is variance. A single return depends on the entire trajectory of random actions, transitions, and rewards from $t$ to termination. Long episodes compound many sources of randomness into one number, so individual returns scatter widely around their mean. Many episodes are needed before the average settles. TD methods accept a small bootstrapping bias in exchange for far lower variance, since their target $r + \gamma V(s')$ depends on only one random step plus a (low-variance) estimate.</span>

<span style="font-size: 14px;">A practical implication: in environments with long horizons or sparse, late rewards, MC estimates can be very noisy, and the high variance directly slows learning. This is the core trade-off that motivates TD($0$) and the spectrum of TD($\lambda$) methods that interpolate between MC and one-step TD.</span>

---

## <span style="font-size: 16px;">Practical Considerations</span>

<span style="font-size: 14px;">A few details matter when MC evaluation is run at scale or used as a building block inside control:</span>

* <span style="font-size: 14px;">**Exploration coverage.** A state is only learned if $\pi$ actually visits it. Deterministic policies may never reach large parts of the state space, leaving those values stuck at the default. In control settings this motivates exploring starts or $\epsilon$-soft policies so every state-action pair is visited infinitely often.</span>
* <span style="font-size: 14px;">**Memory.** The incremental update keeps only $V(s)$ and $N(s)$ per state, so memory is $O(|\mathcal{S}|)$ regardless of how many episodes are processed. Storing raw returns is never necessary.</span>
* <span style="font-size: 14px;">**Action values.** Replacing states with state-action pairs gives MC estimation of $Q^\pi(s,a)$, the quantity needed for control. The mechanics are identical: average the returns following the first visit to each $(s,a)$ pair. This is the bridge from MC evaluation to MC control via generalized policy iteration.</span>
* <span style="font-size: 14px;">**Variance reduction.** Baselines and control variates can cut the variance of returns without introducing bias, partially addressing MC's main weakness when full TD-style bootstrapping is undesirable.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Computing returns forward instead of backward.** The recurrence is $G_t = r_t + \gamma G_{t+1}$, which only works iterating from the last timestep to the first. Iterating forward and trying to discount on the fly is a common source of off-by-one and mis-discounting bugs; always accumulate $G$ in reverse.</span>
* <span style="font-size: 14px;">**Counting every visit when first-visit is intended.** Forgetting the "have I seen this state earlier in the episode" check turns first-visit MC into every-visit MC. They converge to the same value but give different finite-sample estimates, so a test expecting first-visit results will fail.</span>
* <span style="font-size: 14px;">**Dividing by zero for unvisited states.** A state that never appears has count $0$. Guard the final division and assign the default value (here $0.0$) rather than producing NaN.</span>
* <span style="font-size: 14px;">**Applying MC to non-terminating tasks.** A full return requires the episode to end. On a continuing task or an episode truncated mid-way, the return is undefined or biased, and the unbiasedness guarantee no longer holds.</span>

---