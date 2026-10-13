# <span style="font-size: 20px;">TD(0) Value Update</span>

<span style="font-size: 14px;">TD(0) is the simplest **temporal-difference** learning algorithm and the prototype for almost all modern value-based reinforcement learning. It estimates the state-value function $V^\pi$ of a policy from experience, updating $V(s)$ after every single step using a **bootstrapped** target rather than waiting for the full return at the end of the episode.</span>

---

## <span style="font-size: 16px;">The Update Rule</span>

<span style="font-size: 14px;">Each time the agent is in state $s$, takes a step, receives reward $r$, and lands in next state $s'$, TD(0) applies:</span>

$$
V(s) \leftarrow V(s) + \alpha \, \bigl[ r + \gamma \, V(s') - V(s) \bigr]
$$

<span style="font-size: 14px;">where $\alpha \in (0,1]$ is the **learning rate** (step size) and $\gamma \in [0,1]$ is the discount factor. The quantity in brackets is the **TD error**:</span>

$$
\delta = r + \gamma \, V(s') - V(s)
$$

<span style="font-size: 14px;">The update can be read as: move $V(s)$ a fraction $\alpha$ of the way from its current value toward the **TD target** $r + \gamma V(s')$. If $\delta > 0$ the target is higher than the current estimate and $V(s)$ increases; if $\delta < 0$ it decreases. At a fixed point all TD errors are zero in expectation, which is exactly the Bellman expectation equation being satisfied.</span>

---

## <span style="font-size: 16px;">Bootstrapping: The Key Idea</span>

<span style="font-size: 14px;">The defining feature of TD(0) is **bootstrapping**: the target $r + \gamma V(s')$ uses the agent's own current estimate $V(s')$ of the next state in place of the unknown true return from $s'$. TD does not wait to observe the actual discounted sum of all future rewards. It takes one real reward $r$, then trusts its existing value table for everything beyond.</span>

<span style="font-size: 14px;">This is what separates TD from Monte Carlo. MC's target is the full return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$, a quantity measured entirely from data once the episode ends. TD's target replaces the tail $\gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ with $\gamma V(s')$. Bootstrapping is the same mechanism dynamic programming uses, but DP requires a model to compute the expectation while TD samples a single transition instead.</span>

<span style="font-size: 14px;">The payoff is enormous: because TD only needs one transition, it can learn **online**, after every step, and works on **continuing** (non-terminating) tasks where a full return never exists. MC must wait for termination; TD never does.</span>

---

## <span style="font-size: 16px;">The TD Error as a Prediction Error</span>

<span style="font-size: 14px;">The TD error $\delta$ measures how surprised the agent is. Before the step, it predicted the return from $s$ to be $V(s)$. After seeing $r$ and the estimated value of where it landed, its revised one-step estimate is $r + \gamma V(s')$. The difference $\delta$ is the correction. A positive $\delta$ means things went better than expected.</span>

<span style="font-size: 14px;">This interpretation is more than intuition. The TD error is the biological and computational signal at the heart of reward-prediction-error theories of dopamine, and it is the elementary building block of the entire TD($\lambda$) family, actor-critic methods, and deep RL targets. Every value-based algorithm in this section computes some form of $\delta$ and steps its estimate in its direction.</span>

---

## <span style="font-size: 16px;">Online, In-Place Updates</span>

<span style="font-size: 14px;">TD(0) is applied **in place**: each transition reads the current value table, computes $\delta$, and immediately writes the new $V(s)$. Later transitions therefore see the values that earlier transitions have already changed. This matters when a state is updated more than once in a sequence, or when a transition's $s'$ was the $s$ of an earlier update, because the bootstrapped target $V(s')$ reflects the most recent write.</span>

<span style="font-size: 14px;">Processing transitions in order with in-place writes is the standard online formulation. It differs subtly from a batch update that would freeze all values, compute every $\delta$ against the old table, and apply them together. The online version is what propagates information through the state space step by step.</span>

---

## Worked Example ($\alpha = 0.5$, $\gamma = 0.9$)

<span style="font-size: 14px;">Take a three-state chain with initial values $V = [0,\ 0,\ 0]$ and the transition sequence (each is $s \to s'$ with reward $r$):</span>

* <span style="font-size: 14px;">Transition 1: $s = 0$, $r = 1$, $s' = 1$</span>
* <span style="font-size: 14px;">Transition 2: $s = 1$, $r = 2$, $s' = 2$</span>
* <span style="font-size: 14px;">Transition 3: $s = 0$, $r = 1$, $s' = 1$</span>

<span style="font-size: 14px;">**Transition 1**: $\delta = 1 + 0.9 \cdot V(1) - V(0) = 1 + 0 - 0 = 1$. Update $V(0) \leftarrow 0 + 0.5 \cdot 1 = 0.5$. Table now $[0.5,\ 0,\ 0]$.</span>

<span style="font-size: 14px;">**Transition 2**: $\delta = 2 + 0.9 \cdot V(2) - V(1) = 2 + 0 - 0 = 2$. Update $V(1) \leftarrow 0 + 0.5 \cdot 2 = 1.0$. Table now $[0.5,\ 1.0,\ 0]$.</span>

<span style="font-size: 14px;">**Transition 3**: now $V(1) = 1.0$ from the previous write, so $\delta = 1 + 0.9 \cdot 1.0 - 0.5 = 1 + 0.9 - 0.5 = 1.4$. Update $V(0) \leftarrow 0.5 + 0.5 \cdot 1.4 = 1.2$. Table now $[1.2,\ 1.0,\ 0]$.</span>

<span style="font-size: 14px;">The example shows bootstrapping in action: the third update's target depends on the value $V(1)$ that the second transition learned. Information flows backward through the chain one step per update, which is precisely how TD(0) eventually propagates rewards to distant states.</span>

---

## <span style="font-size: 16px;">Bias and Variance Trade-off</span>

<span style="font-size: 14px;">TD(0) is **biased but low variance**, the mirror image of Monte Carlo:</span>

* <span style="font-size: 14px;">**Bias.** The target $r + \gamma V(s')$ uses an estimate $V(s')$ that is generally wrong early in training. Substituting an inaccurate value introduces bias into every update. Only at convergence, when $V$ equals the true $V^\pi$, does the target become unbiased.</span>
* <span style="font-size: 14px;">**Variance.** The target depends on only one random reward and one random next state, plus a deterministic table lookup. Compared to the full MC return, which compounds randomness over an entire trajectory, the TD target has far lower variance. This is why TD often learns much faster in practice despite its bias.</span>

<span style="font-size: 14px;">The bias from bootstrapping is the price for the variance reduction. In the long run TD(0) still converges to the correct $V^\pi$ under standard step-size conditions, so the bias vanishes asymptotically. The general principle, sampling reduces bias while bootstrapping reduces variance, is the central tension of the whole field, and TD(0) is its cleanest illustration.</span>

---

## <span style="font-size: 16px;">Connection to the Bellman Equation</span>

<span style="font-size: 14px;">TD(0) is a stochastic, sample-based way of solving the Bellman expectation equation for a fixed policy:</span>

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[ r + \gamma V^\pi(s') \mid s \right]
$$

<span style="font-size: 14px;">Dynamic programming turns this identity into an iterative assignment $V(s) \leftarrow \mathbb{E}_\pi[r + \gamma V(s')]$, which requires the full transition model to evaluate the expectation over all next states. TD(0) cannot compute that expectation, so it draws a single sample $(r, s')$ from the environment and treats $r + \gamma V(s')$ as a noisy, one-sample estimate of the right-hand side.</span>

<span style="font-size: 14px;">Each TD update is thus a stochastic-approximation step toward the DP fixed point. The learning rate $\alpha$ averages out the sampling noise over many visits, recovering the expectation implicitly. This is why TD is often described as combining the sampling of Monte Carlo with the bootstrapping of dynamic programming: it inherits the model-free data efficiency of one and the incremental, one-step structure of the other.</span>

---

## <span style="font-size: 16px;">Why TD Often Beats MC in Practice</span>

<span style="font-size: 14px;">Beyond the variance argument, TD(0) exploits the **Markov structure** of the problem. By bootstrapping it effectively builds a maximum-likelihood model of the underlying Markov process implicitly and solves it, whereas MC ignores the Markov property and treats each episode as an opaque return. On a problem that is genuinely Markovian, TD's estimates can be better than MC's for the same data because they reuse value information across overlapping trajectories.</span>

<span style="font-size: 14px;">A second practical advantage is **credit assignment speed**. When a state's value is corrected, every state that bootstraps off it benefits on its next visit. Reward information ratchets backward through the state space one link per update, so once a high-value region is discovered, neighbouring states learn quickly. MC must wait for a full episode to revisit each state before any of that information moves.</span>

<span style="font-size: 14px;">A subtlety worth internalizing: early in training TD estimates are systematically biased because they bootstrap off values that are still wrong, and they can sit far from $V^\pi$ for many updates. The method should be judged by its converged behaviour rather than its first few passes. As the value table improves, the bootstrap targets improve too, the bias shrinks, and the estimates tighten around the true values. This self-correcting dynamic, where better values feed better targets which feed better values, is what makes bootstrapping converge despite starting from arbitrary initial estimates.</span>

---

## <span style="font-size: 16px;">Convergence Conditions</span>

<span style="font-size: 14px;">For a fixed policy, tabular TD(0) converges to $V^\pi$ with probability one provided every state is visited infinitely often and the step sizes satisfy the Robbins-Monro conditions:</span>

$$
\sum_{t} \alpha_t = \infty, \qquad \sum_{t} \alpha_t^2 < \infty
$$

<span style="font-size: 14px;">The first condition ensures the steps are large enough to reach any value; the second ensures they shrink fast enough to suppress noise. A constant $\alpha$ violates the second condition, so constant-$\alpha$ TD(0) does not converge to a point but instead fluctuates around $V^\pi$ in a region whose size scales with $\alpha$. This is acceptable, and even desirable, in non-stationary problems where the agent should keep tracking a changing target.</span>

---

## <span style="font-size: 16px;">Step Size and Its Effects</span>

<span style="font-size: 14px;">The learning rate $\alpha$ controls how aggressively each TD error is absorbed. Its role is best seen by rewriting the update as a weighted average of the old estimate and the target:</span>

$$
V(s) \leftarrow (1 - \alpha) \, V(s) + \alpha \, \bigl[ r + \gamma V(s') \bigr]
$$

<span style="font-size: 14px;">A small $\alpha$ keeps most of the old estimate, so learning is slow but stable and the estimate averages over many samples, suppressing noise. A large $\alpha$ throws away history and tracks the latest target closely, which adapts fast but is jumpy and, combined with bootstrapping, risks amplifying errors. The choice mirrors an exponential moving average: $\alpha$ sets the effective memory length over which returns are blended.</span>

<span style="font-size: 14px;">In stationary problems a decaying schedule that satisfies the Robbins-Monro conditions gives the best of both worlds: large early steps for fast initial progress, shrinking later steps for a precise final estimate. In non-stationary problems a small constant $\alpha$ is preferred precisely because it never stops adapting.</span>

---

## <span style="font-size: 16px;">Relation to TD($\lambda$) and Control</span>

<span style="font-size: 14px;">TD(0) is the one-step end of a spectrum. An $n$-step return uses $n$ real rewards before bootstrapping, $r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$, trading more variance for less bias as $n$ grows. TD($\lambda$) averages all $n$-step returns with geometric weights, recovering TD(0) at $\lambda = 0$ and Monte Carlo at $\lambda = 1$.</span>

<span style="font-size: 14px;">TD(0) as written evaluates a fixed policy: it learns $V^\pi$, not how to act. Extending the same TD error to state-action values $Q$ and coupling it with a policy that acts greedily on $Q$ yields the control algorithms SARSA and Q-learning, which appear in the following problems. They differ only in how the bootstrap target's next-state value is formed.</span>

---

## <span style="font-size: 16px;">On-Policy Evaluation Scope</span>

<span style="font-size: 14px;">TD(0) as presented is **on-policy prediction**: the transitions are generated by the policy $\pi$ being evaluated, so the sampled $(r, s')$ pairs are distributed according to $\pi$'s dynamics and the expected target equals the Bellman backup for $\pi$. No importance-sampling correction is needed because the data and the evaluated policy match.</span>

<span style="font-size: 14px;">This scope is worth stating explicitly because the very next problems change exactly this aspect. SARSA keeps the on-policy character by bootstrapping off the action $\pi$ actually took next, while Q-learning becomes off-policy by bootstrapping off the greedy action regardless of what was taken. TD(0) for $V$ is the neutral starting point: it predicts the value of whatever policy generated the data, and the choice of how to form the next-state value is what later turns prediction into control.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Reading stale next-state values.** When updating in place, the target must read the **current** $V(s')$. A common bug is to snapshot the value table at the start and bootstrap against old values, which changes the result whenever a state is updated more than once in a sequence.</span>
* <span style="font-size: 14px;">**Forgetting the terminal-state convention.** At an episode's last transition, the target should treat the terminal next state as having value $0$: the target is just $r$. Bootstrapping against a nonzero terminal value leaks phantom future reward into the estimate.</span>
* <span style="font-size: 14px;">**Confusing the TD error sign or the update direction.** The error is $r + \gamma V(s') - V(s)$, and the update **adds** $\alpha \delta$ to $V(s)$. Flipping the subtraction order or subtracting the step pushes values the wrong way and diverges.</span>
* <span style="font-size: 14px;">**Using too large a learning rate.** A large $\alpha$ combined with bootstrapping can amplify errors faster than they are corrected, causing oscillation or divergence. Step sizes should be modest and ideally decayed over time.</span>

---