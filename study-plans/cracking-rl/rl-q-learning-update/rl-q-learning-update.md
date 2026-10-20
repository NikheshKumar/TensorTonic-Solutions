# <span style="font-size: 20px;">Q-Learning Update</span>

<span style="font-size: 14px;">Q-learning (Watkins, 1989) is the canonical **off-policy** temporal-difference control algorithm. It learns the optimal action-value function $Q^*$ directly from experience, while the agent follows any sufficiently exploratory behaviour policy. The defining move is that its bootstrap target uses the **greedy** next action, the one that maximizes $Q$ at the next state, regardless of what the agent actually did.</span>

---

## <span style="font-size: 16px;">The Update Rule</span>

<span style="font-size: 14px;">Given a transition $(s, a, r, s')$, Q-learning updates:</span>

$$
Q[s][a] \leftarrow Q[s][a] + \alpha \, \bigl[ r + \gamma \, \max_{a'} Q[s'][a'] - Q[s][a] \bigr]
$$

<span style="font-size: 14px;">where $\alpha \in (0,1]$ is the learning rate and $\gamma \in [0,1]$ is the discount factor. The bracketed quantity is the TD error:</span>

$$
\delta = r + \gamma \, \max_{a'} Q[s'][a'] - Q[s][a]
$$

<span style="font-size: 14px;">Notice the experience tuple has **no** $a'$. SARSA needed the actually-taken next action to form its target; Q-learning computes $\max_{a'} Q[s'][a']$ over all actions at $s'$ from the current $Q$-table, ignoring whatever the policy would have sampled. That single $\max$ is the entire difference between the two algorithms.</span>

---

## <span style="font-size: 16px;">Off-Policy: Two Policies at Once</span>

<span style="font-size: 14px;">Off-policy learning means the policy that **generates** the data (the behaviour policy) is distinct from the policy being **learned** (the target policy). Q-learning's target policy is the greedy policy $\arg\max_a Q(s,a)$, while its behaviour policy can be anything that explores enough, typically $\epsilon$-greedy. The $\max$ operator is what evaluates the greedy target policy: it asks "if I acted optimally from $s'$ onward, what would the value be?", even though the agent may have explored instead.</span>

<span style="font-size: 14px;">This decoupling is powerful. The agent can explore freely, take random actions, follow a demonstration, replay old experience from a buffer, and still converge to the optimal values. The learned $Q$ is unaffected by how exploratory the behaviour is, because the target never references the behaviour's chosen action. This property is exactly what makes Q-learning the backbone of Deep Q-Networks (Mnih et al., 2015), where experience replay feeds old, off-policy transitions into the update.</span>

---

## <span style="font-size: 16px;">The Bellman Optimality Equation</span>

<span style="font-size: 14px;">Q-learning is a sample-based solver for the **Bellman optimality equation**, not the expectation equation that SARSA targets:</span>

$$
Q^*(s, a) = \mathbb{E}\!\left[ r + \gamma \, \max_{a'} Q^*(s', a') \mid s, a \right]
$$

<span style="font-size: 14px;">The expectation is only over the reward and next state, not over a next action, because the optimal policy is deterministic and the $\max$ already commits to the best action. Q-learning draws a single sample of $r$ and $s'$, plugs in its current $\max_{a'} Q$, and steps toward that target. The learning rate averages the sampling noise over visits.</span>

<span style="font-size: 14px;">Contrast this with SARSA, whose target estimates $\mathbb{E}_\pi[r + \gamma Q(s', a')]$ with $a' \sim \pi$. SARSA tracks the value of the current policy; Q-learning tracks the value of the optimal policy. The $\max$ versus the policy-sampled $a'$ is precisely the difference between the optimality and expectation forms of the Bellman equation.</span>

---

## <span style="font-size: 16px;">Sampling and Bootstrapping</span>

<span style="font-size: 14px;">Q-learning is model-free and bootstrapped, like all TD control:</span>

* <span style="font-size: 14px;">**Sampling**: $r$ and $s'$ are observed from real interaction, so no transition model is required.</span>
* <span style="font-size: 14px;">**Bootstrapping**: the return's tail is replaced by $\gamma \max_{a'} Q[s'][a']$, the agent's own current estimate, allowing single-step online updates.</span>

<span style="font-size: 14px;">The bootstrapped target has low variance compared to a full Monte Carlo return, at the cost of bias from an imperfect $Q$. There is, however, a second bias unique to the $\max$: because Q-learning takes the maximum over noisy estimates, it systematically **overestimates** action values, a phenomenon called maximization bias. The $\max$ of unbiased-but-noisy estimates is larger than the true max in expectation. Double Q-learning (van Hasselt, 2010) corrects this by maintaining two $Q$-tables and using one to select the action and the other to evaluate it.</span>

<span style="font-size: 14px;">Compared with SARSA, Q-learning also tends to have slightly higher variance per update in stochastic settings, because the $\max$ can latch onto whichever next-state action is momentarily inflated, whereas SARSA's sampled $a'$ is smoothed by the policy's own averaging over many visits. The trade is deliberate: Q-learning accepts this in exchange for estimating the optimal values directly, sidestepping the need to anneal exploration that SARSA requires to reach the same target.</span>

---

## Worked Example ($\alpha = 0.5$, $\gamma = 0.9$)

<span style="font-size: 14px;">Two states $\{0, 1\}$, two actions $\{0, 1\}$, $Q$ initialized as $\begin{pmatrix} 0 & 0 \\ 1 & 3 \end{pmatrix}$ (rows are states, columns actions). Process two transitions $(s, a, r, s')$ in order:</span>

* <span style="font-size: 14px;">Transition 1: $(s{=}0,\ a{=}0,\ r{=}2,\ s'{=}1)$</span>
* <span style="font-size: 14px;">Transition 2: $(s{=}1,\ a{=}0,\ r{=}1,\ s'{=}0)$</span>

<span style="font-size: 14px;">**Transition 1**: the next state is $1$, and $\max_{a'} Q[1][a'] = \max(1, 3) = 3$. So $\delta = 2 + 0.9 \cdot 3 - Q[0][0] = 2 + 2.7 - 0 = 4.7$. Update $Q[0][0] \leftarrow 0 + 0.5 \cdot 4.7 = 2.35$. Table now $\begin{pmatrix} 2.35 & 0 \\ 1 & 3 \end{pmatrix}$.</span>

<span style="font-size: 14px;">**Transition 2**: next state is $0$, and $\max_{a'} Q[0][a'] = \max(2.35, 0) = 2.35$, reading the value the first update just wrote. So $\delta = 1 + 0.9 \cdot 2.35 - Q[1][0] = 1 + 2.115 - 1 = 2.115$. Update $Q[1][0] \leftarrow 1 + 0.5 \cdot 2.115 = 2.0575$. Final table $\begin{pmatrix} 2.35 & 0 \\ 2.0575 & 3 \end{pmatrix}$.</span>

<span style="font-size: 14px;">The example shows both the $\max$ in action, picking the larger next-state entry, and the in-place behaviour, where the second update's target reads the value the first update produced.</span>

---

## <span style="font-size: 16px;">Maximization Bias in Detail</span>

<span style="font-size: 14px;">The overestimation problem deserves a closer look because it is the most subtle property of the algorithm. Suppose the true action values at $s'$ are all equal to zero, but the estimates $Q[s'][a']$ are noisy, scattered around zero by estimation error. The true max is zero, yet the **max of the noisy estimates** is almost surely positive, because the maximum operator preferentially selects whichever action happened to be overestimated. Formally, for random variables $X_a$ with mean $\mu$, $\mathbb{E}[\max_a X_a] \geq \max_a \mathbb{E}[X_a]$ by Jensen's inequality applied to the convex max.</span>

<span style="font-size: 14px;">Q-learning uses the same estimates both to choose which action is best and to evaluate that action's value, so the positive bias compounds: the action chosen by the max tends to be one whose noise pushed it up, and its inflated value is then propagated backward through bootstrapping. In stochastic environments with many actions this can produce persistently optimistic values and a distorted policy. Double Q-learning breaks the coupling by learning two independent estimators $Q_A$ and $Q_B$: one selects the maximizing action, the other supplies its value, so the selection noise and evaluation noise are uncorrelated and the bias largely cancels.</span>

---

## <span style="font-size: 16px;">From Tabular to Deep Q-Networks</span>

<span style="font-size: 14px;">The tabular update generalizes directly to function approximation. Replacing the table with a parameterized $Q_\theta(s,a)$ and minimizing the squared TD error gives the loss behind Deep Q-Networks:</span>

$$
L(\theta) = \mathbb{E}\!\left[ \bigl( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \bigr)^2 \right]
$$

<span style="font-size: 14px;">Two engineering ideas make this stable, and both exploit Q-learning's off-policy nature. **Experience replay** stores transitions and samples random minibatches, breaking the temporal correlation between consecutive samples; this is only sound because the target's $\max$ does not depend on the behaviour that produced the stored data. A **target network** $\theta^-$ holds a slowly-updated copy of the parameters to compute the bootstrap target, preventing the moving target from chasing itself into divergence. The single tabular $\max$ update studied here is the conceptual core that these techniques are built around.</span>

---

## <span style="font-size: 16px;">Online In-Place Updates</span>

<span style="font-size: 14px;">Q-learning processes transitions one at a time against a live $Q$-table, so each update sees the effect of all earlier ones. This matters whenever a later transition's $s'$ has an entry an earlier transition changed, because the $\max$ reads the most recent value. Transitions must therefore be applied in order with in-place writes; a batched update against a frozen table would generally produce different numbers.</span>

<span style="font-size: 14px;">Because Q-learning is off-policy, the order and source of transitions are otherwise flexible. The same update applies whether the data is fresh, sampled from a replay buffer, or generated by an entirely different policy, which is why it scales naturally to the large off-policy training regimes used in deep reinforcement learning.</span>

<span style="font-size: 14px;">One subtlety in the in-place setting is that the value being maximized can change between when a state is a target and when it is later updated as a source. Early in training these $\max$ values are unreliable, so the bootstrap targets are biased, and the estimates can wander before they tighten. As the table improves the targets improve, the bias shrinks, and the values converge: the same self-correcting dynamic that makes all bootstrapping methods work despite arbitrary initialization.</span>

---

## <span style="font-size: 16px;">Convergence</span>

<span style="font-size: 14px;">Tabular Q-learning converges to $Q^*$ with probability one under two conditions. First, the Robbins-Monro step sizes:</span>

$$
\sum_{t} \alpha_t = \infty, \qquad \sum_{t} \alpha_t^2 < \infty
$$

<span style="font-size: 14px;">Second, every state-action pair must be visited infinitely often, which requires sufficient exploration in the behaviour policy. Crucially, the behaviour policy does **not** need to decay toward greedy for Q-learning to find $Q^*$: the $\max$ in the target already evaluates the optimal policy, so persistent exploration does not bias the learned values. This is a sharper guarantee than SARSA's, which needs the GLIE condition (decaying $\epsilon$) to converge to the optimal policy rather than the best $\epsilon$-soft one.</span>

---

## <span style="font-size: 16px;">Why Off-Policy Is Desirable</span>

<span style="font-size: 14px;">Decoupling the learned policy from the behaviour policy unlocks several capabilities that on-policy methods lack. The agent can **reuse data**: every transition ever collected remains valid training material, since the target never depends on the policy that generated it. It can **learn from others**: demonstrations, logged data, or a hand-crafted controller can all drive learning toward the optimal policy. And it can **explore aggressively without penalty to the learned values**, because the $\max$ always evaluates optimal continuation regardless of how reckless the exploration was.</span>

<span style="font-size: 14px;">These advantages are not free. Off-policy bootstrapping with function approximation can diverge, the so-called deadly triad of off-policy learning, bootstrapping, and approximation together being unstable. In the tabular setting this problem addresses, no such instability arises and convergence to $Q^*$ is guaranteed, but the same $\max$ update that is rock-solid in a table requires careful stabilization once a neural network replaces it.</span>

---

## <span style="font-size: 16px;">Q-Learning vs SARSA</span>

<span style="font-size: 14px;">The two algorithms differ only in the next-state value of the target:</span>

* <span style="font-size: 14px;">**SARSA (on-policy)**: target $r + \gamma Q(s', a')$ uses the actually-taken $a'$, evaluating the behaviour policy including exploration.</span>
* <span style="font-size: 14px;">**Q-learning (off-policy)**: target $r + \gamma \max_{a'} Q(s', a')$ uses the greedy action, evaluating the optimal policy regardless of behaviour.</span>

<span style="font-size: 14px;">When the policy is greedy with no exploration, the taken action equals the greedy action and the two algorithms coincide. They diverge because of exploratory actions: in the cliff-walking task Q-learning learns the optimal edge path but falls off frequently during training, while SARSA learns a safer route that yields more reward while still exploring. Expected SARSA, the next problem, replaces the sampled $a'$ with the policy expectation $\sum_{a'} \pi(a'|s') Q(s',a')$, and recovers Q-learning exactly when $\pi$ is the greedy policy.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the taken action instead of the $\max$.** Bootstrapping off $Q(s', a')$ for some specific $a'$ turns Q-learning into SARSA. The target must take the maximum over all next-state actions; using a sampled action makes the algorithm on-policy and changes what it converges to.</span>
* <span style="font-size: 14px;">**Forgetting the terminal-state convention.** At an episode's final transition there is no future, so $\max_{a'} Q[s'][a']$ must be treated as zero and the target reduces to $r$. Taking the max over a terminal or absorbing state's stale row leaks phantom value into the estimate.</span>
* <span style="font-size: 14px;">**Reading stale $Q$ values.** Updates are in place; the $\max$ in the target must read the current $Q[s']$, including changes earlier transitions wrote. Snapshotting the table gives wrong results when entries are revisited within a sequence.</span>
* <span style="font-size: 14px;">**Ignoring maximization bias.** The $\max$ over noisy estimates systematically overestimates values, which can slow learning or distort the policy in stochastic environments. When this matters, double Q-learning decouples action selection from evaluation to remove the bias.</span>

---