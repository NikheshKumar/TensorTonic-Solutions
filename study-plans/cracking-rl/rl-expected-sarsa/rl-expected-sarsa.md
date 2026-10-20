# <span style="font-size: 20px;">Expected SARSA</span>

<span style="font-size: 14px;">Expected SARSA is the variance-reduced cousin of SARSA. Instead of bootstrapping from a single sampled next action $a'$, it bootstraps from the **expected** action value at the next state under the policy $\pi$. Replacing the random sample with its exact expectation removes the noise that sampling injects into the target, yielding lower-variance, more stable updates while inheriting SARSA's structure.</span>

---

## <span style="font-size: 16px;">The Update Rule</span>

<span style="font-size: 14px;">Given a transition $(s, a, r, s')$ and the policy $\pi$, Expected SARSA updates:</span>

$$
Q[s][a] \leftarrow Q[s][a] + \alpha \, \left[ r + \gamma \sum_{a'} \pi(a' \mid s')\, Q[s'][a'] - Q[s][a] \right]
$$

<span style="font-size: 14px;">where $\alpha \in (0,1]$ is the learning rate, $\gamma \in [0,1]$ is the discount factor, and $\pi(a' \mid s')$ is the probability of selecting action $a'$ in state $s'$. The TD error is:</span>

$$
\delta = r + \gamma \sum_{a'} \pi(a' \mid s')\, Q[s'][a'] - Q[s][a]
$$

<span style="font-size: 14px;">The next-state value is the **policy-weighted average** $\sum_{a'} \pi(a'|s') Q[s'][a']$, which is exactly the expected value of $Q[s'][a']$ when $a'$ is drawn from $\pi$. Compare the three TD control targets at the next state: SARSA uses one sampled $Q(s', a')$, Q-learning uses $\max_{a'} Q(s', a')$, and Expected SARSA uses the full expectation $\mathbb{E}_{a' \sim \pi}[Q(s', a')]$.</span>

---

## <span style="font-size: 16px;">Why Take the Expectation</span>

<span style="font-size: 14px;">SARSA's target depends on which next action $a'$ the policy happened to sample. The same $(s, a, r, s')$ can yield very different targets across visits purely because the exploration drew different actions, and this added randomness propagates into the value estimate. Expected SARSA eliminates this source of variance by analytically averaging over all possible $a'$ instead of sampling one.</span>

<span style="font-size: 14px;">The decomposition is clean. The total variance of SARSA's target has two parts: variance from the random reward and next state, and variance from the random next action. Expected SARSA removes the second part entirely, keeping only the irreducible environment variance. Because $\sum_{a'} \pi(a'|s') Q[s'][a'] = \mathbb{E}_{a'}[Q(s',a')]$, the expected SARSA target equals SARSA's target in expectation, so the change is pure variance reduction with no added bias relative to SARSA. Lower target variance means each update moves $Q$ in a more reliable direction, which typically allows a larger learning rate and faster, smoother learning.</span>

---

## <span style="font-size: 16px;">On-Policy or Off-Policy</span>

<span style="font-size: 14px;">Expected SARSA is more flexible than its name suggests. The policy $\pi$ used to form the expectation in the target need not be the same as the behaviour policy that generated the action $a$. This makes the algorithm capable of both on-policy and off-policy learning depending on which $\pi$ is plugged into the expectation:</span>

* <span style="font-size: 14px;">**On-policy**: set $\pi$ to the behaviour policy (for example the same $\epsilon$-greedy policy the agent acts with). The target estimates the value of the policy actually being followed, like SARSA but with lower variance.</span>
* <span style="font-size: 14px;">**Off-policy**: set $\pi$ to a different target policy, such as the greedy policy, while behaving $\epsilon$-greedily. The expectation evaluates the target policy regardless of behaviour, exactly the off-policy idea.</span>

<span style="font-size: 14px;">This dual capability is why Expected SARSA is often described as a generalization that bridges SARSA and Q-learning, rather than a single fixed algorithm.</span>

---

## <span style="font-size: 16px;">Generalizing SARSA and Q-Learning</span>

<span style="font-size: 14px;">Expected SARSA contains both of the previous algorithms as special cases, determined entirely by the choice of $\pi$ in the expectation:</span>

* <span style="font-size: 14px;">**Recovering Q-learning**: if $\pi$ is the **greedy** policy, it puts probability $1$ on $\arg\max_{a'} Q[s'][a']$ and $0$ elsewhere. The sum collapses to $\max_{a'} Q[s'][a']$, which is precisely the Q-learning target. Expected SARSA with a greedy target policy is Q-learning, but with the maximization-bias-amplifying single sample replaced by an exact computation.</span>
* <span style="font-size: 14px;">**Relation to SARSA**: SARSA samples $a' \sim \pi$ and uses $Q[s'][a']$; Expected SARSA averages over that same distribution. SARSA is an unbiased single-sample estimate of the Expected SARSA target.</span>

<span style="font-size: 14px;">This unifying view is the conceptual payoff of the problem: SARSA, Expected SARSA, and Q-learning are not three unrelated algorithms but three points on a single axis defined by how the next-state value is formed, sample, expectation, or maximum.</span>

---

## <span style="font-size: 16px;">Sampling and Bootstrapping</span>

<span style="font-size: 14px;">Expected SARSA remains a model-free, bootstrapped TD method:</span>

* <span style="font-size: 14px;">**Sampling**: $r$ and $s'$ still come from real environment interaction, so no model of transitions is needed.</span>
* <span style="font-size: 14px;">**Bootstrapping**: the return's tail is replaced by the estimate $\gamma \sum_{a'} \pi(a'|s') Q[s'][a']$, supporting one-step online updates.</span>
* <span style="font-size: 14px;">**Expectation over actions**: unlike the dynamics, the policy $\pi$ is known to the agent, so the expectation over $a'$ can be computed exactly rather than sampled. This is the key insight, the action distribution is internal and need not be sampled.</span>

<span style="font-size: 14px;">The cost is a modest increase in per-update computation: forming the expectation requires summing over all actions at $s'$, an $O(|\mathcal{A}|)$ operation, versus a single lookup for SARSA. In tabular and small-action settings this cost is negligible and is almost always worth the variance reduction.</span>

<span style="font-size: 14px;">It is worth stressing why the expectation over actions can be computed exactly while the expectation over next states cannot. The transition dynamics are part of the unknown environment, so the agent must sample $s'$. The policy $\pi$, by contrast, is the agent's own construct: it knows the action probabilities exactly. Expected SARSA exploits precisely this asymmetry, sampling only what it must (the environment) and integrating analytically over what it controls (the action choice).</span>

---

## Worked Example ($\alpha = 0.5$, $\gamma = 0.9$)

<span style="font-size: 14px;">Two states $\{0, 1\}$, two actions $\{0, 1\}$, $Q$ initialized as $\begin{pmatrix} 0 & 0 \\ 2 & 4 \end{pmatrix}$ (rows states, columns actions). Use an $\epsilon$-greedy policy with $\epsilon = 0.2$, so the greedy action gets probability $1 - \epsilon + \epsilon/2 = 0.9$ and the other gets $\epsilon/2 = 0.1$. Process one transition $(s{=}0,\ a{=}0,\ r{=}1,\ s'{=}1)$.</span>

<span style="font-size: 14px;">At $s' = 1$ the values are $Q[1] = [2, 4]$, so the greedy action is $1$. The policy probabilities are $\pi(0|1) = 0.1$ and $\pi(1|1) = 0.9$. The expected next-state value is:</span>

$$
\sum_{a'} \pi(a'|1)\, Q[1][a'] = 0.1 \cdot 2 + 0.9 \cdot 4 = 0.2 + 3.6 = 3.8
$$

<span style="font-size: 14px;">Then $\delta = 1 + 0.9 \cdot 3.8 - Q[0][0] = 1 + 3.42 - 0 = 4.42$, and the update is $Q[0][0] \leftarrow 0 + 0.5 \cdot 4.42 = 2.21$.</span>

<span style="font-size: 14px;">For contrast, SARSA would have used a single sampled $a'$: if it drew $a' = 1$ the target value would be $4$, if it drew $a' = 0$ it would be $2$. Expected SARSA's $3.8$ is the probability-weighted blend of these outcomes, which is why its target does not jump around with the luck of the draw. Q-learning would use $\max = 4$, slightly higher than the expectation because the policy is not fully greedy.</span>

---

## <span style="font-size: 16px;">The Bias-Variance Picture</span>

<span style="font-size: 14px;">Placing Expected SARSA in the bias-variance landscape of this section sharpens the intuition. All three TD control methods share the bias from bootstrapping off an imperfect $Q$, which vanishes as the estimates converge. They differ in variance:</span>

* <span style="font-size: 14px;">**Monte Carlo control** has zero bootstrapping bias but the highest variance, since its target is a full trajectory return.</span>
* <span style="font-size: 14px;">**SARSA** has bootstrapping bias plus variance from both the environment and the sampled next action.</span>
* <span style="font-size: 14px;">**Expected SARSA** has the same bootstrapping bias but removes the action-sampling variance entirely, leaving only environment variance.</span>
* <span style="font-size: 14px;">**Q-learning** has the lowest action-related variance (a deterministic $\max$) but introduces maximization bias from taking the max over noisy estimates.</span>

<span style="font-size: 14px;">Expected SARSA therefore occupies a sweet spot: it keeps SARSA's unbiasedness relative to the policy being evaluated, gains Q-learning-level stability by computing the action term exactly, and avoids Q-learning's maximization bias as long as the target policy is not greedy. The exactness of the action expectation is the lever that buys the variance reduction without paying any bias.</span>

---

## <span style="font-size: 16px;">Online In-Place Updates</span>

<span style="font-size: 14px;">Like the other TD control methods, Expected SARSA processes transitions in order against a live $Q$-table, so each update reads the values written by earlier ones. The expectation $\sum_{a'} \pi(a'|s') Q[s'][a']$ must use the current row $Q[s']$, including any updates an earlier transition applied to it. A batched variant against a frozen table would generally give different numbers; the standard formulation is online and in-place.</span>

<span style="font-size: 14px;">When the policy $\pi$ is itself derived from $Q$ (such as $\epsilon$-greedy on the current values), the action probabilities can shift as $Q$ changes during a sequence. The expectation should be computed against the policy implied by the current $Q$ at the moment of each update, keeping the evaluation consistent with the live table.</span>

---

## <span style="font-size: 16px;">Computing the Policy Expectation</span>

<span style="font-size: 14px;">The heart of the algorithm is forming $\sum_{a'} \pi(a'|s') Q[s'][a']$ correctly, so it is worth spelling out for the common $\epsilon$-greedy case. With $|\mathcal{A}|$ actions and exploration rate $\epsilon$, the greedy action $a^* = \arg\max_{a'} Q[s'][a']$ receives probability $1 - \epsilon + \epsilon/|\mathcal{A}|$ and every other action receives $\epsilon/|\mathcal{A}|$. The expectation can then be written as a blend of the greedy value and the uniform average:</span>

$$
\sum_{a'} \pi(a'|s')\, Q[s'][a'] = (1 - \epsilon) \max_{a'} Q[s'][a'] + \frac{\epsilon}{|\mathcal{A}|} \sum_{a'} Q[s'][a']
$$

<span style="font-size: 14px;">This form makes the SARSA-to-Q-learning interpolation explicit: as $\epsilon \to 0$ the first term dominates and the target approaches $\max_{a'} Q[s'][a']$, recovering Q-learning; as $\epsilon \to 1$ it approaches the uniform average over actions. For a general policy the agent simply iterates over all next-state actions, multiplies each $Q[s'][a']$ by its probability, and sums. Ties in the greedy action should be handled consistently with whatever tie-breaking the policy definition uses, since they affect the probability mass assigned to the maximizing actions.</span>

---

## <span style="font-size: 16px;">Convergence and Practical Notes</span>

<span style="font-size: 14px;">Expected SARSA converges under the same conditions as SARSA and Q-learning: the Robbins-Monro step sizes $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$, and sufficient exploration so every state-action pair is visited infinitely often. Because the target has lower variance, Expected SARSA can in practice use a larger constant step size, even $\alpha = 1$ in deterministic environments, without the instability that would cause in SARSA, since there is no action-sampling noise to amplify.</span>

<span style="font-size: 14px;">Empirically Expected SARSA matches or beats both SARSA and Q-learning across a range of tasks (van Seijen et al., 2009), trading a small amount of extra computation per step for noticeably more stable learning curves. It is the default choice when the action space is small enough that the expectation is cheap to compute.</span>

<span style="font-size: 14px;">The stability advantage is most pronounced in **stochastic** environments. When rewards and transitions are noisy, SARSA's additional action-sampling variance compounds with the environment noise and can force a small, conservative learning rate to keep the estimates from oscillating. Expected SARSA, having stripped out the action variance, tolerates more aggressive step sizes and converges faster on the same data. In deterministic environments the gap narrows but Expected SARSA never does worse, since the removed variance term is non-negative.</span>

<span style="font-size: 14px;">The practical limitation appears only when the action space is large or continuous, where summing over all $a'$ becomes expensive or impossible. In those regimes the single-sample SARSA target or specialized continuous-action methods are preferred, and the exact expectation is approximated or abandoned. For the tabular, small-action setting of this problem, the exact computation is both cheap and clearly worthwhile.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to weight by the policy probabilities.** The target is $\sum_{a'} \pi(a'|s') Q[s'][a']$, a weighted average, not a plain average over actions and not the value of a single action. Dropping the $\pi(a'|s')$ weights or using uniform weights gives the wrong expectation and a different algorithm.</span>
* <span style="font-size: 14px;">**Using a stale or inconsistent policy.** When $\pi$ is derived from $Q$, the action probabilities must reflect the current $Q$ at the time of the update. Computing the expectation against an outdated policy, or against a different policy than intended, silently changes whether the update is on-policy or off-policy.</span>
* <span style="font-size: 14px;">**Mishandling the terminal next state.** At an episode's final transition the entire expected-value term must be zero so the target reduces to $r$. Summing the policy over a terminal state's stale $Q$ row leaks phantom future value into the estimate.</span>
* <span style="font-size: 14px;">**Assuming it is strictly on-policy.** Expected SARSA can be on-policy or off-policy depending on the $\pi$ used in the expectation. Treating it as inherently on-policy, and ignoring that a greedy $\pi$ turns it into Q-learning, leads to incorrect reasoning about what it converges to.</span>

---