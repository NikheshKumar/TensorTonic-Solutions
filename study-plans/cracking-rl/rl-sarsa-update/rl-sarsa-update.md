# <span style="font-size: 20px;">SARSA Update</span>

<span style="font-size: 14px;">SARSA is the canonical **on-policy** temporal-difference control algorithm. It learns an action-value function $Q(s,a)$, the expected return of taking action $a$ in state $s$ and then continuing under the current policy. Its name comes from the five-tuple of experience it consumes per update: **S**tate, **A**ction, **R**eward, next **S**tate, next **A**ction.</span>

---

## <span style="font-size: 16px;">The Update Rule</span>

<span style="font-size: 14px;">Given a transition $(s, a, r, s', a')$ collected by the agent's policy, SARSA updates:</span>

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \, \bigl[ r + \gamma \, Q(s', a') - Q(s, a) \bigr]
$$

<span style="font-size: 14px;">where $\alpha \in (0,1]$ is the learning rate and $\gamma \in [0,1]$ is the discount factor. The bracketed term is the **TD error** for action values:</span>

$$
\delta = r + \gamma \, Q(s', a') - Q(s, a)
$$

<span style="font-size: 14px;">The structure is identical to TD(0) for state values, but the quantity being estimated is now indexed by an action, and the bootstrap target uses the value of the **specific action $a'$ that was actually taken** at the next state. This single design choice, using the taken action rather than a hypothetical best one, is what makes SARSA on-policy.</span>

---

## <span style="font-size: 16px;">From Prediction to Control</span>

<span style="font-size: 14px;">TD(0) evaluates a fixed policy by learning $V^\pi$. SARSA goes further: it does **control**, meaning it improves the policy while evaluating it. This works through **generalized policy iteration**. The agent maintains $Q$, derives a policy from it, usually $\epsilon$-greedy (greedy with respect to $Q$ most of the time, random with small probability $\epsilon$), and uses that policy to both act and generate the next action $a'$ in each update.</span>

<span style="font-size: 14px;">Because $Q$ improves with every update and the policy is always defined relative to the current $Q$, evaluation and improvement happen simultaneously and continuously. Over time both the value estimates and the policy converge together. Learning $Q$ rather than $V$ is essential here: choosing actions greedily requires knowing the value of each action, which $V(s)$ alone does not provide without a model.</span>

---

## <span style="font-size: 16px;">On-Policy: Why $a'$ Matters</span>

<span style="font-size: 14px;">SARSA's target bootstraps off $Q(s', a')$ where $a'$ is the action the behaviour policy genuinely selected at $s'$. The algorithm therefore evaluates the policy it is **actually following**, including that policy's exploratory moves. If the $\epsilon$-greedy policy occasionally takes a bad random action, SARSA's value estimates account for the cost of that exploration.</span>

<span style="font-size: 14px;">This is the defining property of on-policy learning: the policy that generates the data and the policy being evaluated and improved are one and the same. The practical consequence is that SARSA learns a policy that is **safe under its own exploration**. In environments where exploratory mistakes are costly, the cliff-walking task is the textbook example, SARSA learns a conservative path that stays away from the danger because it knows random actions might push it over the edge. Q-learning, which is off-policy, learns the riskier optimal path and suffers more during training.</span>

---

## <span style="font-size: 16px;">Sampling and Bootstrapping</span>

<span style="font-size: 14px;">Like all TD methods, SARSA combines two ingredients:</span>

* <span style="font-size: 14px;">**Sampling**: the reward $r$ and next state $s'$ come from real interaction with the environment, so no model is needed. SARSA is fully model-free.</span>
* <span style="font-size: 14px;">**Bootstrapping**: the tail of the return is replaced by the current estimate $\gamma Q(s', a')$ rather than the observed full return. SARSA updates after a single step, online and incrementally.</span>

<span style="font-size: 14px;">The bootstrapping makes SARSA's target lower variance than a Monte Carlo control return, at the cost of bias from relying on an imperfect $Q$. As $Q$ improves the bias shrinks. This is the same bias-variance trade-off seen throughout TD learning: one real reward plus a bootstrapped estimate is far less noisy than a full trajectory return, which is why SARSA typically learns faster than MC control.</span>

<span style="font-size: 14px;">There is, however, an extra source of variance specific to SARSA that the next algorithm in this section targets. Because the target bootstraps off a single sampled next action $a'$, the randomness of the exploration policy injects noise into every update: the same $(s, a, r, s')$ can yield very different targets depending on which $a'$ the policy happened to draw. Expected SARSA removes exactly this term by averaging over all possible $a'$ under $\pi$, which is the motivation for studying it immediately after SARSA.</span>

---

## Worked Example ($\alpha = 0.5$, $\gamma = 0.9$)

<span style="font-size: 14px;">Take two states $\{0, 1\}$ and two actions $\{0, 1\}$, with the $Q$-table initialized to all zeros. Process two transitions $(s, a, r, s', a')$ in order:</span>

* <span style="font-size: 14px;">Transition 1: $(s{=}0,\ a{=}1,\ r{=}1,\ s'{=}1,\ a'{=}0)$</span>
* <span style="font-size: 14px;">Transition 2: $(s{=}1,\ a{=}0,\ r{=}2,\ s'{=}0,\ a'{=}1)$</span>

<span style="font-size: 14px;">**Transition 1**: $\delta = 1 + 0.9 \cdot Q(1,0) - Q(0,1) = 1 + 0.9 \cdot 0 - 0 = 1$. Update $Q(0,1) \leftarrow 0 + 0.5 \cdot 1 = 0.5$.</span>

<span style="font-size: 14px;">**Transition 2**: $\delta = 2 + 0.9 \cdot Q(0,1) - Q(1,0)$. Crucially, $Q(0,1)$ was just written to $0.5$, so $\delta = 2 + 0.9 \cdot 0.5 - 0 = 2.45$. Update $Q(1,0) \leftarrow 0 + 0.5 \cdot 2.45 = 1.225$.</span>

<span style="font-size: 14px;">The final $Q$-table is $\begin{pmatrix} 0 & 0.5 \\ 1.225 & 0 \end{pmatrix}$, where rows index states and columns index actions. The example demonstrates the in-place behaviour: the second update's bootstrap target reads the value the first update produced, so transitions must be processed in order against the live table.</span>

---

## <span style="font-size: 16px;">Online In-Place Updates</span>

<span style="font-size: 14px;">SARSA processes transitions one at a time, reading and writing the same $Q$-table. Each update sees the effect of all earlier updates in the sequence. This matters whenever a later transition bootstraps off an entry an earlier transition changed, exactly as in the worked example. A batch variant that froze $Q$ and applied all updates against the old table would give different numbers; the standard online formulation uses live, in-place writes.</span>

<span style="font-size: 14px;">In a full training loop, the next action $a'$ used in the update is the same action the agent will execute on the following step. This tight coupling, choose $a'$, use it in the update, then act on it, is what guarantees the algorithm stays on-policy step after step.</span>

---

## <span style="font-size: 16px;">The Action-Value Bellman Equation</span>

<span style="font-size: 14px;">SARSA is a sample-based solver for the Bellman expectation equation written in terms of action values:</span>

$$
Q^\pi(s, a) = \mathbb{E}_\pi\!\left[ r + \gamma \, Q^\pi(s', a') \mid s, a \right]
$$

<span style="font-size: 14px;">where the expectation is over the reward, the next state, and the next action $a'$ drawn from $\pi$. SARSA cannot evaluate this expectation directly because it lacks a model, so it draws a single sample of the whole right-hand side, the observed $r$, the observed $s'$, and the actually-chosen $a'$, and steps $Q(s,a)$ toward it. The learning rate averages out the sampling noise over repeated visits.</span>

<span style="font-size: 14px;">This is exactly why the next action $a'$ enters the target: the Bellman equation for $Q^\pi$ takes its expectation over $a' \sim \pi$, so an unbiased single-sample estimate must use an action drawn from $\pi$. Sampling $a'$ from the same policy is what keeps SARSA's target an unbiased estimate of the on-policy backup, and it is the formal reason SARSA evaluates the behaviour policy rather than some other one.</span>

---

## <span style="font-size: 16px;">Why On-Policy Learning Is Safer</span>

<span style="font-size: 14px;">Because SARSA's value of a state-action pair includes the consequences of the policy's own exploration, the values it learns are realistic estimates of what the agent will actually experience while still exploring. A state next to a catastrophic outcome will have a depressed value under SARSA, because there is a real chance an $\epsilon$-greedy slip sends the agent into the catastrophe.</span>

<span style="font-size: 14px;">The cliff-walking gridworld makes this concrete. The optimal path hugs the edge of a cliff; one wrong step falls off and incurs a large penalty. Q-learning learns the value of the edge path, assuming greedy behaviour, and so prefers it, yet during training its $\epsilon$-greedy exploration repeatedly falls off. SARSA, evaluating its exploratory policy, assigns the edge path a lower value and learns a safer route one row away from the cliff, achieving higher reward during learning. Neither is universally better: SARSA optimizes online performance under exploration, Q-learning targets the asymptotically optimal greedy policy.</span>

---

## <span style="font-size: 16px;">Convergence</span>

<span style="font-size: 14px;">Tabular SARSA converges to the optimal action-value function $Q^*$ and an optimal policy under two conditions. First, the usual Robbins-Monro step-size requirements:</span>

$$
\sum_{t} \alpha_t = \infty, \qquad \sum_{t} \alpha_t^2 < \infty
$$

<span style="font-size: 14px;">Second, the policy must be **greedy in the limit with infinite exploration** (GLIE): every state-action pair is visited infinitely often, and the exploration rate $\epsilon$ decays to zero so that the policy becomes greedy asymptotically. The GLIE condition is what reconciles SARSA's on-policy nature with optimality: it must keep exploring to learn, yet the exploration must vanish so the policy it converges to is the optimal greedy one rather than a permanently $\epsilon$-soft compromise.</span>

<span style="font-size: 14px;">If $\epsilon$ is held fixed and never decays, SARSA converges to the best $\epsilon$-soft policy, not the truly optimal one, because it always evaluates a policy that retains some random actions.</span>

---

## <span style="font-size: 16px;">n-step SARSA and SARSA($\lambda$)</span>

<span style="font-size: 14px;">One-step SARSA is the shallowest member of a family. An $n$-step SARSA target accumulates $n$ real rewards before bootstrapping:</span>

$$
r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n Q(s_{t+n}, a_{t+n})
$$

<span style="font-size: 14px;">Larger $n$ uses more sampled reward and less bootstrapping, raising variance but lowering bias and often speeding credit assignment over long horizons. SARSA($\lambda$) averages all $n$-step targets with geometric weights $\lambda$, implemented efficiently with eligibility traces, and recovers one-step SARSA at $\lambda = 0$ and a Monte Carlo control update at $\lambda = 1$. The single-step version in this problem is the foundation on which these extensions are built; understanding exactly how $a'$ enters the one-step target is what makes the trace-based variants comprehensible.</span>

---

## <span style="font-size: 16px;">SARSA vs Q-Learning</span>

<span style="font-size: 14px;">SARSA and Q-learning differ only in how the next-state value in the target is formed:</span>

* <span style="font-size: 14px;">**SARSA (on-policy)**: target uses $Q(s', a')$ with the actually-taken $a'$. It evaluates and improves the behaviour policy, exploration included.</span>
* <span style="font-size: 14px;">**Q-learning (off-policy)**: target uses $\max_{a'} Q(s', a')$, the value of the greedy action regardless of what was taken. It learns the optimal policy while following an exploratory one.</span>

<span style="font-size: 14px;">When the policy is already greedy (no exploration), the two coincide, since the taken action is the greedy one. They diverge precisely because of exploratory actions. Expected SARSA, the next problem, sits between them by replacing the single sampled $a'$ with the expectation $\sum_{a'} \pi(a'|s') Q(s',a')$ over the policy, reducing variance and recovering Q-learning when $\pi$ is greedy.</span>

---

## <span style="font-size: 16px;">Exploration and the $\epsilon$-Greedy Policy</span>

<span style="font-size: 14px;">Control demands a balance between **exploitation**, taking the action currently believed best, and **exploration**, trying alternatives to discover whether something better exists. SARSA relies on the policy derived from $Q$ to strike this balance, most commonly with $\epsilon$-greedy action selection:</span>

$$
\pi(a \mid s) = \begin{cases} 1 - \epsilon + \epsilon/|\mathcal{A}| & a = \arg\max_{a'} Q(s, a') \\ \epsilon/|\mathcal{A}| & \text{otherwise} \end{cases}
$$

<span style="font-size: 14px;">With probability $1 - \epsilon$ the agent acts greedily; with probability $\epsilon$ it picks uniformly at random. Because SARSA is on-policy, these random choices are folded into the next action $a'$ used in the update, so the value estimates literally reflect an $\epsilon$-greedy agent. A practical schedule decays $\epsilon$ from a high initial value toward zero, exploring aggressively early when $Q$ is unreliable and exploiting more as the estimates sharpen. The decay rate trades off learning speed against the risk of converging prematurely to a suboptimal policy before the state space has been adequately covered.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the greedy action instead of the taken $a'$.** Bootstrapping off $\max_{a'} Q(s', a')$ turns SARSA into Q-learning. SARSA must use the action $a'$ that was actually selected, otherwise it stops being on-policy and the learned values no longer reflect the behaviour policy.</span>
* <span style="font-size: 14px;">**Reading stale $Q$ values.** Updates are in place: the target must read the current $Q(s', a')$, including any change an earlier transition in the sequence wrote. Snapshotting the table at the start gives wrong results whenever entries are revisited.</span>
* <span style="font-size: 14px;">**Mishandling the terminal next state.** On the last transition of an episode there is no next action. The bootstrap term must be set to zero so the target is just $r$; bootstrapping off a nonexistent $a'$ or a nonzero terminal value corrupts the estimate.</span>
* <span style="font-size: 14px;">**Never decaying exploration.** With a fixed $\epsilon$, SARSA converges to the optimal $\epsilon$-soft policy, not the true optimum. Expecting it to recover $Q^*$ without annealing $\epsilon$ toward zero is a common misconception.</span>

---