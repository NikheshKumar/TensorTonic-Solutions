# <span style="font-size: 20px;">Prioritized Experience Replay</span>

<span style="font-size: 14px;">Prioritized Experience Replay (Schaul et al., 2016, "Prioritized Experience Replay") replaces the uniform sampling of a standard replay buffer with sampling **proportional to TD-error magnitude**, so the agent revisits surprising, poorly-fit transitions more often than ones it already predicts well. Because this biases the data distribution, each sampled transition is reweighted by an importance-sampling correction. The result is a strictly more sample-efficient learner that the paper shows lifts DQN performance across most of the Atari suite.</span>

---

## <span style="font-size: 16px;">The Problem with Uniform Replay</span>

<span style="font-size: 14px;">A uniform buffer treats every stored transition as equally worth learning from. But not all experience is equally informative. A transition whose value the network already predicts almost perfectly carries a tiny gradient, while a transition with a large **TD error** is one the agent is currently getting wrong and therefore has the most to learn from. Sampling them at the same rate wastes most updates on transitions that barely move the parameters.</span>

<span style="font-size: 14px;">The pathological case the paper highlights is rare but pivotal experience: in a sparse-reward task, the single transition that first encounters reward can sit in a million-entry buffer and be sampled, on average, once per million draws under the uniform scheme. Prioritization lets that transition be replayed promptly and repeatedly while its error is large, propagating the reward signal backward through the value function far faster.</span>

<span style="font-size: 14px;">The authors motivate this with a toy environment called "Blind Cliffwalk", constructed so that reward is reachable only through one exact sequence of actions. They show that a uniform learner needs a number of updates that grows exponentially in the state count to discover the reward, whereas an oracle that always replays the maximally useful transition needs a number that grows only polynomially. Prioritized replay is the practical approximation to that oracle, recovering most of the gap without knowing the ideal transition in advance.</span>

---

## <span style="font-size: 16px;">Priorities from TD Error</span>

<span style="font-size: 14px;">The priority of transition $i$ is a power of its absolute TD error plus a small constant:</span>

$$
p_i = (|\delta_i| + \varepsilon)^{\alpha}, \qquad P(i) = \frac{p_i}{\sum_j p_j}
$$

<span style="font-size: 14px;">where $\delta_i = r_i + \gamma \max_{a'} Q_{\theta^-}(s_i', a') - Q_\theta(s_i, a_i)$ is the same TD error the DQN loss minimizes. The terms play distinct roles:</span>

* <span style="font-size: 14px;">$|\delta_i|$ measures how surprising the transition is. Larger error means the prediction and the bootstrapped target disagree more, so revisiting it yields a larger learning signal.</span>
* <span style="font-size: 14px;">$\alpha \in [0, 1]$ is the **prioritization exponent**. It interpolates between two extremes: $\alpha = 0$ makes every $p_i = 1$ and recovers ordinary uniform replay, while $\alpha = 1$ makes the sampling probability directly proportional to the error. Intermediate values (the paper uses $\alpha \approx 0.6$ for the proportional variant) prioritize without being fully greedy.</span>
* <span style="font-size: 14px;">$\varepsilon > 0$ is a small floor that keeps a transition with zero TD error from receiving probability zero and being **permanently** excluded. Without it, any transition the network momentarily fits exactly would never be revisited, even if its true error later grows.</span>

<span style="font-size: 14px;">Normalizing by $\sum_j p_j$ turns the raw priorities into a proper probability distribution $P$ over the buffer. This is the **proportional** variant; the paper also describes a rank-based variant where $p_i = 1 / \text{rank}(i)$ with the rank taken over sorted $|\delta_i|$, which is more robust to outliers and heavy-tailed errors but is not used here. Both variants share the same $\alpha$ tempering and the same IS correction below; they differ only in how the raw priority is derived from the error.</span>

---

## <span style="font-size: 16px;">What TD Error Measures as a Priority</span>

<span style="font-size: 14px;">Using $|\delta_i|$ as the priority is a deliberate proxy for "how much can the agent learn from this transition right now". The paper frames the ideal criterion as the expected magnitude of the update a transition would induce, and the TD error is the most readily available, cheap surrogate for it: the squared-error gradient is exactly the TD error times the prediction gradient, so a large $|\delta_i|$ directly implies a large parameter update.</span>

<span style="font-size: 14px;">There are two important caveats the authors stress. First, the TD error is **stochastic**: it reflects not only how wrong the prediction is but also the randomness of the sampled reward and next state, so a high error can be noise rather than genuine learning signal. Second, it is **only known for transitions that have been evaluated**, which is why fresh transitions inherit the current maximum priority. The exponent $\alpha$ exists precisely to soften the proxy, so the agent leans toward high-error transitions without trusting the raw error completely.</span>

---

## <span style="font-size: 16px;">Why Greedy Prioritization Alone Fails</span>

<span style="font-size: 14px;">A purely greedy scheme that always replays the highest-error transitions sounds appealing but breaks in three ways the paper identifies, which is exactly why the stochastic, $\alpha$-tempered form exists:</span>

* <span style="font-size: 14px;">**Stale priorities.** TD errors are only refreshed for transitions that are actually sampled. A transition that starts with a low error is never resampled, so its error is never updated, and it stays starved forever even if the value function drifts.</span>
* <span style="font-size: 14px;">**Sensitivity to noise.** Reward or value noise produces spuriously large errors. Greedy replay would lock onto these noisy spikes and overfit them.</span>
* <span style="font-size: 14px;">**Loss of diversity.** Focusing only on the current top errors collapses the effective training set to a small high-error subset, which causes overfitting and forgetting. Stochastic prioritization keeps every transition's probability strictly positive, guaranteeing the whole buffer remains reachable.</span>

---

## <span style="font-size: 16px;">Importance-Sampling Correction</span>

<span style="font-size: 14px;">Changing the sampling distribution changes the expectation that stochastic gradient descent estimates. Under uniform sampling, the expected update is the mean gradient over the buffer; under $P$, transitions appear at the wrong frequencies, so the gradient estimate is **biased** toward high-priority regions. Left uncorrected, this bias changes the solution the network converges to, not just its path there. The fix is the standard importance-sampling identity, reweighting each sample by the ratio of the target distribution to the sampling distribution:</span>

$$
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^{\beta} = (N \cdot P(i))^{-\beta}, \qquad \tilde w_i = \frac{w_i}{\max_j w_j}
$$

At beta equal to one, the raw weight is the uniform probability divided by the sampling probability. For fixed per-transition gradient vectors, raw weighting gives

$$
\sum_{i=1}^{N}P(i)w_i g_i=\frac{1}{N}\sum_{i=1}^{N}g_i.
$$

After normalization by the maximum raw weight across the buffer,

$$
\sum_{i=1}^{N}P(i)\widetilde w_i g_i=\frac{1}{\max_j w_j}\frac{1}{N}\sum_{i=1}^{N}g_i.
$$

Here, N is the buffer size, i and j index transitions, P is the sampling probability, w is the raw weight, w tilde is its normalized value, and g is a per-transition gradient. Probabilities and weights are held fixed for this expectation. Max normalization preserves the expected direction but rescales its magnitude; exact equality to the uniform-replay gradient applies to the raw weights.

* <span style="font-size: 14px;">$\beta \in [0, 1]$ controls how fully the bias is corrected: $\beta = 0$ applies no correction (all weights become 1), and $\beta = 1$ restores the uniform expectation with raw weights; max-normalized weights retain a shared scale factor. The paper anneals $\beta$ from an initial value (around $0.4$) toward $1$ over training, because the bias matters most near convergence, when the network is settling on its final solution, and least early on, when updates are large and noisy anyway.</span>
* <span style="font-size: 14px;">Dividing by $\max_j w_j$ normalizes the weights so the largest is exactly $1$. This only ever **scales the weights down**, which keeps the effective learning rate from blowing up. Without it, a rare low-probability transition could receive a huge weight and produce a destabilizing gradient spike.</span>

---

## Worked Example ($N = 3$, $\alpha = 0.5$, $\beta = 1$, $\varepsilon = 0$)

<span style="font-size: 14px;">Let the absolute TD errors be $|\delta| = [4, 1, 0.25]$.</span>

<span style="font-size: 14px;">1. **Priorities:** $p_i = |\delta_i|^{0.5} = [2, 1, 0.5]$.</span>

<span style="font-size: 14px;">2. **Normalize:** $\sum p = 3.5$, so $P = [2/3.5, 1/3.5, 0.5/3.5] = [0.5714, 0.2857, 0.1429]$.</span>

<span style="font-size: 14px;">3. **Raw IS weights** with $N = 3$, $\beta = 1$: $w_i = (3 \cdot P(i))^{-1} = [1/1.7143, 1/0.8571, 1/0.4286] = [0.5833, 1.1667, 2.3333]$.</span>

<span style="font-size: 14px;">4. **Normalize by the max** ($2.3333$): $\tilde w = [0.25, 0.5, 1.0]$.</span>

The highest-priority transition receives the smallest normalized weight. Each probability times its normalized weight equals 1/7, compared with 1/3 under uniform replay. The expected weighted gradient is therefore 3/7 of the uniform-replay gradient, with the same direction and a smaller magnitude.

---

## <span style="font-size: 16px;">Implementation: the Sum-Tree</span>

<span style="font-size: 14px;">Naively recomputing $\sum_j p_j$ and searching for a sampled cumulative-probability point would cost $O(N)$ per draw, far too slow for a million-entry buffer sampled thousands of times per second. PER uses a **sum-tree**: a binary tree where each leaf holds a transition's priority and each internal node holds the sum of its children. The root holds the total priority. Sampling draws a uniform value in $[0, \text{total})$ and walks from the root to a leaf in $O(\log N)$, and updating a single priority after its TD error is refreshed is also $O(\log N)$. This data structure is what makes prioritization practical at DQN scale.</span>

<span style="font-size: 14px;">In practice the batch is sampled by **stratified sampling**: the total priority range $[0, \text{total})$ is split into $B$ equal segments, and one uniform draw is taken from each segment before walking the tree. This guarantees the batch spans the whole priority distribution rather than clustering on a few very high-priority leaves, which further protects diversity. Only the priorities of the actually sampled leaves are refreshed each step, so the per-update cost stays $O(B \log N)$ regardless of buffer size, the same order as ordinary replay up to the logarithmic factor.</span>

<span style="font-size: 14px;">When a transition is sampled and its loss computed, its priority is immediately updated with the fresh $|\delta_i|$, and newly inserted transitions are given maximum priority so they are guaranteed to be replayed at least once before their error is ever measured.</span>

---

## <span style="font-size: 16px;">The Interaction Between $\alpha$ and $\beta$</span>

<span style="font-size: 14px;">The two exponents are coupled and should be reasoned about together. $\alpha$ sets how skewed the sampling distribution is, and $\beta$ sets how much of that skew gets corrected back out in the gradient. At one extreme, $\alpha = 0$ with any $\beta$ is plain uniform replay (nothing to correct). At the other, large $\alpha$ with $\beta = 0$ aggressively prioritizes and never corrects, which learns fast early but converges to a biased solution.</span>

<span style="font-size: 14px;">The recipe that works in practice, and the one the paper recommends, is moderate $\alpha$ (around $0.6$ proportional, $0.7$ rank-based) with $\beta$ annealed from about $0.4$ up to $1$. The intuition is that bias is harmless early in training when the value function is changing rapidly and approximate anyway, but as learning converges the small remaining gradients must be unbiased to land on the correct fixed point. Annealing $\beta \to 1$ delivers the best of both: fast, focused early learning and correction toward the uniform-replay direction, with max normalization retaining a shared scale factor. The normalization $\tilde w_i = w_i / \max_j w_j$ ties the two together numerically, ensuring that whatever $\alpha$ and $\beta$ are chosen, the largest correction weight is held at $1$ so the effective step size stays controlled.</span>

---

## <span style="font-size: 16px;">Paper Results and Modern Context</span>

<span style="font-size: 14px;">Schaul et al. report that PER on top of Double DQN improves the median human-normalized score across the 57-game Atari benchmark and reaches a given performance level in substantially fewer environment steps, with the proportional variant the strongest overall. PER is one of the six components combined in Rainbow (Hessel et al., 2018), where ablations confirm it is among the most important contributors, second only to multi-step learning and distributional value estimation. The combination of priority $p^\alpha$ and annealed IS weights $\beta \to 1$ remains the standard recipe in modern off-policy agents.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Omitting $\varepsilon$.** If $\varepsilon = 0$, any transition whose TD error happens to hit exactly zero gets priority zero and probability zero, so it can never be sampled again to have its error re-evaluated. A small positive floor keeps every transition reachable.</span>
* <span style="font-size: 14px;">**Forgetting to apply the IS weights to the loss.** Computing $\tilde w_i$ but not multiplying it into the per-sample gradient leaves the bias fully intact. The network then converges to the wrong fixed point, optimizing a reweighted objective rather than the intended one, a silent error that only shows up as degraded final performance.</span>
* <span style="font-size: 14px;">**Not refreshing priorities after an update.** Priorities must be recomputed from the new TD errors of sampled transitions. If they are never updated, the distribution freezes at the initial errors and prioritization becomes meaningless, often worse than uniform.</span>
* <span style="font-size: 14px;">**Skipping the max-normalization of weights.** Dividing by $\max_j w_j$ caps the largest weight at $1$. Without it, low-probability transitions yield very large weights that act as effective learning-rate spikes, destabilizing training, the opposite of what the correction is meant to achieve.</span>

---