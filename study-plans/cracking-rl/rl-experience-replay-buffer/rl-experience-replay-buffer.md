# <span style="font-size: 20px;">Experience Replay Buffer</span>

<span style="font-size: 14px;">The experience replay buffer is the memory that made deep Q-learning stable (Mnih et al., 2015; the idea traces to Lin, 1992). It stores past transitions $(s, a, r, s', d)$ in a fixed-size circular store and feeds the learner uniform mini-batches sampled from that store, decorrelating consecutive experiences and recycling every interaction many times.</span>

---

## <span style="font-size: 16px;">Why an Agent Cannot Learn Online from a Stream</span>

<span style="font-size: 14px;">An agent interacting with an environment produces a stream of transitions that are **highly correlated** in time: consecutive frames in a game differ by a few pixels, and consecutive states lie along a single trajectory. Training a neural network on this stream in arrival order violates the i.i.d. assumption that stochastic gradient descent relies on. Two concrete failures follow:</span>

* <span style="font-size: 14px;">**Correlated gradients.** A mini-batch drawn from a short window of time looks like a tight cluster of near-identical points. The gradient it produces is biased toward whatever the agent happens to be doing right now, causing the network to overfit the current situation and forget earlier ones.</span>
* <span style="font-size: 14px;">**Wasted data and instability.** Each environment step is expensive, yet online learning uses each transition exactly once and then discards it. Worse, because the policy generating the data is changing, the input distribution is non-stationary, which can push value estimates into oscillation or divergence.</span>

<span style="font-size: 14px;">The replay buffer addresses both: it stores many transitions and samples them uniformly, so a mini-batch mixes experiences from many different times and policies, approximating i.i.d. sampling, and each transition can be reused across many gradient steps.</span>

---

## <span style="font-size: 16px;">The Circular Buffer Mechanism</span>

<span style="font-size: 14px;">The store is a fixed-capacity array with a write pointer, the **head**. Insertion writes the new transition at the head and advances it modulo the capacity:</span>

$$
\texttt{buf}[\texttt{head}] \leftarrow (s, a, r, s', d), \quad \texttt{head} \leftarrow (\texttt{head} + 1) \bmod \texttt{capacity}
$$

<span style="font-size: 14px;">While the buffer is not yet full, the head simply marches forward and the logical size grows. Once $\texttt{capacity}$ transitions have been written, the head wraps to index $0$ and begins **overwriting the oldest slot**. This gives a sliding window over the most recent $\texttt{capacity}$ transitions with $O(1)$ insertion and constant memory, never reallocating or shifting elements. In the original Atari setup the capacity is one million transitions.</span>

<span style="font-size: 14px;">Sampling is explicit and index-based. Given a list of indices $(i_0, i_1, \ldots, i_{B-1})$, each chosen uniformly at random from the valid range, the buffer reads the corresponding rows and **transposes** them into five parallel lists, one per field:</span>

$$
\texttt{batch}[j][k] = \texttt{buf}[i_k][j], \quad j \in \{s, a, r, s', d\}
$$

<span style="font-size: 14px;">The output is a tuple $(\texttt{states}, \texttt{actions}, \texttt{rewards}, \texttt{next\_states}, \texttt{dones})$, each of length $B$, in the same order as the requested indices. This columnar layout is exactly what a vectorized learner consumes: the whole batch of states goes through the network in one forward pass.</span>

<span style="font-size: 14px;">The transpose from row-major storage to column-major batch is the operation most worth getting right. Each slot is a row $(s, a, r, s', d)$; the learner instead wants all states together, all rewards together, and so on. Concretely, the field index $j$ stays fixed while the transition index $k$ varies, producing one list per field. In array terms this is gathering rows at $\texttt{sample\_indices}$ and then unpacking the five fields into parallel arrays, with each array preserving the order of $\texttt{sample\_indices}$ so that position $k$ in every output list refers to the same original transition.</span>

---

## <span style="font-size: 16px;">The Mixing Effect Made Precise</span>

<span style="font-size: 14px;">It is worth being precise about what uniform sampling buys. Suppose the agent generates transitions in episodes, and within an episode successive states are nearly identical. A mini-batch drawn from the live stream is essentially a sample of a single point repeated, so its empirical gradient has tiny variance but large **bias** relative to the gradient over the full state distribution. The network repeatedly takes confident steps in the direction of the current local region, then has to undo them when the agent moves elsewhere, the cause of the oscillation seen in pre-replay neural value methods.</span>

<span style="font-size: 14px;">Uniform sampling from a buffer that spans many episodes draws each transition with probability close to $1 / N$ where $N$ is the number of stored transitions, so the expected mini-batch gradient approximates the gradient averaged over the buffer's full state-visitation distribution. Trading a little extra variance for a large reduction in bias is exactly the right move for stable stochastic optimization, and it is the formal reason replay works. The buffer also smooths the input distribution over time: as the policy improves, fresh transitions slowly replace old ones, so the training distribution shifts gradually rather than abruptly.</span>

---

## <span style="font-size: 16px;">What a Transition Stores</span>

<span style="font-size: 14px;">Each slot holds the five quantities a temporal-difference update needs:</span>

* <span style="font-size: 14px;">$s$ - the state (for Atari, a stack of recent frames) in which the action was taken.</span>
* <span style="font-size: 14px;">$a$ - the discrete action executed.</span>
* <span style="font-size: 14px;">$r$ - the immediate reward received.</span>
* <span style="font-size: 14px;">$s'$ - the resulting next state, used to bootstrap the target $\max_{a'} Q(s', a')$.</span>
* <span style="font-size: 14px;">$d$ - the terminal flag indicating whether $s'$ ended the episode. It controls whether the next-state value is included in the target.</span>

<span style="font-size: 14px;">Storing $s'$ explicitly (rather than reconstructing it from the next transition) keeps each entry self-contained, so any subset of transitions can be sampled and used independently regardless of their order in the buffer.</span>

---

## <span style="font-size: 16px;">How the Buffer Fits the DQN Training Loop</span>

<span style="font-size: 14px;">The buffer sits between acting and learning, and the two run at decoupled rates. A typical DQN step looks like this:</span>

<span style="font-size: 14px;">1. **Act:** the agent selects an action with an epsilon-greedy policy over $Q_\theta(s, \cdot)$, executes it, and observes $(r, s', d)$.</span>

<span style="font-size: 14px;">2. **Store:** the transition $(s, a, r, s', d)$ is pushed into the circular buffer, possibly overwriting the oldest entry.</span>

<span style="font-size: 14px;">3. **Sample:** once the buffer has passed its warm-up size, a uniform mini-batch of $B$ transitions is drawn.</span>

<span style="font-size: 14px;">4. **Learn:** the batch is fed through the network to compute the TD targets and the squared-error loss, and a gradient step updates $\theta$.</span>

<span style="font-size: 14px;">Because storing is cheap and sampling is independent of arrival order, the agent can take one environment step and several learning steps, or vice versa. This decoupling, enabled entirely by the buffer, is what turns a slow, expensive interaction stream into a fast, reusable training set. The most recent transition is no longer special: it is simply one more row that may or may not appear in any given batch.</span>

---

## <span style="font-size: 16px;">Off-Policy Learning Makes Replay Valid</span>

<span style="font-size: 14px;">Replay is only sound because Q-learning is **off-policy**: it learns the value of the greedy policy regardless of which policy generated the data. A transition collected long ago under a now-outdated, more exploratory policy is still a valid sample of the environment's dynamics $(s, a) \to (r, s')$, and the Bellman target $r + \gamma \max_{a'} Q(s', a')$ does not reference the behavior policy at all. This is what lets the buffer mix stale and fresh data freely. On-policy methods such as vanilla policy gradients cannot reuse a buffer this way, because their updates assume the data comes from the current policy.</span>

---

## Worked Example (capacity $= 3$)

<span style="font-size: 14px;">Start empty, $\texttt{head} = 0$. Push three transitions $T_0, T_1, T_2$:</span>

* <span style="font-size: 14px;">After pushes: $\texttt{buf} = [T_0, T_1, T_2]$, $\texttt{head} = 0$ (wrapped), size $= 3$ (full).</span>

<span style="font-size: 14px;">Now push a fourth transition $T_3$. The head points at index $0$, so $T_0$ is overwritten:</span>

* <span style="font-size: 14px;">$\texttt{buf} = [T_3, T_1, T_2]$, $\texttt{head} = 1$. The oldest experience $T_0$ is gone.</span>

<span style="font-size: 14px;">Sample with indices $(2, 0)$. Read $\texttt{buf}[2] = T_2 = (s_2, a_2, r_2, s_2', d_2)$ and $\texttt{buf}[0] = T_3 = (s_3, a_3, r_3, s_3', d_3)$, then transpose into columns:</span>

* <span style="font-size: 14px;">$\texttt{states} = (s_2, s_3)$, $\texttt{actions} = (a_2, a_3)$, $\texttt{rewards} = (r_2, r_3)$.</span>
* <span style="font-size: 14px;">$\texttt{next\_states} = (s_2', s_3')$, $\texttt{dones} = (d_2, d_3)$.</span>

<span style="font-size: 14px;">The output order follows the index order $(2, 0)$, not the physical buffer order, which is exactly what downstream code expects when it zips the columns back together.</span>

---

## <span style="font-size: 16px;">Design Decisions and Trade-offs</span>

<span style="font-size: 14px;">The buffer exposes a handful of hyperparameters with real effects on learning:</span>

* <span style="font-size: 14px;">**Capacity.** Too small and the buffer behaves almost like online learning, with recent, correlated data dominating; too large and it retains experience from policies so old that it slows adaptation. One million is a common default for Atari.</span>
* <span style="font-size: 14px;">**Sampling distribution.** Uniform sampling is the baseline. Prioritized Experience Replay (Schaul et al., 2016) instead samples in proportion to TD-error magnitude to focus on surprising transitions, at the cost of needing importance-sampling corrections.</span>
* <span style="font-size: 14px;">**Warm-up.** Learning usually does not begin until the buffer holds a minimum number of transitions, so the first mini-batches are not drawn from a near-empty, highly correlated store.</span>
* <span style="font-size: 14px;">**Batch size.** Larger batches give lower-variance gradients but cost more compute per update; $32$ is the classic DQN value.</span>

<span style="font-size: 14px;">A memory-aware variant stores each frame once and reconstructs stacked states on the fly, since consecutive transitions share most of their frames; this cuts the memory footprint of the one-million buffer by roughly the stack length.</span>

---

## <span style="font-size: 16px;">Variants Beyond Uniform Replay</span>

<span style="font-size: 14px;">The plain circular buffer is the baseline that later work extends:</span>

* <span style="font-size: 14px;">**Prioritized Experience Replay** (Schaul et al., 2016) replaces uniform sampling with sampling proportional to TD-error priority $p^\alpha$, so transitions the agent predicts poorly are replayed more often. It pairs this with importance-sampling weights to undo the bias the skewed sampling introduces, and typically uses a sum-tree to sample and update priorities in $O(\log N)$.</span>
* <span style="font-size: 14px;">**Hindsight Experience Replay** (Andrychowicz et al., 2017) relabels failed trajectories with the goals that were actually achieved, manufacturing useful reward signal from sparse-reward episodes without changing the buffer mechanics.</span>
* <span style="font-size: 14px;">**Reservoir and recency-weighted buffers** trade off how aggressively old data is discarded, relevant when the environment or policy distribution drifts over time.</span>

<span style="font-size: 14px;">All of these keep the same core abstraction: a store of self-contained transitions that can be sampled out of order, differing only in how an index is chosen and how the sampled loss is weighted.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Insertion is $O(1)$ time: one write and a pointer increment, with no shifting or reallocation thanks to the circular layout. Sampling a batch of size $B$ is $O(B)$ time, one read per index, plus the transpose into five lists. Space is $O(\texttt{capacity})$, fixed for the lifetime of training, which is the entire point of the circular design: a bounded, predictable memory budget regardless of how long the agent trains.</span>

<span style="font-size: 14px;">The constant-memory property matters at scale. A naive append-only list would grow without bound over the tens of millions of steps a DQN agent runs, eventually exhausting RAM. By overwriting in place, the circular buffer guarantees the footprint is decided once, at allocation time, by the chosen capacity. This predictability is why production RL frameworks preallocate the storage arrays up front and treat the buffer as a fixed ring rather than a dynamic container.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Sampling from uninitialized slots.** Before the buffer fills, valid indices range only over the number of transitions actually written, not the full capacity. Sampling up to $\texttt{capacity}$ too early returns garbage (empty or default-initialized) slots and silently corrupts the batch.</span>
* <span style="font-size: 14px;">**Losing the index ordering.** The returned columns must align with the requested index order. If the transpose reorders or sorts the indices, the $(s, a, r, s', d)$ fields of different transitions get mismatched, pairing one transition's state with another's reward and producing nonsensical, untraceable targets.</span>
* <span style="font-size: 14px;">**Storing references instead of copies.** If states are stored by reference to a mutable array the environment reuses, every slot can end up pointing at the same later observation. The buffer then appears full of identical states. Each transition must store an independent snapshot.</span>
* <span style="font-size: 14px;">**Forgetting the terminal flag or storing the wrong $s'$.** A transition that omits $d$, or that records the reset state of the next episode as $s'$, makes the learner bootstrap value across an episode boundary, injecting phantom future return into states that had none.</span>

---