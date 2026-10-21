# <span style="font-size: 20px;">Thompson Sampling (Beta-Bernoulli)</span>

<span style="font-size: 14px;">Thompson Sampling (Thompson, 1933) is a Bayesian approach to the explore-exploit problem that maintains a **posterior distribution** over each arm's unknown reward probability and explores by **sampling** from those posteriors. For a Bernoulli bandit it uses a **Beta** posterior per arm, which is the conjugate prior for the Bernoulli likelihood, so every observed pull updates the posterior in closed form. Exploration is automatic: an arm we are uncertain about has a wide posterior, so its sampled value is more likely to come out high and win the round. This is **probability matching**, choosing each arm with exactly the probability that it is optimal given the data.</span>

---

## <span style="font-size: 16px;">The Explore-Exploit Dilemma</span>

<span style="font-size: 14px;">A Bernoulli $K$-armed bandit offers $K$ arms, each yielding reward $1$ with unknown probability $\theta_a$ and $0$ otherwise. The agent sees only the outcome of the arm it pulls, so it must learn the $\theta_a$ from binary feedback while still earning reward. This is the standard **exploration versus exploitation** tension:</span>

* <span style="font-size: 14px;">**Exploitation** plays the arm with the best estimated success rate, risking that an undersampled arm is truly better.</span>
* <span style="font-size: 14px;">**Exploration** plays an uncertain arm to sharpen its estimate, at the cost of likely lower immediate reward.</span>

<span style="font-size: 14px;">Where $\epsilon$-greedy injects undirected randomness and UCB1 adds a deterministic optimism bonus, Thompson Sampling resolves the dilemma through Bayesian uncertainty: it explores in proportion to the probability that each arm is the best, which falls out naturally from sampling the posteriors rather than being engineered as a separate exploration term.</span>

---

## <span style="font-size: 16px;">The Beta-Bernoulli Conjugate Pair</span>

<span style="font-size: 14px;">A **Beta** distribution $\text{Beta}(\alpha, \beta)$ over a probability $\theta \in [0, 1]$ has density:</span>

$$
p(\theta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}}{B(\alpha, \beta)}
$$

<span style="font-size: 14px;">where $B(\alpha, \beta)$ is the Beta function normalizer. The parameters act as **pseudo-counts**: $\alpha$ behaves like a count of prior successes and $\beta$ like a count of prior failures. The mean and variance are:</span>

$$
\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}, \qquad \text{Var}[\theta] = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}
$$

<span style="font-size: 14px;">The Beta is the **conjugate prior** for the Bernoulli likelihood, meaning the posterior after observing Bernoulli data is again a Beta. Starting from a prior $\text{Beta}(\alpha, \beta)$ and observing reward $r \in \{0, 1\}$, the posterior is:</span>

$$
\alpha_a \leftarrow \alpha_a + r, \qquad \beta_a \leftarrow \beta_a + (1 - r)
$$

<span style="font-size: 14px;">A success ($r = 1$) bumps $\alpha$ by one; a failure ($r = 0$) bumps $\beta$ by one. This closed-form update is the entire reason the Beta-Bernoulli pairing is so attractive: there is no integral to evaluate, no sampling needed to maintain the posterior, just two integer counters per arm. Arms that are never pulled keep their prior $(\alpha, \beta)$ unchanged.</span>

---

## <span style="font-size: 16px;">Why Conjugacy Gives a Closed Form</span>

<span style="font-size: 14px;">The closed-form update is a direct consequence of Bayes' rule combined with the matching algebraic shapes of the prior and likelihood. The Bernoulli likelihood of a single reward $r$ is $\theta^r (1 - \theta)^{1 - r}$. Multiplying it by the Beta prior gives:</span>

$$
p(\theta \mid r) \propto \theta^r(1 - \theta)^{1 - r} \cdot \theta^{\alpha - 1}(1 - \theta)^{\beta - 1} = \theta^{(\alpha + r) - 1}(1 - \theta)^{(\beta + 1 - r) - 1}
$$

<span style="font-size: 14px;">The result has exactly the functional form of a Beta density with parameters $\alpha + r$ and $\beta + (1 - r)$, so the posterior is $\text{Beta}(\alpha + r,\ \beta + 1 - r)$. The exponents of $\theta$ and $1 - \theta$ simply accumulate the observed successes and failures. Because each observation contributes additively to the exponents, folding in a whole stream of observations is equivalent to adding the total successes to $\alpha$ and the total failures to $\beta$, regardless of order.</span>

<span style="font-size: 14px;">This order-independence is more than a convenience; it is a statement that the posterior is a **sufficient statistic** for the data. After processing any sequence of pulls for an arm, the pair $(\alpha_a, \beta_a)$ captures everything the rewards say about $\theta_a$. The individual outcomes can be discarded. For the deterministic update task this means the answer depends only on the total number of successes and failures routed to each arm, so the same observation multiset always yields the same final parameters, which is exactly why the task is reproducible without seeds. Only the sampling step introduces randomness, and it is deliberately left out here.</span>

---

## <span style="font-size: 16px;">The Sampling Step and Probability Matching</span>

<span style="font-size: 14px;">The deterministic posterior maintained above is only half of Thompson Sampling. The stochastic step that gives the algorithm its name, excluded from this particular task so the answer is reproducible without seeds, works as follows on each round:</span>

<span style="font-size: 14px;">1. **Sample** a candidate success rate for every arm from its current posterior, $\tilde\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$.</span>

<span style="font-size: 14px;">2. **Select** the arm with the highest sampled value, $a_t = \arg\max_a \tilde\theta_a$, and pull it.</span>

<span style="font-size: 14px;">3. **Update** that arm's posterior with the observed reward using the conjugate rule.</span>

<span style="font-size: 14px;">This selection rule implements **probability matching**: arm $a$ is chosen with probability equal to the posterior probability that it is the best arm, $P(a = \arg\max_b \theta_b \mid \text{data})$. An arm with a wide posterior (few pulls) occasionally samples a very high $\tilde\theta_a$ and gets explored; an arm whose posterior has concentrated near a low mean rarely samples high and is naturally abandoned. Exploration and exploitation emerge from the same sampling operation, with no separate tuning knob like $\epsilon$ or $c$.</span>

<span style="font-size: 14px;">Probability matching is subtly different from both greedy exploitation and uniform exploration. A greedy agent would always pick the current posterior-mean leader, collapsing exploration to zero. A uniform explorer would pick every arm equally, ignoring the data. Thompson Sampling threads between them: an arm that the data say is best with $80\%$ posterior probability is pulled about $80\%$ of the time, and the remaining $20\%$ is spent on the arms that still have a plausible chance of being best. As the posteriors sharpen, the matching probability of the true best arm climbs toward $1$, so the strategy is greedy in the limit while remaining exploratory exactly as long as genuine uncertainty exists.</span>

---

## <span style="font-size: 16px;">How the Posterior Shape Drives Exploration</span>

<span style="font-size: 14px;">The reason sampling produces the right amount of exploration is entirely in the geometry of the Beta posterior. Two quantities matter: where the posterior sits (its mean) and how spread out it is (its variance). As pulls accumulate, $\alpha + \beta$ grows and the variance $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ shrinks like $1 / (\alpha + \beta)$, so the posterior concentrates around the true $\theta_a$.</span>

* <span style="font-size: 14px;">**Early, wide posterior:** with few pulls the density is broad. Samples scatter widely, so even an arm with a modest mean occasionally draws a high $\tilde\theta_a$ and gets explored. This is the source of optimism, but it is data-driven rather than a fixed bonus.</span>
* <span style="font-size: 14px;">**Late, narrow posterior:** once an arm is well sampled its density spikes near the true rate. A genuinely good arm reliably samples high and is exploited; a genuinely bad arm reliably samples low and is dropped.</span>

<span style="font-size: 14px;">This self-annealing is the key contrast with $\epsilon$-greedy. There is no schedule to tune: the exploration rate decays automatically as evidence sharpens each posterior, which is why Thompson Sampling needs no $\epsilon$ decay and no $c$ constant yet still reaches logarithmic regret.</span>

---

## <span style="font-size: 16px;">Regret and Empirical Strength</span>

<span style="font-size: 14px;">**Regret** is the cumulative reward lost relative to always playing the optimal arm:</span>

$$
\text{Regret}(T) = T\theta^* - \sum_{t=1}^{T}\mathbb{E}[\theta_{a_t}] = \sum_{a}\Delta_a\,\mathbb{E}[N_a(T)]
$$

<span style="font-size: 14px;">where $\Delta_a = \theta^* - \theta_a$ is the suboptimality gap. Agrawal and Goyal (2012) and Kaufmann et al. (2012) proved that Thompson Sampling achieves $O(\log T)$ regret for Bernoulli bandits, matching the Lai and Robbins (1985) asymptotic lower bound and the order achieved by UCB1. Crucially, it attains this with a constant that is often better in practice: across many empirical studies Thompson Sampling outperforms UCB1, especially when the number of arms is large or rewards are sparse, because its posterior sampling adapts the exploration rate to the actual evidence rather than to a worst-case bound. Where fixed $\epsilon$-greedy suffers $\Theta(T)$ linear regret from undirected exploration, Thompson Sampling stops sampling clearly inferior arms once their posteriors concentrate below the leader's.</span>

<span style="font-size: 14px;">The method's history underscores how durable the idea is. Thompson proposed posterior sampling in 1933 in the context of clinical trials, deciding which of two treatments to assign to the next patient, but it was largely overlooked in favor of frequentist methods for decades. Interest revived after Chapelle and Li (2011) showed empirically that it beats UCB on large-scale display-advertising bandits, and the theoretical regret guarantees followed shortly after. Today it is a default exploration strategy in production recommendation and A/B-testing systems precisely because it balances strong guarantees with simple, cheap conjugate updates.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $K = 3$ arms with uniform priors $\alpha_a = 1,\ \beta_a = 1$ for all arms (the $\text{Beta}(1, 1)$ prior is the uniform distribution on $[0, 1]$). Process the observation stream $[(0, 1),\ (0, 0),\ (1, 1),\ (1, 1),\ (2, 0)]$ where each pair is $(\text{arm}, r)$.</span>

<span style="font-size: 14px;">1. **$(0, 1)$:** arm 0 success, $\alpha_0 = 1 + 1 = 2$, $\beta_0 = 1$.</span>

<span style="font-size: 14px;">2. **$(0, 0)$:** arm 0 failure, $\alpha_0 = 2$, $\beta_0 = 1 + 1 = 2$.</span>

<span style="font-size: 14px;">3. **$(1, 1)$:** arm 1 success, $\alpha_1 = 2$, $\beta_1 = 1$.</span>

<span style="font-size: 14px;">4. **$(1, 1)$:** arm 1 success again, $\alpha_1 = 3$, $\beta_1 = 1$.</span>

<span style="font-size: 14px;">5. **$(2, 0)$:** arm 2 failure, $\alpha_2 = 1$, $\beta_2 = 1 + 1 = 2$.</span>

<span style="font-size: 14px;">The final state is $\alpha = [2,\ 3,\ 1]$ and $\beta = [2,\ 1,\ 2]$. Posterior means: arm 0 has $2 / 4 = 0.5$, arm 1 has $3 / 4 = 0.75$, arm 2 has $1 / 3 \approx 0.333$. Arm 1 not only has the highest mean but the tightest posterior (most evidence), so under the sampling step it would be selected most often, while arm 0's symmetric posterior keeps it in contention and arm 2 is mostly written off.</span>

<span style="font-size: 14px;">It is worth confirming the order-independence claim on this example. Arm 0 received one success and one failure regardless of when they arrived, so it lands at $(2, 2)$; arm 1 received two successes, landing at $(3, 1)$; arm 2 received one failure, landing at $(1, 2)$. Shuffling the five observations into any other order produces identical final parameters, because each only adds $1$ to one of the two counters. This is the practical face of sufficiency: the agent never needs to remember the sequence, only the running totals.</span>

---

## <span style="font-size: 16px;">Choosing the Prior and Context</span>

<span style="font-size: 14px;">The prior $(\alpha, \beta)$ encodes belief before any data and shapes early exploration:</span>

* <span style="font-size: 14px;">**Uniform $\text{Beta}(1, 1)$:** the flat, uninformative default. Every success rate is equally likely a priori, so the posterior is driven entirely by data.</span>
* <span style="font-size: 14px;">**Jeffreys $\text{Beta}(0.5, 0.5)$:** a less informative prior that places more mass near $0$ and $1$, useful when extreme success rates are plausible.</span>
* <span style="font-size: 14px;">**Informative priors:** larger pseudo-counts encode prior knowledge and slow the posterior's response to new data, which can stabilize learning when reliable domain estimates exist.</span>

<span style="font-size: 14px;">Within this section's family of methods, Thompson Sampling sits opposite UCB1: both reach logarithmic regret, but UCB1 is frequentist and deterministic while Thompson Sampling is Bayesian and stochastic. The conjugate update generalizes well beyond Bernoulli; Gaussian rewards use a Normal-Inverse-Gamma posterior, and contextual versions sample from a posterior over a linear model, the Bayesian analogue of LinUCB.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Mixing up which counter a reward updates.** A success ($r = 1$) increments $\alpha$ and a failure ($r = 0$) increments $\beta$. Swapping them inverts every posterior and makes the agent prefer the worst arms. The mnemonic is that $\alpha$ counts ones.</span>
* <span style="font-size: 14px;">**Starting parameters at zero.** A $\text{Beta}(0, 0)$ is improper and its mean $\alpha / (\alpha + \beta)$ is undefined. Priors must be strictly positive; $\text{Beta}(1, 1)$ is the safe uniform default.</span>
* <span style="font-size: 14px;">**Forgetting that unpulled arms keep their priors.** Arms absent from the observation stream must retain their initial $(\alpha, \beta)$ unchanged. Resetting them or skipping their entries produces the wrong-length output.</span>
* <span style="font-size: 14px;">**Confusing the posterior mean with a sample.** The selection step samples $\tilde\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$; it does not pick the arm with the highest posterior mean. Using the mean directly discards the variance that drives exploration and collapses Thompson Sampling into a near-greedy policy.</span>

---