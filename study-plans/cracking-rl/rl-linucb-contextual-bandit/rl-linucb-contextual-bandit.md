# <span style="font-size: 20px;">LinUCB Contextual Bandit</span>

<span style="font-size: 14px;">LinUCB (Li, Chu, Langford, and Schapire, 2010) extends the optimism-under-uncertainty principle of UCB1 from plain multi-armed bandits to **contextual** bandits, where a feature vector describing the current situation arrives before each decision. It assumes each arm's expected reward is a **linear** function of the context, fits that linear model online with ridge regression, and then selects the arm that maximizes a predicted reward plus a **confidence-ellipsoid** exploration bonus. This lets a single agent generalize across contexts rather than learning each situation from scratch, which is exactly what made it the workhorse for personalized news recommendation in the paper that introduced it.</span>

---

## <span style="font-size: 16px;">From Bandits to Contextual Bandits</span>

<span style="font-size: 14px;">A plain bandit assumes each arm has a single fixed mean reward. A **contextual** bandit instead reveals a context vector $x \in \mathbb{R}^d$ at every round, and the expected reward of an arm depends on that context. The agent must learn, per arm, how reward relates to context, then choose an arm given the current $x$. This is the regime between bandits and full reinforcement learning: there is context like in RL, but actions do not change future states, so there is no long-term credit assignment.</span>

<span style="font-size: 14px;">The explore-exploit dilemma persists and sharpens:</span>

* <span style="font-size: 14px;">**Exploitation** picks the arm whose learned model predicts the highest reward for the current context.</span>
* <span style="font-size: 14px;">**Exploration** picks an arm whose prediction is uncertain in this region of context space, to improve the model where it is weak.</span>

<span style="font-size: 14px;">Uncertainty is now **context-dependent**: an arm may be well understood for some contexts and poorly understood for others. LinUCB's bonus captures exactly this, growing large for contexts unlike anything the arm has seen.</span>

---

## <span style="font-size: 16px;">The Linear Reward Model and Ridge Regression</span>

<span style="font-size: 14px;">LinUCB models each arm $a$'s expected reward as a linear function of the context with an unknown coefficient vector $\theta_a$:</span>

$$
\mathbb{E}[r_{a} \mid x] = \theta_a^\top x
$$

<span style="font-size: 14px;">The coefficients are estimated by **ridge regression**, the $\ell_2$-regularized least squares fit over the contexts in which arm $a$ was pulled and the rewards observed. Maintaining the fit online requires only two per-arm accumulators:</span>

* <span style="font-size: 14px;">**Gram matrix** $A_a = \lambda I + \sum x x^\top$, summing the outer products of every context where arm $a$ was chosen, plus a regularization term $\lambda I$. The $\lambda I$ makes $A_a$ invertible even before any data arrives.</span>
* <span style="font-size: 14px;">**Response vector** $b_a = \sum r\, x$, summing each observed reward times its context.</span>

<span style="font-size: 14px;">The ridge solution in closed form is then:</span>

$$
\theta_a = A_a^{-1} b_a
$$

<span style="font-size: 14px;">This is the unique minimizer of $\sum (r - \theta_a^\top x)^2 + \lambda \|\theta_a\|^2$. The regularization both stabilizes the inverse and shrinks the estimate toward zero when data is scarce, which is the right default for an arm with few observations.</span>

<span style="font-size: 14px;">Keeping a separate model per arm is the **disjoint** formulation, and it has a clean interpretation: each arm runs its own ridge regression, and the arms interact only through the shared selection rule, not through shared parameters. The two accumulators $A_a$ and $b_a$ are sufficient statistics for that regression, so the agent never needs to store the raw history of contexts and rewards, exactly as the Beta posterior summarizes a Bernoulli arm's history. This bounded memory, $O(d^2)$ per arm for $A_a$ plus $O(d)$ for $b_a$, is what lets LinUCB run online indefinitely without its storage growing with the number of rounds.</span>

---

## <span style="font-size: 16px;">The Confidence Ellipsoid Bonus</span>

<span style="font-size: 14px;">The action-selection rule combines the predicted mean with an optimism bonus, mirroring UCB1's exploit-plus-bonus structure but now in feature space:</span>

$$
\text{UCB}_a = \theta_a^\top x + \alpha \sqrt{x^\top A_a^{-1} x}
$$

<span style="font-size: 14px;">The two terms:</span>

* <span style="font-size: 14px;">**Exploitation term** $\theta_a^\top x$: the ridge model's predicted reward for arm $a$ in the current context.</span>
* <span style="font-size: 14px;">**Exploration bonus** $\alpha\sqrt{x^\top A_a^{-1} x}$: the standard error of that prediction under the ridge posterior, scaled by $\alpha > 0$.</span>

<span style="font-size: 14px;">The bonus is the natural generalization of UCB1's $\sqrt{\ln t / N[a]}$. In the plain bandit, $N[a]$ measures how many times an arm was pulled; here $x^\top A_a^{-1} x$ measures how much the current context aligns with directions the arm has already explored. If $x$ points in a direction with many past observations, $A_a^{-1}$ is small along that direction and the bonus is tiny. If $x$ points somewhere the arm has rarely been, $A_a^{-1}$ is large there and the bonus is big. The quantity $\sqrt{x^\top A_a^{-1} x}$ is the **Mahalanobis norm** of $x$ under $A_a^{-1}$, and it defines a **confidence ellipsoid** around $\theta_a$: the true coefficient lies inside that ellipsoid with high probability, so adding the bonus gives an optimistic upper bound on the reward exactly as UCB1 does for scalars.</span>

---

## <span style="font-size: 16px;">Why the Mahalanobis Norm Measures Uncertainty</span>

<span style="font-size: 14px;">The bonus deserves a closer look because it is the heart of why LinUCB explores intelligently. The ridge estimator $\theta_a$ is, under a Gaussian noise model, the mean of a posterior whose covariance is proportional to $A_a^{-1}$. The variance of the predicted reward $\theta_a^\top x$ is therefore proportional to $x^\top A_a^{-1} x$, and its standard deviation is the square root. The bonus is literally one standard error of the prediction, scaled by $\alpha$.</span>

<span style="font-size: 14px;">Geometrically, $A_a = \lambda I + \sum x x^\top$ accumulates information along every direction the arm has been observed. A direction seen many times contributes large eigenvalues to $A_a$, hence small eigenvalues to $A_a^{-1}$, hence a small bonus: the model is confident there. A direction never seen keeps only the $\lambda$ floor, so $A_a^{-1}$ stays large and the bonus is high: the model is guessing. Two consequences follow that make the policy well-behaved:</span>

* <span style="font-size: 14px;">**The bonus shrinks as evidence accumulates,** because every pull adds $x x^\top$ to $A_a$ and monotonically decreases $x^\top A_a^{-1} x$ for related contexts. An arm's optimism in a familiar context decays toward zero, so the policy eventually exploits.</span>
* <span style="font-size: 14px;">**The bonus is direction-specific,** so an arm can be confidently exploited for one kind of context while still being explored for an unfamiliar one. This is impossible in a context-free bandit, where uncertainty is a single scalar per arm.</span>

---

## <span style="font-size: 16px;">The Algorithm</span>

<span style="font-size: 14px;">LinUCB runs as a per-round loop, maintaining $A_a$ and $b_a$ for every arm:</span>

<span style="font-size: 14px;">1. **Initialize** each arm with $A_a = \lambda I$ (commonly $\lambda = 1$, giving $A_a = I$) and $b_a = 0$.</span>

<span style="font-size: 14px;">2. **Observe** the context $x$ for the current round.</span>

<span style="font-size: 14px;">3. **Score** every arm: compute $\theta_a = A_a^{-1} b_a$ and $\text{UCB}_a = \theta_a^\top x + \alpha\sqrt{x^\top A_a^{-1} x}$.</span>

<span style="font-size: 14px;">4. **Select** the arm $a_t = \arg\max_a \text{UCB}_a$, breaking ties by a fixed rule, and pull it.</span>

<span style="font-size: 14px;">5. **Update** the chosen arm with the observed reward $r$: $A_{a_t} \leftarrow A_{a_t} + x x^\top$ and $b_{a_t} \leftarrow b_{a_t} + r x$.</span>

<span style="font-size: 14px;">Each update is a rank-one modification of the Gram matrix, and the Sherman-Morrison formula lets the inverse be updated in $O(d^2)$ rather than recomputed in $O(d^3)$, which is what makes LinUCB practical at scale. Like UCB1, the policy is **deterministic** given the history; all exploration comes from the bonus, not from randomness.</span>

---

## <span style="font-size: 16px;">Regret and the Role of $\alpha$</span>

<span style="font-size: 14px;">The constant $\alpha$ tunes the explore-exploit balance, just as $c$ does in UCB1. Larger $\alpha$ inflates the confidence ellipsoid and explores more; smaller $\alpha$ trusts the predicted mean and exploits more. The theory ties $\alpha$ to a high-probability bound: setting $\alpha = 1 + \sqrt{\ln(2/\delta)/2}$ makes the optimistic estimate a valid upper bound with probability $1 - \delta$.</span>

<span style="font-size: 14px;">**Regret** is again the cumulative reward gap against an oracle that knows every $\theta_a$. For the linear contextual bandit with $d$-dimensional contexts and $T$ rounds, the LinUCB family (analyzed rigorously by Abbasi-Yadkori et al., 2011, as OFUL) achieves:</span>

$$
\text{Regret}(T) = \tilde{O}\!\left(d\sqrt{T}\right)
$$

<span style="font-size: 14px;">where $\tilde{O}$ hides logarithmic factors. The dependence is on the feature dimension $d$ rather than on the number of arms $K$, which is the central payoff of the linear assumption: an agent can handle huge or even changing arm sets as long as each arm is described by a $d$-dimensional feature vector. This is what fixed $\epsilon$-greedy can never achieve, since its $\Theta(T)$ linear regret has no way to share information across contexts or arms.</span>

<span style="font-size: 14px;">The original paper validated this on the Yahoo Front Page Today module, choosing which news article to show each visitor based on user and article features. With millions of users and a rotating pool of articles, a context-free bandit would have to relearn every article-user combination independently, but LinUCB shares a learned mapping from features to clicks across all of them. The reported click-through-rate gains over a context-free baseline came precisely from this generalization, demonstrating that the $d\sqrt{T}$ rather than $K$-dependent regret is not just a theoretical nicety but the practical reason the method scales.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Take $d = 2$, two arms, and context $x = [1,\ 0]^\top$, with $\alpha = 1$. Suppose arm 0 has been pulled often along the first feature, so $A_0 = \begin{pmatrix} 5 & 0 \\ 0 & 1 \end{pmatrix}$, $b_0 = [4,\ 0]^\top$, while arm 1 is fresh, $A_1 = I$, $b_1 = [0.5,\ 0]^\top$.</span>

<span style="font-size: 14px;">1. **Arm 0:** $A_0^{-1} = \text{diag}(0.2,\ 1)$, so $\theta_0 = A_0^{-1} b_0 = [0.8,\ 0]^\top$. Mean: $\theta_0^\top x = 0.8$. Bonus: $\sqrt{x^\top A_0^{-1} x} = \sqrt{0.2} = 0.4472$, so $\text{UCB}_0 = 0.8 + 0.4472 = 1.2472$.</span>

<span style="font-size: 14px;">2. **Arm 1:** $A_1^{-1} = I$, so $\theta_1 = [0.5,\ 0]^\top$. Mean: $\theta_1^\top x = 0.5$. Bonus: $\sqrt{x^\top I x} = \sqrt{1} = 1.0$, so $\text{UCB}_1 = 0.5 + 1.0 = 1.5$.</span>

<span style="font-size: 14px;">Arm 1 wins ($1.5 > 1.2472$) despite predicting a lower mean reward. Its larger bonus reflects that it has barely been explored in this context direction, so the agent optimistically tries it. This is the contextual analogue of UCB1 favoring an undersampled arm, with uncertainty now measured along the specific direction of $x$.</span>

---

## <span style="font-size: 16px;">LinUCB in Context</span>

<span style="font-size: 14px;">LinUCB connects the simple methods of this section to modern large-scale exploration:</span>

* <span style="font-size: 14px;">**Disjoint vs hybrid models.** The original paper presents a disjoint model with separate $\theta_a$ per arm and a hybrid model that adds shared features across arms, letting arms with little data borrow strength from related arms.</span>
* <span style="font-size: 14px;">**Relationship to UCB1.** With a single constant feature $x = 1$ and $\lambda = 0$, $A_a$ reduces to the pull count $N[a]$ and the bonus to $\alpha / \sqrt{N[a]}$, recovering a UCB1-style rule. LinUCB is genuinely a contextual generalization.</span>
* <span style="font-size: 14px;">**Bayesian counterpart.** LinTS (linear Thompson Sampling) replaces the deterministic ellipsoid bonus with a sample from the Gaussian posterior over $\theta_a$, the contextual version of Beta-Bernoulli Thompson Sampling, and is often preferred for its empirical performance and simpler tuning.</span>

<span style="font-size: 14px;">The linear assumption is also the gateway to nonlinear extensions: kernelizing gives KernelUCB and GP-UCB, while replacing the linear model with a neural network gives Neural-UCB, all of which keep the same optimism-plus-uncertainty skeleton introduced here.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting the regularization term in $A_a$.** Without $\lambda I$, the Gram matrix $\sum x x^\top$ is singular until at least $d$ linearly independent contexts have been seen, so $A_a^{-1}$ does not exist and the score is undefined. The $\lambda I$ is what guarantees invertibility from the very first round.</span>
* <span style="font-size: 14px;">**Updating the wrong arm's statistics.** Only the arm that was actually pulled receives the $x x^\top$ and $r x$ updates. Updating all arms, or the highest-scoring arm rather than the played one, corrupts every model and destroys the regret guarantee.</span>
* <span style="font-size: 14px;">**Recomputing the inverse from scratch each round.** Naively inverting $A_a$ every step is $O(d^3)$ per arm and dominates runtime. The rank-one Sherman-Morrison update maintains $A_a^{-1}$ incrementally in $O(d^2)$; missing this makes LinUCB needlessly slow at scale.</span>
* <span style="font-size: 14px;">**Mismatched feature scaling.** Because the bonus is the Mahalanobis norm $\sqrt{x^\top A_a^{-1} x}$, features on wildly different scales distort both the ridge fit and the exploration geometry. Standardizing context features keeps the confidence ellipsoid meaningful and $\alpha$ interpretable.</span>

---