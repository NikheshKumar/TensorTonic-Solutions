# <span style="font-size: 20px;">Bradley-Terry Reward Model Loss</span>

<span style="font-size: 14px;">The Bradley-Terry reward loss is the objective used to train the **reward model** in RLHF (Reinforcement Learning from Human Feedback). Introduced as the preference-modeling stage of InstructGPT (Ouyang et al., 2022), it fits a scalar reward function $r_\theta(x, y)$ to pairwise human comparisons by maximizing the log-likelihood that the chosen response outscores the rejected one.</span>

---

## <span style="font-size: 16px;">Where This Sits in the RLHF Pipeline</span>

<span style="font-size: 14px;">Modern LLM alignment has three classical stages. The reward loss covered here is the second.</span>

* <span style="font-size: 14px;">**Stage 1 - Supervised fine-tuning (SFT):** the base model is fine-tuned on human demonstrations to follow instructions, producing a policy $\pi_{ref}$ that is also used as the frozen reference later.</span>
* <span style="font-size: 14px;">**Stage 2 - Reward modeling:** humans rank model outputs, and a reward model is trained to predict those preferences. This is the Bradley-Terry stage.</span>
* <span style="font-size: 14px;">**Stage 3 - RL fine-tuning:** the SFT policy is optimized against the learned reward, usually with PPO and a KL penalty to $\pi_{ref}$.</span>

<span style="font-size: 14px;">Preference-optimization alternatives such as DPO, GRPO, and RLOO restructure or remove parts of this pipeline, but they all trace back to the same Bradley-Terry assumption about how humans compare two outputs.</span>

<span style="font-size: 14px;">The motivation for a learned reward model is that human preferences are expensive and sparse. A labeler can comfortably compare two completions and say which is better, but cannot assign a calibrated numeric score to a single completion, and cannot label the millions of samples an RL loop will generate. A reward model amortizes a finite set of comparisons into a differentiable function that scores any new completion, turning a small human-labeled dataset into a dense training signal for the policy.</span>

---

## <span style="font-size: 16px;">Why Preferences Instead of Absolute Scores</span>

<span style="font-size: 14px;">RLHF deliberately collects **rankings** rather than absolute quality ratings. Humans are noisy and inconsistent when asked for an absolute number on a scale, but far more reliable at relative judgments: given two answers, which is better. The Bradley-Terry model is the natural probabilistic bridge from these relative judgments back to a latent absolute score.</span>

* <span style="font-size: 14px;">**Calibration:** different labelers anchor numeric scales differently, so absolute ratings are not comparable across people. Pairwise wins are anchor-free.</span>
* <span style="font-size: 14px;">**Transitivity:** the model assumes a single latent reward exists such that preferences follow score differences. Real preferences can violate transitivity, and the loss tolerates this by treating each pair as an independent noisy observation.</span>
* <span style="font-size: 14px;">**Efficiency:** a $K$-way ranking yields $\binom{K}{2}$ pairwise constraints from one labeling session, which is why InstructGPT collected rankings of $K$ between 4 and 9 completions per prompt.</span>

---

## <span style="font-size: 16px;">The Bradley-Terry Preference Model</span>

<span style="font-size: 14px;">The Bradley-Terry model (Bradley and Terry, 1952) is a classical model of pairwise comparisons. Each item gets a latent score, and the probability that item $A$ beats item $B$ is a logistic function of the difference of their scores. In RLHF the latent score is the reward model output $r$, so for a chosen response $y_w$ and a rejected response $y_l$ given prompt $x$:</span>

$$
P(y_w \succ y_l \mid x) = \frac{\exp r_\theta(x, y_w)}{\exp r_\theta(x, y_w) + \exp r_\theta(x, y_l)} = \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)
$$

<span style="font-size: 14px;">where $\sigma(z) = 1 / (1 + e^{-z})$ is the logistic sigmoid. Only the **difference** of rewards matters, which means the reward model is identified only up to an additive constant. This shift-invariance is intentional: in RLHF only relative reward differences drive policy updates.</span>

---

## <span style="font-size: 16px;">The Loss Function</span>

<span style="font-size: 14px;">Training maximizes the log-likelihood of the observed human preferences, which is the same as minimizing the negative log-likelihood. With $r^{w}_i$ the reward for the chosen response and $r^{l}_i$ the reward for the rejected response in comparison $i$, averaged over $N$ pairs:</span>

$$
L = -\frac{1}{N}\sum_{i=1}^{N} \log \sigma\!\left(r^{w}_i - r^{l}_i\right)
$$

<span style="font-size: 14px;">The term inside the sigmoid, $m_i = r^{w}_i - r^{l}_i$, is the **preference margin**. The loss for a single pair is $-\log\sigma(m_i)$, which is exactly the binary logistic (cross-entropy) loss applied to the margin with a target label of 1, meaning the chosen response should always win.</span>

* <span style="font-size: 14px;">When $m_i \to +\infty$ the model is confident the right response won, $\sigma(m_i) \to 1$, and the loss $\to 0$.</span>
* <span style="font-size: 14px;">When $m_i = 0$ the model is indifferent, $\sigma(0) = 0.5$, and the loss is $-\log 0.5 \approx 0.6931$.</span>
* <span style="font-size: 14px;">When $m_i \to -\infty$ the model ranks the pair backwards and the loss grows without bound, applying a strong gradient.</span>

<span style="font-size: 14px;">Because the loss is unbounded below in the margin, a single grossly misranked pair can dominate the batch average, which is one reason the margin variant and gradient clipping are common in practice.</span>

---

## <span style="font-size: 16px;">Gradient and What It Learns</span>

<span style="font-size: 14px;">The derivative of the per-pair loss with respect to the margin is clean:</span>

$$
\frac{\partial}{\partial m}\left[-\log\sigma(m)\right] = -\left(1 - \sigma(m)\right) = \sigma(m) - 1
$$

<span style="font-size: 14px;">The gradient is the prediction error $\sigma(m) - 1$, ranging in $(-1, 0)$. Confident-correct pairs ($\sigma(m)$ near 1) contribute almost no gradient, while wrong or uncertain pairs dominate the update. By the chain rule this pushes $r_\theta(x, y_w)$ up and $r_\theta(x, y_l)$ down, increasing the margin between chosen and rejected responses.</span>

<span style="font-size: 14px;">Expanding through the network parameters, the gradient with respect to $\theta$ is:</span>

$$
\nabla_\theta L = -\frac{1}{N}\sum_{i=1}^{N}\left(1 - \sigma(m_i)\right)\left(\nabla_\theta r^{w}_i - \nabla_\theta r^{l}_i\right)
$$

<span style="font-size: 14px;">The scalar weight $1 - \sigma(m_i)$ is a per-pair difficulty weight that emerges automatically. This is a built-in form of hard-example mining: there is no need to manually upweight difficult comparisons, because the logistic loss does it. It also means that once a pair is comfortably ranked correctly, it stops contributing, so the model does not waste capacity pushing already-separated margins to infinity.</span>

<span style="font-size: 14px;">In InstructGPT the reward model is the SFT model with the unembedding head replaced by a single scalar output. Ouyang et al. (2022) used a 6B reward model and trained on all $\binom{K}{2}$ pairs from each $K$-way ranking, batching the comparisons from one prompt together to avoid overfitting on repeated completions.</span>

<span style="font-size: 14px;">A subtle but important detail from the paper: batching all comparisons from a single prompt into one forward pass is not just an efficiency trick. If the $\binom{K}{2}$ pairs from one prompt are scattered across different minibatches, each completion is fed through the reward model up to $K-1$ times across an epoch, and the model overfits on those repeated completions, degrading validation accuracy. Treating each prompt's full ranking as a single training element fixes this.</span>

---

## <span style="font-size: 16px;">Reward Modeling as Logistic Regression</span>

<span style="font-size: 14px;">It is worth recognizing the loss as ordinary binary logistic regression in disguise. Define a single feature, the margin $m = r^{w} - r^{l}$, and a fixed label $t = 1$. The cross-entropy loss is $-[t\log\sigma(m) + (1-t)\log(1-\sigma(m))] = -\log\sigma(m)$. So the reward head is a logistic classifier whose input is the difference of two scalar rewards. This framing explains several properties immediately:</span>

* <span style="font-size: 14px;">The loss is convex in the margin, so optimization is well behaved with respect to $m$, even though it is non-convex in the full network parameters $\theta$.</span>
* <span style="font-size: 14px;">The minimum achievable loss on a noisy pair where humans agree with probability $p$ is the binary entropy $-p\log p - (1-p)\log(1-p)$, not zero. A reward model that perfectly matches human preference probabilities still has nonzero loss on ambiguous pairs.</span>
* <span style="font-size: 14px;">Calibration matters: at the optimum, $\sigma(m)$ recovers the true preference probability, so a well-trained reward model's sigmoid output is interpretable as the chance a human prefers the chosen response.</span>

---

## <span style="font-size: 16px;">Numerically Stable Log-Sigmoid</span>

<span style="font-size: 14px;">A naive implementation computes $\sigma(m)$ first and then takes its log. For large negative $m$, $\sigma(m)$ underflows to 0 and $\log 0 = -\infty$. For large positive $m$, $e^{-m}$ underflows harmlessly but the intermediate can still lose precision. The stable identity is:</span>

$$
\log\sigma(m) = -\log\!\left(1 + e^{-m}\right) = -\,\text{softplus}(-m)
$$

<span style="font-size: 14px;">Using $\text{softplus}(z) = \max(z, 0) + \log(1 + e^{-|z|})$ keeps the exponent argument non-positive, so $e^{-|z|} \in (0, 1]$ never overflows. With this form the loss is well defined even for margins of $\pm 100$. This is exactly why the problem asks for a stable log-sigmoid rather than chaining `log` after `sigmoid`.</span>

<span style="font-size: 14px;">Concretely, for a margin of $m = -50$, the naive path computes $\sigma(-50) \approx 1.9 \times 10^{-22}$, which in single precision underflows toward zero, and $\log 0$ returns $-\infty$, poisoning the batch average with `NaN` after the negation. The stable path computes $-\log(1 + e^{50})$, which in the $\max$-shifted softplus form becomes $-(50 + \log(1 + e^{-50})) \approx -50$, the correct large finite loss. Frameworks expose this directly as `F.logsigmoid` in PyTorch or `binary_cross_entropy_with_logits`, and production reward-model code should use those rather than reimplementing the sigmoid.</span>

---

## <span style="font-size: 16px;">Worked Example</span>

<span style="font-size: 14px;">Suppose $N = 2$ comparisons with rewards: pair 1 has $r^{w} = 2.0$, $r^{l} = 0.5$; pair 2 has $r^{w} = 1.0$, $r^{l} = 1.5$.</span>

<span style="font-size: 14px;">1. **Margins**: $m_1 = 2.0 - 0.5 = 1.5$ and $m_2 = 1.0 - 1.5 = -0.5$. Pair 2 is ranked backwards by the current model.</span>

<span style="font-size: 14px;">2. **Sigmoids**: $\sigma(1.5) \approx 0.8176$ and $\sigma(-0.5) \approx 0.3775$.</span>

<span style="font-size: 14px;">3. **Per-pair losses**: $-\log(0.8176) \approx 0.2014$ and $-\log(0.3775) \approx 0.9741$.</span>

<span style="font-size: 14px;">4. **Average**: $L = (0.2014 + 0.9741) / 2 \approx 0.5878$, rounded to $0.5878$. The backwards pair contributes most of the loss, as expected.</span>

<span style="font-size: 14px;">Checking the gradient weights confirms the intuition: pair 1 has weight $1 - \sigma(1.5) = 0.1824$ while pair 2 has weight $1 - \sigma(-0.5) = 0.6225$. The misranked pair pushes more than three times as hard on the parameters, steering the model to raise $r^{w}_2$ and lower $r^{l}_2$ until that margin flips positive on the next pass.</span>

---

## <span style="font-size: 16px;">Variants and Modern Context</span>

* <span style="font-size: 14px;">**Margin variants:** Llama 2 (Touvron et al., 2023) added a margin term $-\log\sigma(r^{w} - r^{l} - m(\text{rating}))$ where $m$ grows with how strongly the human preferred one response, sharpening separation on clear-cut pairs.</span>
* <span style="font-size: 14px;">**DPO connection:** DPO (Rafailov et al., 2023) keeps this exact Bradley-Terry loss but substitutes an implicit reward defined by the policy and reference log-probability ratio, removing the separate reward model entirely.</span>
* <span style="font-size: 14px;">**Reward over-optimization:** because the reward model is an imperfect proxy, pushing the policy too hard against it causes reward hacking, motivating the KL penalty in the downstream PPO stage.</span>
* <span style="font-size: 14px;">**Ensembles and uncertainty:** later work trains ensembles of reward models and penalizes high disagreement, treating reward variance as a signal that the policy has wandered into a region the reward model cannot judge reliably.</span>

---

## <span style="font-size: 16px;">Connections to Real Systems</span>

<span style="font-size: 14px;">The Bradley-Terry reward loss is the workhorse behind essentially every production RLHF system. InstructGPT and ChatGPT use it directly. Anthropic's Constitutional AI replaces human comparisons with AI-generated preference labels but keeps the same logistic preference loss. Llama 2-Chat trained two separate reward models, one for helpfulness and one for safety, each with this objective, then combined their scores during RL.</span>

<span style="font-size: 14px;">A practical lesson from these systems is that reward-model accuracy on held-out preference pairs is the single most predictive offline metric for downstream RLHF quality. If the reward model cannot separate chosen from rejected on validation data, no amount of RL fine-tuning will produce a good policy, because the policy can only optimize the signal the reward model provides.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Chaining log after sigmoid.** Computing `log(sigmoid(m))` overflows to $-\infty$ for strongly negative margins. Always use the stable $-\text{softplus}(-m)$ form so large or small margins stay finite.</span>
* <span style="font-size: 14px;">**Swapping chosen and rejected.** The label is always 1 for the chosen response. Feeding $r^{l} - r^{w}$ by accident trains the model to prefer rejected responses, and the loss will hover near or above $0.6931$ without improving.</span>
* <span style="font-size: 14px;">**Forgetting that only the difference matters.** The reward is identified only up to an additive constant, so any per-prompt bias the model adds to both responses cancels out. Trying to interpret absolute reward magnitudes across prompts is meaningless.</span>
* <span style="font-size: 14px;">**Averaging the wrong axis.** The loss averages over comparison pairs. Mixing per-token and per-pair reductions, or summing instead of averaging, changes the scale of the gradient and silently breaks the learning rate that was tuned for the mean form.</span>

---