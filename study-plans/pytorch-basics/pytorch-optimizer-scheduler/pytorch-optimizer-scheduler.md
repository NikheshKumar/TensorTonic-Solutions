## <span style="font-size: 20px;">Why Learning Rate Scheduling Matters</span>

The learning rate is the single most important hyperparameter in neural network training. It controls the magnitude of parameter updates at each step, and its value determines whether training converges, diverges, or stalls. A fixed learning rate forces a fundamental tradeoff: large values enable fast initial progress but cause oscillation near minima, while small values allow precise convergence but waste computation in the early phase when the model is far from any reasonable solution.

Learning rate scheduling resolves this tension by varying the rate over training. The core idea: use a larger rate when big imprecise steps are affordable, and a smaller rate when fine-grained adjustments are needed.

## Constant Learning Rate Limitations

The gradient descent update rule is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

With constant $\eta$, several pathologies arise:

- **Oscillation in narrow valleys**: When the loss surface has high curvature in one direction and low curvature in another, a constant rate causes bouncing across valley walls while making slow progress along the floor. The rate must be small enough for the high-curvature direction, making progress painfully slow in the low-curvature direction. This is the condition number problem
- **Overshooting near convergence**: As the optimizer approaches a minimum, stochastic gradient noise from mini-batching persists. With constant $\eta$ and gradient noise variance $\sigma^2$, the iterates converge to a neighborhood of size proportional to $\eta \sigma^2$ rather than the minimum itself
- **Theoretical convergence requirement**: For SGD to converge to a minimum (not just a neighborhood), the Robbins-Monro conditions require:

$$
\sum_{t=1}^{\infty} \eta_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

A constant rate satisfies the first condition (ensuring the optimizer can reach any point) but violates the second (which ensures noise diminishes). Any schedule where $\eta_t \to 0$ at a rate slower than $1/t$ satisfies both conditions.

## StepLR: Piecewise Constant Decay

StepLR keeps the learning rate constant for a fixed number of epochs, then multiplies by a decay factor. Given initial rate $\eta_0$, step size $s$, and factor $\gamma \in (0, 1)$:

$$
\eta_e = \eta_0 \cdot \gamma^{\lfloor e / s \rfloor}
$$

After $s$ epochs, the rate is multiplied by $\gamma$ once. After $2s$ epochs, it has been multiplied twice, giving $\eta_0 \gamma^2$. In general, the number of decay events after $e$ epochs is $\lfloor e/s \rfloor$.

- Example: $\eta_0 = 0.1$, $s = 30$, $\gamma = 0.1$
  - Epochs 0-29: $\eta = 0.1$
  - Epochs 30-59: $\eta = 0.01$
  - Epochs 60-89: $\eta = 0.001$
- This schedule was the default for training CNNs on ImageNet for many years
- Main advantage: simplicity and predictability
- Main drawback: abrupt rate changes at step boundaries can cause temporary instability

## MultiStepLR

A generalization that allows specifying arbitrary milestones at which the rate is multiplied by $\gamma$. Useful when the optimal schedule is known empirically but does not follow a regular pattern. For example, decaying at epochs 30, 60, and 80 out of 90 total was a common recipe for ResNet training.

## Exponential Decay

ExponentialLR applies multiplicative decay at every epoch:

$$
\eta_e = \eta_0 \cdot \gamma^e
$$

- Equivalent to StepLR with $s = 1$; smooth curve avoids abrupt transitions
- Can reduce the rate too aggressively. For $\gamma = 0.95$ and $e = 100$, the final rate is about 0.6% of the original. For $\gamma = 0.9$, it drops to about 0.003% after 100 epochs
- Choosing $\gamma$ requires calibration: too close to 1 gives almost no decay, too small causes premature convergence
- More common in older training recipes; modern practice prefers cosine annealing or OneCycleLR

## Cosine Annealing

Cosine annealing has become one of the most popular learning rate schedules. The rate follows a cosine curve from $\eta_{\max}$ to $\eta_{\min}$ over $T$ epochs:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

## Intuition Behind the Cosine Shape

- At $t = 0$: $\cos(0) = 1$, so $\eta_0 = \eta_{\max}$. At $t = T$: $\cos(\pi) = -1$, so $\eta_T = \eta_{\min}$
- The rate of decay is non-uniform: slow at the beginning (cosine near its peak, nearly flat), accelerating through the middle, and slowing again near the end (cosine approaching minimum, flattening)
- This matches training dynamics well. In the early phase, the model benefits from a high rate for rapid progress. In the middle, the rate decreases steadily as the optimizer navigates toward good regions. In the final phase, gentle annealing helps settle into a precise minimum
- Has few hyperparameters and performs well across a wide range of architectures and datasets

## Cosine Annealing with Warm Restarts (SGDR)

The cosine schedule is repeated multiple times. After each cycle, the rate resets to $\eta_{\max}$ and decay begins again. Restarts help escape poor local minima by suddenly increasing the rate. Each cycle can be lengthened (e.g., period doubling via $T_{\text{mult}}$), allowing longer refinement in later cycles.

## Learning Rate Warmup

Warmup starts training with a very small rate and gradually increases to the target value over $W$ steps. During linear warmup:

$$
\eta_t = \eta_{\text{target}} \cdot \frac{t}{W} \quad \text{for } t \leq W
$$

After warmup ($t > W$), the rate transitions to the subsequent decay schedule (cosine, linear, constant, etc.). This technique is critical for training transformers and large models.

## Why Transformers Need Warmup

The need stems from the interaction between the Adam optimizer and initial gradient statistics:

- At training start, Adam's second moment estimates are initialized to zero. The bias correction compensates, but corrected estimates can be highly inaccurate in the first few steps, leading to excessively large or erratic updates
- Transformers are particularly sensitive because self-attention computes softmax over dot products of query and key vectors. In the first steps, these dot products are random, and large rates push softmax into saturation (nearly all weight on one position), creating sparse gradients that destabilize training
- A warmup period allows the model to calibrate attention patterns with small, stable updates before ramping to full rate

The original "Attention Is All You Need" paper used:

$$
\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5}, \, t \cdot t_{\text{warmup}}^{-1.5})
$$

This increases linearly for the first $t_{\text{warmup}}$ steps, then decays as $t^{-0.5}$. Modern practice typically uses simple linear warmup followed by cosine decay.

## Cyclical Learning Rates

Cyclical learning rates (CLR) oscillate the rate between minimum and maximum bounds. Periodically increasing the rate helps escape saddle points and sharp minima. Three main policies:

- **Triangular**: Linear increase from $\eta_{\min}$ to $\eta_{\max}$ over half a cycle, linear decrease over the other half
- **Triangular2**: Same but maximum rate halved after each cycle, so oscillations diminish
- **Exp_range**: Maximum bound decays exponentially, providing smoother diminishment

CLR reduces the need to tune the rate precisely. By sweeping through a range of rates, the optimizer naturally spends time at whichever rate is most useful for the current phase.

## OneCycleLR

OneCycleLR uses a single cycle over the entire training run:

- **Phase 1 (warmup, ~30% of total steps)**: Rate increases from $\eta_{\min}$ to $\eta_{\max}$. Serves as aggressive warmup
- **Phase 2 (annealing)**: Rate decreases from $\eta_{\max}$ to a value much lower than $\eta_{\min}$ (often $\eta_{\max} / 10000$), typically via cosine schedule
- Additionally modulates the optimizer's momentum: lower momentum during the high-rate phase (prevents overshooting), higher momentum during the low-rate phase (maintains progress)
- The aggressive warmup acts as a regularizer: large rates prevent settling into sharp minima and push toward flatter regions of the loss landscape that generalize better
- Can match standard accuracy in significantly fewer epochs than traditional schedules

## ReduceLROnPlateau: Adaptive Scheduling

Unlike predetermined schedules, ReduceLROnPlateau adapts based on actual training dynamics. It monitors a metric (typically validation loss) and reduces the rate when it stops improving.

The scheduler maintains a patience counter. If the metric does not improve for $p$ consecutive epochs, the rate is multiplied by $\gamma$ (typically 0.1). Key parameters:

- **Mode**: Monitor for minimum (loss) or maximum (accuracy)
- **Patience**: Epochs without improvement before reducing
- **Factor**: Multiplicative reduction ($\gamma$, default 0.1)
- **Threshold**: Minimum change to qualify as improvement
- **Cooldown**: Epochs to wait after reduction before resuming monitoring
- **Minimum rate**: Lower bound on the learning rate

Unlike other schedulers, requires the metric value passed to the step call. Useful when the optimal schedule is unknown or training dynamics are unpredictable.

## The Learning Rate Finder

Systematic method to determine the right rate range before choosing a schedule:

- Start with a very small rate (e.g., $10^{-7}$) and increase exponentially over one epoch:

$$
\eta_i = \eta_{\text{start}} \cdot \left(\frac{\eta_{\text{end}}}{\eta_{\text{start}}}\right)^{i/N}
$$

- Plot loss vs. rate (log scale). The curve typically shows three regions:
  - Too low: loss decreases very slowly or remains flat
  - Good range: loss decreases rapidly
  - Too high: loss explodes or starts increasing
- Set $\eta_{\max}$ at the steepest downward slope, before the loss starts increasing
- For OneCycleLR, $\eta_{\min}$ is one-tenth to one-twentieth of $\eta_{\max}$

## Batch Size and Learning Rate: The Linear Scaling Rule

When batch size increases by factor $k$, the learning rate should also increase by approximately $k$. The gradient estimate for a batch is:

$$
g_B = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}(x_i, \theta)
$$

Taking $k$ consecutive steps with batch size $B$ and rate $\eta$ produces total parameter change $\approx -k\eta \cdot g_B$. A single step with batch size $kB$ and rate $k\eta$ gives the same result since $g_{kB} \approx g_B$.

- Validated empirically by Goyal et al. (2017) up to batch size 8192 with gradual warmup
- Breaks down at extreme batch sizes (beyond ~8K-32K): gradient estimates become so accurate that the exploration-aiding noise is lost, leading to sharper minima and worse generalization
- Techniques like LARS and LAMB handle extremely large batch training

## Interaction with Adaptive Optimizers

Adam maintains two exponential moving averages per parameter: first moment $m_t$ (mean of gradients) and second moment $v_t$ (mean of squared gradients). The effective update is:

$$
\theta_i \leftarrow \theta_i - \eta \cdot \frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}} + \epsilon}
$$

- The denominator acts as per-parameter scaling: parameters with large gradients get smaller effective rates, and vice versa. Adam already performs "scheduling" at the parameter level
- However, the global $\eta$ still acts as a master scaling factor for all updates
- Scheduling $\eta$ still helps: it reduces overall update noise and allows tighter convergence
- Benefits are less dramatic than with plain SGD, but standard practice for large-scale training. The typical recipe for transformers: AdamW with linear warmup for 1-10% of steps, followed by cosine decay

## AdamW and Decoupled Weight Decay

When using scheduling with weight decay, the implementation matters:

- In standard Adam, L2 regularization adds $\lambda \theta$ to the gradient, which gets scaled by Adam's adaptive denominator. The effective decay rate varies per parameter, which is usually undesirable
- AdamW applies weight decay directly to parameters after the Adam update, independent of adaptive scaling
- When combining a schedule with weight decay, AdamW is preferred because the decay rate remains consistent regardless of the learning rate

## Polynomial and Linear Decay

Linear decay from $\eta_0$ to $\eta_{\min}$ (often 0) over $T$ steps:

$$
\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T}\right)
$$

This is a special case of polynomial decay with exponent 1. The general formula:

$$
\eta_t = (\eta_0 - \eta_{\min}) \cdot \left(1 - \frac{t}{T}\right)^p + \eta_{\min}
$$

- $p = 1$: linear
- $p = 2$: quadratic (faster initial decay, slower near end)
- $p = 0.5$: square root (slower initial decay, faster near end)

## Scheduler Chaining

Multiple schedulers can be combined sequentially. The standard way to implement warmup + decay is to use a sequential scheduler that switches from a warmup scheduler to a decay scheduler at a specified milestone epoch. A chained scheduler applies multiple schedulers simultaneously, with each multiplying its factor into the rate.

## Practical Guidelines

The choice of schedule depends on the model, dataset, optimizer, and training budget:

- **CNNs**: Cosine annealing or StepLR with SGD + momentum. MultiStepLR with milestones at 30%, 60%, 80% of total epochs (factor 0.1) is a reliable baseline. OneCycleLR can match accuracy in fewer epochs if the max rate is well-chosen
- **Transformers**: AdamW with linear warmup (1-10% of total steps) followed by cosine or linear decay. Peak rates $10^{-4}$ to $5 \times 10^{-4}$ for pre-training, $10^{-5}$ to $5 \times 10^{-5}$ for fine-tuning
- **Fine-tuning**: Small constant rate or gentle decay, 10x-100x smaller than original training rate. Cosine annealing with short warmup works well. Some use discriminative rates (smaller for early layers, larger for later layers)
- **Default**: Cosine annealing is the safest general choice. Few hyperparameters, smooth curves, strong performance across architectures. With short warmup (5-10% of training), it is a strong baseline

## Monitoring Learning Rates

Always log the learning rate during training. The current rate for each parameter group is available in the optimizer's parameter groups. Plotting the rate alongside training loss is invaluable for diagnosing issues. A sudden spike in loss often corresponds to a rate that is too high, while a plateau may indicate the rate should be reduced further.

## Common Mistakes

- **Wrong calling frequency**: StepLR, ExponentialLR, CosineAnnealingLR are per-epoch. OneCycleLR and CyclicLR are per-batch. Calling an epoch-level scheduler per batch compresses the entire schedule into one epoch
- **Wrong ordering**: The optimizer must update parameters before the scheduler updates the rate. Calling them in the wrong order means the first batch uses previous parameters but next epoch's rate
- **Missing metric for ReduceLROnPlateau**: Unlike other schedulers, this one requires the metric value in the step call. Calling without it raises an error or silently does nothing
- **Constructor's implicit step**: The scheduler constructor calls step internally. The counter starts at 0 after construction. Calling step again before the first epoch skips the initial rate entirely
- **Not saving scheduler state**: When resuming from a checkpoint, the scheduler's state dictionary must be saved and loaded alongside model and optimizer state. Otherwise it restarts from epoch 0, producing incorrect rates

## Summary of Key Formulas

**StepLR:**
$$\eta_e = \eta_0 \cdot \gamma^{\lfloor e/s \rfloor}$$

**ExponentialLR:**
$$\eta_e = \eta_0 \cdot \gamma^e$$

**CosineAnnealingLR:**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$$

**Linear warmup:**
$$\eta_t = \eta_{\text{target}} \cdot t / W \quad \text{for } t \leq W$$

**Polynomial decay:**
$$\eta_t = (\eta_0 - \eta_{\min})(1 - t/T)^p + \eta_{\min}$$

**Linear scaling rule:**
$$\eta_{kB} = k \cdot \eta_B$$
