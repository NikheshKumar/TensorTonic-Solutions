## <span style="font-size: 20px;">Gradient Descent Fundamentals</span>

At the heart of neural network training lies gradient descent: minimize a loss function $L(\theta)$ by repeatedly stepping in the direction that decreases the loss most rapidly. The gradient $\nabla_\theta L(\theta)$ points in the direction of steepest ascent, so we move opposite:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

where $\eta$ is the learning rate. This update rule is simple, but its behavior in practice is nuanced. The loss landscape of a neural network is a high-dimensional surface with valleys, ridges, saddle points, and flat regions. Vanilla gradient descent treats every step independently with no memory of past gradients, leading to key pathologies:

- In a narrow valley (steep in one direction, shallow in another), the optimizer oscillates across steep walls while making slow progress along the floor
- Increasing the learning rate to speed floor-progress amplifies oscillations; decreasing it calms oscillations but slows convergence further
- This tension between directions with different curvatures is fundamental to understanding why momentum was invented

In stochastic gradient descent (SGD), the problem is compounded by noise. Each gradient estimate $g_t = \nabla_\theta L_{B_t}(\theta_t)$ from a mini-batch is a noisy approximation of the true gradient. The noise causes the trajectory to jitter, making convergence slower and less stable. The key insight behind momentum: the noise tends to cancel over time (approximately zero-mean), while the consistent descent direction accumulates.

## Momentum: The Physics Analogy

Imagine a heavy ball rolling down a hilly landscape:

- **No mass (vanilla SGD)**: the ball has no inertia - it changes direction instantly with the slope, oscillating in narrow valleys
- **With mass (momentum)**: the ball builds velocity over time; inertia carries it forward along the valley floor rather than bouncing between walls
- Sideways gradient components push the velocity left and right on alternating steps, canceling in the velocity, while forward components accumulate
- The result: faster progress along the consistent descent direction and damped oscillation in inconsistent directions

The friction in this analogy corresponds to the momentum coefficient $\mu$:

- With $\mu = 0$, there is no inertia (vanilla SGD)
- As $\mu$ approaches 1, the ball becomes heavier and velocity changes more slowly
- Typical values like $\mu = 0.9$ mean velocity retains 90% of its previous value each step

There is a terminal velocity in this system. If the gradient is constant at $g$ for many steps, velocity converges to $v = g / (1 - \mu)$. With $\mu = 0.9$, the effective step is 10 times larger than the raw gradient step. This acceleration is crucial: momentum allows the optimizer to move much faster through regions of consistent gradient without increasing the learning rate, which would destabilize the oscillatory directions.

## Mathematical Derivation of Momentum

Momentum maintains a velocity vector $v_t$ that is an exponential moving average of gradients:

$$
v_t = \mu \cdot v_{t-1} + g_t
$$
$$
\theta_t = \theta_{t-1} - \eta \cdot v_t
$$

where $v_0 = 0$, $g_t = \nabla_\theta L(\theta_{t-1})$ is the gradient at step $t$, $\mu$ is the momentum coefficient, and $\eta$ is the learning rate. Unrolling the recurrence reveals a weighted sum of all past gradients:

$$
v_t = g_t + \mu g_{t-1} + \mu^2 g_{t-2} + \cdots + \mu^{t-1} g_1
$$

- Each past gradient $g_k$ contributes with weight $\mu^{t-k}$, decaying exponentially
- The effective window of influence is approximately $1/(1-\mu)$ steps: for $\mu = 0.9$ the last 10 gradients dominate; for $\mu = 0.99$ the last 100
- Consistent signal is amplified while noise variance is reduced by approximately $(1-\mu)/(1+\mu)$

## Why This is Not Quite a Standard EMA

A standard EMA is $\bar{g}_t = \mu \bar{g}_{t-1} + (1-\mu) g_t$, keeping the average's scale the same as individual values. PyTorch's formulation omits the $(1-\mu)$ factor, so velocity grows by $1/(1-\mu)$ relative to raw gradients. This rescaling is absorbed into the learning rate: when you add momentum, you may need to reduce the learning rate to compensate for the amplification.

## Convergence Properties

For a quadratic loss $L(\theta) = \frac{1}{2} \theta^T A \theta - b^T \theta$ with positive definite Hessian $A$ and condition number $\kappa = \lambda_{\max} / \lambda_{\min}$:

- Vanilla gradient descent converges at rate $(\kappa - 1)/(\kappa + 1)$ - approaches 1 for ill-conditioned problems
- Momentum improves this to $(\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ - dramatically better for large $\kappa$
- This is the optimal rate achievable by any first-order method for quadratic objectives
- The optimal momentum coefficient:

$$
\mu^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2
$$

For neural networks the loss surface is not quadratic and these guarantees do not apply directly. However, the intuition transfers: momentum helps most when the loss surface has directions of very different curvature, which is almost always the case in deep learning.

## Dampening

PyTorch's SGD includes an optional dampening parameter $\tau$:

$$
v_t = \mu \cdot v_{t-1} + (1 - \tau) \cdot g_t
$$

- Default $\tau = 0$ gives standard momentum
- $\tau > 0$ reduces the current gradient's contribution, placing more weight on accumulated history
- Useful for smoothing the trajectory more aggressively
- Rarely used standalone - its effect overlaps with adjusting momentum or learning rate
- Important for matching PyTorch's built-in SGD behavior, since dampening appears in the API and interacts with Nesterov momentum

## Nesterov Momentum: The Lookahead Gradient

The key idea: instead of computing the gradient at the current position, first step in the direction of the current velocity (a "lookahead"), then compute the gradient there:

$$
\theta_{\text{lookahead}} = \theta_{t-1} - \eta \cdot \mu \cdot v_{t-1}
$$
$$
v_t = \mu \cdot v_{t-1} + \nabla_\theta L(\theta_{\text{lookahead}})
$$
$$
\theta_t = \theta_{t-1} - \eta \cdot v_t
$$

The momentum term will be applied regardless, so we evaluate the gradient at where we are going to end up. If momentum carries us too far (past the minimum), the gradient at the lookahead point points back, providing a corrective signal. This gives Nesterov momentum a "look before you leap" quality.

In PyTorch's implementation, the update is reformulated to avoid an extra forward pass:

$$
\theta_t = \theta_{t-1} - \eta \cdot (\mu \cdot v_t + g_t)
$$

The difference from classical momentum: in the update, $\mu v_t + g_t$ replaces just $v_t$, adding an extra nudge in the current gradient direction. For convex optimization, Nesterov provably achieves the optimal convergence rate. In deep learning, benefits are less clear-cut but often provide modest improvements near convergence.

## Weight Decay vs L2 Regularization

These are often treated as equivalent, but they differ when momentum is involved:

- **L2 regularization**: adds $\frac{\lambda}{2} ||\theta||^2$ to the loss, so the gradient becomes $g_t + \lambda \theta_t$ - this modified gradient flows through the entire optimizer including momentum buffers
- **Weight decay**: shrinks parameters directly by $(1 - \eta \lambda)$, independent of gradient computation:

$$
\theta_t = (1 - \eta \lambda) \theta_{t-1} - \eta \cdot v_t
$$

For vanilla SGD without momentum, these are equivalent. With momentum, L2 regularization accumulates regularization forces in the velocity buffer, creating complex interactions between regularization strength and momentum dynamics. Decoupled weight decay applies the same relative shrinkage regardless of gradient history.

The paper "Decoupled Weight Decay Regularization" showed that L2 regularization through the gradient gets scaled by adaptive learning rates, meaning different parameters receive different effective regularization. Decoupled weight decay is the intended behavior, leading to the AdamW optimizer.

## The Optimizer Base Class Internals

PyTorch's Optimizer base class provides the scaffolding that all optimizers build upon. Understanding its internal data structures is essential for implementing custom optimizers correctly.

## Constructor and Defaults

- Takes two arguments: params (an iterable of parameters or parameter groups) and defaults (a dictionary of default hyperparameter values)
- Processes params into a list of parameter group dictionaries, each containing a "params" key with parameter tensors and keys for each hyperparameter
- If params is a simple iterable of tensors, creates a single parameter group
- If params is a list of dictionaries (each with a "params" key and optional hyperparameter overrides), creates one group per dictionary, filling in missing hyperparameters from defaults

## The param_groups Structure

A list of dictionaries enabling per-group hyperparameter overrides. This design is one of the most important practical features of the optimizer API:

- Different learning rates for different model parts (backbone vs head in transfer learning) - use a smaller rate for pretrained layers, larger for new layers
- Selective weight decay: apply weight decay to weight matrices but not to bias terms or batch normalization parameters by placing them in separate groups
- Learning rate schedulers interact with parameter groups by modifying the "lr" value stored in each group dictionary - when the scheduler steps, it updates each group's learning rate
- The optimizer reads the learning rate from the group dictionary on each step rather than storing it as an instance variable, enabling this clean separation of concerns

## The state Dictionary

- A defaultdict(dict) keyed by parameter tensor identity (the Python object, not a copy)
- Stores the optimizer's internal state for each parameter: for SGD with momentum, this includes the velocity buffer; for Adam, both moment estimates and step counter
- Lazily initialized: rather than allocating all buffers up front, the optimizer creates them the first time each parameter's gradient is processed
- This is both memory-efficient (parameters that never receive gradients do not get state) and robust (works correctly even if the parameter set changes between steps)
- The lazy initialization pattern checks whether a key exists in the state dict and creates the buffer with zeros_like if not

## State Serialization

- state_dict() and load_state_dict() handle saving and restoring the optimizer's complete state: all per-parameter state buffers and all hyperparameters
- Essential for resuming training from checkpoints
- The state dict maps parameter indices (not tensors themselves, which would not survive serialization) to their state dictionaries
- When loading, parameters are matched by index within each group, so model architecture must match exactly

## Comparison with Adaptive Methods

SGD with momentum uses the same learning rate for every parameter. Adaptive methods adjust per-parameter based on gradient history:

- **AdaGrad**: accumulates sum of squared gradients $s_t = s_{t-1} + g_t^2$; update divides by $\sqrt{s_t} + \epsilon$; useful for sparse features but learning rate decays monotonically to zero
- **RMSProp**: uses exponential moving average of squared gradients instead of cumulative sum, allowing the effective learning rate to recover
- **Adam**: combines momentum (first moment EMA) with adaptive scaling (second moment EMA):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
$$
s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t^2
$$

Bias-corrected to account for zero initialization:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
$$

Update:

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{s}_t} + \epsilon}
$$

Adam is "momentum in a normalized gradient space": the first moment provides directional signal, the second moment normalizes each parameter's gradient by its typical magnitude. Adam is less sensitive to learning rate choice and often converges faster early. However, adaptive rates can be too aggressive in some directions, leading to worse generalization compared to SGD with momentum, particularly in image classification.

## Learning Rate Scheduling

The learning rate is the most important hyperparameter, and keeping it fixed is rarely optimal:

- **Step decay**: reduce by a fixed factor at predetermined epochs (e.g., multiply by 0.1 every 30 epochs)
- **Cosine annealing**: decrease following a cosine curve from initial value to near zero
- **Warm-up**: start very small, linearly increase to target over the first few epochs - stabilizes early training, especially with large batch sizes
- **One-cycle policy**: warm-up plus cosine decay in a single cycle - enables larger learning rates and "super-convergence"

Schedulers operate on the optimizer's param_groups, modifying the "lr" value directly. The optimizer reads the learning rate from the group dictionary on each step.

## Gradient Clipping Interaction

Gradient clipping prevents gradient explosions (common in RNNs and transformers). Two forms exist:

- **Gradient norm clipping**: if the total norm of all gradients exceeds a threshold, scale all gradients down proportionally - preserves gradient direction while limiting magnitude
- **Gradient value clipping**: clamp each gradient element independently to a fixed range - simpler but can change gradient direction

Gradient clipping is applied after backward but before the optimizer step. The interaction with momentum matters: when gradients are clipped, the clipped gradient (not the original) enters the velocity buffer. This can smooth the velocity, but overly aggressive clipping prevents the velocity from building up, effectively defeating momentum. Finding the right threshold balances stability against convergence speed.

## The zero_grad() Method

Before each backward pass, gradients from the previous step must be cleared. By default, PyTorch accumulates gradients: calling backward adds to existing .grad tensors rather than replacing them. The zero_grad method sets all parameter gradients to zero (or, with set_to_none=True, sets them to None, which is slightly more memory-efficient and faster because it avoids a memset operation). This method is inherited from the base class and does not need custom implementation.

## The step() Method in Detail

The step method is where the actual parameter update happens. A well-structured step follows this pattern:

- Handle the optional closure (a callable that re-evaluates the loss, needed for algorithms like L-BFGS that require multiple function evaluations per step; for SGD and most optimizers, rarely used but supporting it is part of the API contract)
- Iterate over parameter groups and parameters
- For each parameter with a gradient: read the gradient from the .data attribute (to avoid triggering autograd tracking), look up or initialize state, apply the update rule, modify parameter data in place
- Skip parameters where grad is None (unused in the computation graph or frozen via requires_grad=False)
- All updates use in-place operations (mul_, add_) - creating new tensors would break the connection between the parameter tensor and the model, since the model holds a reference to the original tensor object; in-place ensures the model sees changes immediately
- Read hyperparameters from the current group dictionary (not self.defaults) to respect per-group overrides

## Practical Guidelines for Custom Optimizers

- **Match reference implementations**: test against built-in optimizers with identical hyperparameters and initial conditions; even small numerical differences accumulate over many steps
- **Use in-place operations**: the model holds references to original tensor objects; in-place ensures the model sees changes immediately
- **Test with multiple parameter groups**: verify per-group hyperparameters are respected; a common bug is reading from self.defaults instead of the current group
- **Test state serialization**: save and load optimizer state dict, verify training continues identically
- **Key state by tensor identity**: use self.state[p] where p is from group["params"], not a copy
- **Numerical stability**: add epsilon when dividing by running averages; large momentum near 1 can cause velocity overflow in float16
- **Ecosystem compatibility**: the optimizer interacts with LR schedulers, gradient clipping, gradient scaling for mixed precision, and model parallelism - following the standard API ensures seamless integration
