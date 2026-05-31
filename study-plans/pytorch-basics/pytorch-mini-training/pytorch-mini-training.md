## <span style="font-size: 20px;">The Optimization Objective</span>

Training a neural network is a numerical optimization problem. Given a model $f_\theta$ with parameters $\theta$, a dataset $\{(x_i, y_i)\}_{i=1}^{N}$, and a loss function $\ell$ that measures the discrepancy between predictions and ground truth, the goal is to find parameter values that minimize the empirical risk:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell(f_\theta(x_i),\; y_i)
$$

This objective connects "learning" to concrete computation:

- The forward pass evaluates $f_\theta$
- The loss function computes $\ell$
- The backward pass computes $\nabla_\theta \mathcal{L}$
- The optimizer uses that gradient to update $\theta$

The choice of loss function shapes the loss landscape. For regression, mean squared error produces a smooth, convex surface for linear models. For classification, cross-entropy loss penalizes confident wrong predictions heavily, creating steep gradients that accelerate early learning. The loss function is not just a metric: it is the signal that guides every parameter update.

## The Loss Landscape

Think of the loss function as defining a surface over the space of all possible parameter values. For a model with $d$ parameters, this surface lives in $d+1$ dimensions: $d$ axes for the parameters and one axis for the loss value. Training is the process of navigating downhill on this surface.

- For simple models (linear regression with MSE loss), this surface is a convex paraboloid with a single global minimum. Gradient descent is guaranteed to find it
- For neural networks, the surface is highly non-convex, riddled with local minima, saddle points, and flat plateaus
- Saddle points are especially common in high-dimensional spaces: a point where the gradient is zero but the Hessian has both positive and negative eigenvalues. For large networks, most local minima have loss values close to the global minimum, so getting stuck in a "bad" local minimum is less of a concern than classical theory would suggest
- Flat regions (plateaus) can be more problematic in practice than local minima. When the gradient magnitude is near zero, parameter updates become tiny and training appears to stall. This is one reason adaptive optimizers like Adam often outperform vanilla SGD on complex loss surfaces
- The curvature of the loss surface matters enormously. In directions of high curvature, loss changes rapidly; in directions of low curvature, it changes slowly. The condition number of the Hessian $H = \nabla^2 \mathcal{L}$, the ratio of its largest to smallest eigenvalue, quantifies this mismatch. Ill-conditioned problems are harder to optimize with simple gradient descent

## Anatomy of a Training Step

Each mini-batch update follows a strict five-step sequence. Getting this order wrong produces subtle, silent bugs.

## Clearing Old Gradients

PyTorch accumulates gradients by default. When backward is called, computed gradients are added to whatever is already in each parameter's gradient attribute rather than replacing it. This design supports gradient accumulation across multiple mini-batches (useful when a single batch does not fit in memory), but it means you must explicitly zero out gradients at the start of each iteration. Setting gradients to None instead of zero is slightly more memory-efficient.

## The Forward Pass

The forward pass evaluates $\hat{y} = f_\theta(X_b)$ for the current batch. This simultaneously constructs the computational graph that autograd will later traverse. Every tensor operation involving a parameter with gradient tracking creates a node in a directed acyclic graph (DAG). The graph records what function was applied, which tensors were inputs, and enough information to compute the local Jacobian during the backward pass.

## Computing the Loss

The loss function collapses the batch of predictions and targets into a single scalar. This scalar is the root of the computational graph from which backpropagation begins. The reduction method matters: using mean reduction divides by the number of elements, producing a loss whose scale is independent of batch size. Using sum reduction instead would make the effective learning rate proportional to batch size.

## The Backward Pass

Calling backward triggers reverse-mode automatic differentiation. Starting from the loss scalar (with an implicit gradient of 1.0), PyTorch walks backward through the computational graph, applying the chain rule at each node to propagate gradients to every leaf parameter. After this call, every parameter tensor $\theta_k$ has its gradient populated with $\frac{\partial \mathcal{L}}{\partial \theta_k}$.

## The Optimizer Step

The optimizer uses the computed gradients to update parameters. For vanilla SGD: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. For Adam, it involves running averages of the first and second moments of the gradients with bias-corrected adaptive learning rates. The optimizer never touches the computational graph; it operates purely on parameter tensors and their gradients.

## Computational Graphs and Autograd Mechanics

The computational graph makes neural network training tractable. Without automatic differentiation, you would need to derive and implement the gradient of every architecture by hand.

PyTorch uses a dynamic computational graph (define-by-run). The graph is constructed anew during each forward pass, which means the graph structure can change from iteration to iteration. This allows control flow (conditionals, variable-length loops, recursion) inside models: the graph simply records whatever operations actually executed.

Each node in the graph stores a reference to a gradient function: the function that computes the local gradient during the backward pass. For example, a matrix multiplication node knows how to compute the gradient of $C = AB$ with respect to both $A$ and $B$. A ReLU node knows that the gradient is 1 where the input is positive and 0 where it is negative.

Consider a simple two-layer network:

$$
h = W_1 x + b_1
$$

$$
a = \sigma(h)
$$

$$
\hat{y} = W_2 a + b_2
$$

$$
\mathcal{L} = \ell(\hat{y}, y)
$$

During the backward pass, the chain rule unrolls as:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot a^T
$$

$$
\frac{\partial \mathcal{L}}{\partial a} = W_2^T \cdot \frac{\partial \mathcal{L}}{\partial \hat{y}}
$$

$$
\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial a} \odot \sigma'(h)
$$

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial h} \cdot x^T
$$

Each node only needs to know its own local derivative; the chain rule handles composition automatically. After backward completes, the graph is destroyed by default, freeing substantial memory. The graph holds references to all intermediate tensors, and releasing it allows garbage collection.

## Gradient Computation and Backpropagation

Backpropagation is the chain rule applied systematically via reverse-mode automatic differentiation. The term "reverse-mode" refers to the direction of traversal: gradients flow from the output backward through the graph to the parameters. This is efficient because it computes the gradient of one scalar output with respect to many parameters in a single pass. Forward-mode would require one pass per parameter, which is prohibitively expensive for models with millions of parameters.

- The computational cost of one backward pass is roughly 2-3 times that of the forward pass
- Gradients are defined with respect to the scalar loss. You cannot call backward on a non-scalar tensor without providing a gradient argument. This is why the loss function reduces predictions to a single number
- Autograd computes exact derivatives (up to floating-point precision): no finite-difference approximation is involved

## Mini-Batch SGD vs Full-Batch vs Pure Stochastic

The three main variants of gradient descent differ in how many samples estimate the gradient:

## Full-Batch Gradient Descent

Uses the entire dataset:

$$
\theta \leftarrow \theta - \eta \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

- Gives the exact gradient of the empirical risk; the trajectory is smooth and deterministic
- Impractical for large datasets: entire dataset must fit in memory, one gradient computation scales as $O(N)$
- Tends to converge to sharp minima that generalize poorly

## Pure Stochastic Gradient Descent (batch size 1)

Uses a single randomly sampled example:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

- Unbiased but high-variance gradient estimate
- Noisy trajectory can help escape shallow local minima but makes convergence erratic
- Fails to exploit GPU parallelism, making it computationally wasteful

## Mini-Batch SGD

The practical compromise using $B$ samples:

$$
\theta \leftarrow \theta - \eta \frac{1}{B} \sum_{j=1}^{B} \nabla_\theta \ell(f_\theta(x_j), y_j)
$$

- Inherits statistical benefits of stochastic optimization while being efficient on parallel hardware
- Batch size $B$ controls gradient variance: larger batches give lower-variance estimates but require more computation per step
- The linear scaling rule (Goyal et al., 2017): doubling batch size allows proportionally increasing the learning rate to maintain the same effective noise level
- Mini-batch noise is not a nuisance: it serves as an implicit regularizer. Smaller batch sizes tend to find flatter minima that generalize better; very large batch sizes converge to sharper minima with worse generalization

## The Learning Rate and Its Role

The learning rate $\eta$ is the single most important hyperparameter:

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
$$

- Too large: updates overshoot the minimum, loss oscillates or diverges
- Too small: updates are tiny, training takes impractically many steps and may get stuck
- For a convex quadratic with Hessian eigenvalues between $\mu$ and $L$, gradient descent converges if and only if $\eta < 2/L$. The optimal rate is $\eta^* = 2/(\mu + L)$
- In practice, learning rate schedules are common: warmup, step decay, cosine annealing, reduce-on-plateau

Adaptive optimizers (Adam, AdaGrad, RMSProp) adjust the effective learning rate per parameter based on gradient history:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates. Parameters with consistently large gradients get effectively smaller learning rates, while parameters with small gradients get larger effective rates. This addresses the condition number problem without manual per-layer tuning.

## Convergence Properties

For convex objectives with Lipschitz-continuous gradient (smoothness constant $L$), gradient descent with $\eta = 1/L$ achieves:

$$
\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \leq \frac{L \|\theta_0 - \theta^*\|^2}{2T}
$$

This is $O(1/T)$ convergence. If the objective is also strongly convex (minimum Hessian eigenvalue $\mu > 0$), the rate improves to linear (exponential) convergence:

$$
\begin{aligned}
&\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \\
&\quad \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2T} \\
&\quad \phantom{\leq} \left(\mathcal{L}(\theta_0) - \mathcal{L}(\theta^*)\right)
\end{aligned}
$$

where $\kappa = L/\mu$ is the condition number. For SGD with mini-batches and constant learning rate, the iterates converge to a neighborhood of the optimum with size proportional to $\eta \sigma^2$, where $\sigma^2$ is the gradient variance.

For neural networks, convex guarantees do not formally apply, but SGD empirically converges to stationary points that tend to be good enough, especially in overparameterized networks.

## Loss Averaging and Its Statistical Meaning

During training, per-batch loss is accumulated and averaged:

$$
\bar{\mathcal{L}} = \frac{1}{N_{\text{batches}}} \sum_{b=1}^{N_{\text{batches}}} \mathcal{L}_b
$$

- If all batches have equal size $B$ and each loss uses mean reduction, this equals the per-sample mean loss
- If the last batch is smaller, it contributes equal weight to the average despite representing fewer samples. Solutions: drop the incomplete batch, or weight each batch's loss by its size
- The epoch-average loss is computed using parameters at different points in the trajectory. It mixes losses from slightly different models, so it is a running estimate, not the loss at any single parameter configuration. For precise evaluation, a separate validation pass without parameter updates is needed

## The DataLoader and Batching

Key parameters:

- **Batch size**: Affects gradient variance, memory usage, and training dynamics
- **Shuffle**: Critical for SGD convergence. Without shuffling, gradient estimates within each batch are biased. Shuffling ensures each batch roughly represents the full dataset
- **Drop last**: Discard the final incomplete batch. Useful for consistent batch normalization behavior and clean loss averaging
- **Number of workers**: Parallel data loading subprocesses prevent the GPU from starving while waiting for data
- **Pin memory**: Copies tensors into CUDA pinned memory, enabling asynchronous GPU transfers

## Training Mode vs Evaluation Mode

Two layer types behave differently between modes:

- **Dropout**: In training mode, randomly zeros fraction $p$ of activations and scales the rest by $1/(1-p)$, ensuring consistent expected values. In eval mode, all activations pass through unchanged. Leaving train mode on during validation adds noise to the loss estimate
- **Batch normalization**: In training mode, normalizes using batch statistics and updates running averages. In eval mode, uses stored running averages for deterministic predictions. Forgetting eval mode during validation means normalization depends on batch composition rather than learned statistics

Always switch to train mode before each training epoch and eval mode before each validation pass.

## Memory Management During Training

Major memory consumers:

- **Parameters**: Each tensor occupies 4 bytes per element (float32) or 2 bytes (float16)
- **Gradients**: Same size as parameters, roughly doubling memory
- **Optimizer state**: Adam maintains two additional tensors per parameter (first and second moments), tripling effective parameter memory. SGD with momentum maintains one extra tensor
- **Activations**: Intermediate values stored for gradient computation during the backward pass. Scales with both batch size and network depth. Gradient checkpointing trades computation for memory by recomputing activations during backward
- **Graph metadata**: As long as the graph exists, all intermediate tensors are pinned in memory

Critical point: if you accumulate loss tensors without extracting the scalar value, you keep the entire computational graph alive. After many batches, this chains into massive memory consumption. Always extract the Python float value to sever the graph connection.

Disabling gradient computation during validation eliminates both the memory overhead and computational cost of graph construction.

## Common Pitfalls and Their Consequences

- **Forgetting to zero gradients**: Gradients accumulate across batches. The effective gradient at step $t$ becomes $\sum_{i=1}^{t} g_i$, causing the effective learning rate to grow linearly. Loss typically explodes after the first epoch. Completely silent: no error is raised
- **Wrong operation order**: Calling the optimizer step before backward reads zero or stale gradients. The model either does not train or trains at half the expected rate
- **Not extracting scalar loss**: Accumulating tensor losses causes out-of-memory errors that appear later in the epoch (a strong hint something is accumulating)
- **Sum vs mean reduction**: Sum reduction multiplies the effective learning rate by batch size and produces confusingly large loss values
- **Forgetting to restore train mode**: After eval-mode validation, dropout stays off and batch normalization uses running statistics instead of batch statistics during training

## Practical Debugging Techniques

- **Overfit a single batch**: Extract one batch and train on it for many iterations. Loss should drop to near zero. If not, the bug is in the architecture, loss function, or training step logic
- **Check gradient flow**: After backward, inspect gradient magnitudes. Exactly zero gradients suggest a disconnected graph; extremely large gradients suggest explosion; extremely small (e.g., $10^{-10}$) suggest vanishing gradients
- **Monitor parameter update ratio**: The ratio $\eta \|\nabla_\theta \mathcal{L}\| / \|\theta\|$ should be roughly $10^{-3}$. Much larger indicates learning rate too high; much smaller indicates too low
- **Read the loss curve shape**:
  - No decrease at all: learning rate too small, insufficient model capacity, or training loop bug
  - Decrease then explosion: learning rate too high
  - Rapid decrease then plateau: underfitting or learning rate needs decay
  - Wild oscillation: batch size too small or learning rate near the stability boundary
- **Set random seeds for debugging**: Ensures weight initialization, data shuffling, and dropout masks are identical across runs, isolating the effect of code changes from random variation
