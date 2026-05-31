## <span style="font-size: 20px;">Gradient Descent from First Principles</span>

- Every neural network training procedure reduces to a single idea: find parameter values that minimize a loss function $\mathcal{L}(\theta)$, where $\theta$ represents all learnable parameters (weights and biases across all layers)
- The gradient $\nabla_\theta \mathcal{L}$ is a vector pointing in the direction of steepest increase of the loss
- By moving in the opposite direction, we decrease the loss
- This does not guarantee reaching a global minimum: for non-convex loss surfaces (which neural networks always have), we can only guarantee convergence to a local minimum or saddle point under certain conditions
- In practice, local minima in high-dimensional overparameterized networks tend to have loss values close to the global minimum, so local minima are "good enough"
- With multiple parameters, the gradient becomes a vector with one component per parameter, and each parameter is updated independently using its own gradient component
- Parameters do not interact during the update step itself, though they do interact during the forward and backward passes that produce the gradients

**Worked example:** Consider a single parameter $\theta$ with loss $\mathcal{L}(\theta) = (\theta - 3)^2$. The gradient is $\frac{d\mathcal{L}}{d\theta} = 2(\theta - 3)$. If $\theta = 7$, the gradient is $+8$ (positive means loss increases as $\theta$ increases), so we should decrease $\theta$. If $\theta = 1$, the gradient is $-4$ (negative), so we should increase $\theta$. The update rule $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ handles both cases automatically by always moving $\theta$ in the direction that reduces the loss.

## The Parameter Update Equation

The vanilla gradient descent update rule:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Breaking this apart:

- $\theta_t$ is the current parameter value at step $t$
- $\nabla_\theta \mathcal{L}(\theta_t)$ is the gradient of the loss with respect to $\theta$, evaluated at the current parameter values
- $\eta$ is the learning rate, a positive scalar that controls the step size
- $\theta_{t+1}$ is the updated parameter value
- The subtraction is critical: gradients point uphill (toward increasing loss), and we want to go downhill. This is why the operation is called gradient descent rather than gradient ascent
- In implementation, this translates to subtracting the product of learning rate and gradient from each parameter tensor. The entire sophistication of training comes from (a) computing the gradient efficiently via backpropagation and (b) choosing good learning rates and update rules

## Learning Rate as Step Size

The learning rate $\eta$ determines how far we move along the negative gradient direction. Its effect is profound:

- If $\eta$ is too large, the update overshoots the minimum. In the quadratic example, if $\theta = 7$ and $\eta = 1.5$, the update gives $\theta_{t+1} = 7 - 1.5 \times 8 = -5$, which is farther from the minimum at $\theta = 3$ than where we started. The loss actually increased. With an even larger learning rate, updates can diverge entirely
- If $\eta$ is too small, convergence is painfully slow. Each step makes negligible progress, and training might require millions of steps to reach acceptable loss. The loss may appear to plateau because steps are too small to escape flat regions
- The right learning rate depends on the curvature of the loss surface. In regions of high curvature (where the loss changes rapidly), a smaller $\eta$ is needed to avoid overshooting. In flat regions, a larger $\eta$ helps make meaningful progress
- This observation motivates adaptive optimizers like Adam, which maintain per-parameter learning rates that adjust based on gradient history. Manual updates use a single scalar $\eta$ for all parameters
- Mental model: the learning rate scales the gradient vector to produce a displacement vector in parameter space. The gradient gives the direction; the learning rate gives the magnitude. Too large a step and you miss the target; too small and you never arrive

## The Relationship Between Gradients and Parameter Updates

- The gradient $\nabla_\theta \mathcal{L}$ encodes how sensitive the loss is to each parameter. Large gradient magnitude means large influence on the loss; near-zero gradient means little effect
- The update rule naturally adjusts step size per parameter based on sensitivity: high-sensitivity parameters get larger updates
- However, gradient magnitude depends on architecture, input distribution, and current parameter values
- In deep networks, gradients in early layers tend to be much smaller than in later layers (the vanishing gradient problem), making vanilla gradient descent difficult without careful initialization or normalization
- The gradient is a local quantity: it says nothing about what happens after a large step. The loss surface might curve sharply, making this the fundamental limitation of first-order methods

## The Computational Graph Lifecycle

Understanding the computational graph is essential for understanding why a no-gradient context is necessary during manual updates. The lifecycle has three phases:

## Graph Construction During the Forward Pass

- When the model processes input, PyTorch executes the forward method and records every operation performed on tensors that have gradient tracking enabled
- Each operation creates a node in a directed acyclic graph (DAG) that tracks which tensors were inputs, which operation was performed, and which tensor was produced as output
- For a single linear layer computing $y = Wx + b$, nodes are created for the matrix multiplication and addition, with edges connecting input $x$, weight $W$, and bias $b$ to the output $y$
- The graph grows with every operation and is stored in memory

## Graph Traversal During the Backward Pass

- When backward is called on the loss, PyTorch starts at the loss node and traverses the graph in reverse topological order, applying the chain rule at each node
- Each intermediate tensor has a gradient function attribute pointing to the function that created it, which knows how to compute the local gradient
- After the backward pass, each parameter's gradient attribute contains $\nabla_\theta \mathcal{L}$
- The graph is consumed during this process: by default, PyTorch frees the graph after backward to reclaim memory

## Graph Destruction After the Backward Pass

- Once backward completes, the computational graph is gone. Parameter tensors still exist with their gradient attributes populated, but the graph connecting them through operations is released
- Each training step builds a new graph during forward, uses it during backward, and discards it
- This lifecycle matters for manual updates because the update step occurs after the graph is destroyed. If PyTorch were to build a new graph during the update, it would create wasteful connections between new parameter values and old gradients

## Why We Need a No-Gradient Context During Updates

Parameters have gradient tracking enabled. Any arithmetic operation involving such tensors is tracked by autograd and creates a new computational graph. When you write an out-of-place subtraction of the scaled gradient from the parameter:

- PyTorch sees a subtraction and multiplication involving a tracked parameter and starts building a new graph
- The result is a new tensor connected to the parameter through the subtraction operation
- This new tensor is not the same object as the original parameter; assigning it back to the loop variable does not update the parameter in the model
- Even in-place operations trigger autograd tracking, which can lead to errors because in-place operations on leaf tensors that require gradients are restricted

The no-gradient context manager solves both problems. Inside this context:

- No operations are recorded in the computational graph
- Results of computations do not have gradient tracking even if inputs do
- In-place operations on parameters are permitted because autograd is not watching
- This is exactly what we want: modify parameter values directly, without creating graph structure, without allocating memory for gradient tracking, and without affecting the parameter's status as a leaf tensor

## The Alternative: Using the Data Attribute

- An older approach modifies the underlying data attribute directly, bypassing the autograd wrapper entirely
- Operations on this attribute are never tracked
- This works but is considered less safe: if you accidentally modify the data attribute during a forward pass, autograd will not know the parameter changed and may compute incorrect gradients
- The no-gradient context is recommended because it is explicit about the scope where tracking is disabled

## In-Place Operations on Parameters

- Inside the no-gradient context, the update uses the in-place subtraction operator
- The in-place operation modifies the existing parameter tensor rather than creating a new one
- Since the parameter iterator yields references to the actual parameter objects stored in the model, in-place modification updates the model's parameters directly
- Using out-of-place subtraction creates a new tensor and binds it to the local variable, but the model's internal reference still points to the old tensor; the model is not updated
- This is a common bug in manual training loops
- The distinction is fundamental to Python's object model: in-place subtraction calls a method that modifies the tensor object itself, while out-of-place subtraction creates a new tensor and rebinds the local name

## Gradient Zeroing: Why and When

- PyTorch accumulates gradients by default. When backward is called, computed gradients are added to whatever is already in each parameter's gradient attribute
- If the attribute is None (before first backward), a new tensor is created. If it already contains values from a previous backward pass, the new gradients are added on top
- This design is intentional: gradient accumulation is useful for simulating larger batch sizes by running multiple forward-backward passes before updating. For a batch of 32 when you want the effect of 128, run four passes before updating
- For standard training (update after every batch), accumulated gradients are a bug. The gradient from step $t$ should not leak into step $t+1$
- Two approaches to clearing:
  - Set gradients to zero: iterates over all parameters and sets gradient tensors to zero. Tensors remain allocated, avoiding reallocation overhead on next backward
  - Set gradients to None: deallocates gradient tensors entirely, saving memory but requiring reallocation. For very large models, this can reduce peak memory usage
- Timing: zero gradients either before backward (start of step) or after update (end of step). Both achieve the same result. The convention is to zero at the beginning, but zeroing at the end is equally valid

## Gradient Flow Through Layers

In a multi-layer network with layers $f_1, f_2, \ldots, f_L$, the gradient of the loss with respect to parameters in layer $k$:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \theta_k}
&= \frac{\partial \mathcal{L}}{\partial f_L}
\cdot \frac{\partial f_L}{\partial f_{L-1}}
\cdots \frac{\partial f_{k+1}}{\partial f_k} \\
&\quad \cdot \frac{\partial f_k}{\partial \theta_k}
\end{aligned}
$$

- This chain of multiplications means the gradient at layer $k$ is a product of $L - k$ Jacobian matrices (or scalars in the 1D case)
- If these Jacobians have norms consistently greater than 1, the product grows exponentially with depth, causing exploding gradients
- If norms are consistently less than 1, the product shrinks exponentially, causing vanishing gradients
- Healthy gradient flow shows gradient norms roughly similar across layers (within an order of magnitude)
- Pathological flow shows norms varying by many orders of magnitude
- Techniques to maintain healthy flow: proper weight initialization, batch normalization (re-centers and re-scales activations at each layer), residual connections (provide direct gradient path from later to earlier layers), and gradient clipping (caps gradient magnitudes)

## Weight Initialization and Its Effect on Training

- If all weights are initialized to zero, every neuron in a layer computes the same output, receives the same gradient, and after update still has the same weights. This symmetry is never broken: the network behaves as if it has one neuron per layer regardless of width (the symmetry-breaking problem)
- If weights are too large, outputs grow exponentially with depth, leading to enormous loss values and gradients (exploding gradient problem)
- If weights are too small, outputs shrink exponentially and gradients vanish to near-zero (vanishing gradient problem)
- Xavier (Glorot) initialization: weights drawn from a distribution with variance $\frac{2}{n_{in} + n_{out}}$, where $n_{in}$ and $n_{out}$ are the fan-in and fan-out of the layer
- Kaiming (He) initialization: variance $\frac{2}{n_{in}}$, designed for ReLU activations which halve variance (since they zero out negative values)
- PyTorch linear layers use Kaiming uniform by default for weights and uniform for biases
- When using manual weight updates, you inherit whatever initialization the model layers use, so the first gradient step operates on well-initialized parameters

## Gradient Clipping in Manual Updates

When gradients are very large (common in recurrent networks), clipping caps the gradient norm to a maximum value, preserving direction but limiting magnitude:

$$
\hat{g} = \begin{cases} g & \text{if } \|g\| \leq c \\ c \cdot \frac{g}{\|g\|} & \text{if } \|g\| > c \end{cases}
$$

- $c$ is the clipping threshold
- Applied between the backward pass and the update step
- Compute total gradient norm across all parameters; if it exceeds the threshold, scale all gradients down proportionally
- This is an example of an operation that is easy to understand in a manual loop: you have full access to the gradients between backward and update and can transform them however you wish

## Debugging Gradient Issues

Manual weight updates make it straightforward to instrument the training loop for debugging:

- **Gradients are None:** the parameter was not involved in computing the loss. Causes include: the parameter belongs to a layer not used in the forward pass, the computation was detached at some point, or gradient tracking was disabled
- **Gradients are exactly zero:** the parameter does not affect the loss at the current operating point. Common with ReLU when all inputs to a neuron are negative (dead ReLU problem): the neuron outputs zero, the gradient through ReLU for negative inputs is zero, and the weight can never recover
- **Gradients are NaN or Inf:** typically from numerical instability (division by zero, log of zero, overflow). These propagate through the update and corrupt all subsequent computations
- **Loss not decreasing:** learning rate may be wrong (too high causes oscillation, too low causes plateaus). Another cause is accumulated gradients (forgot to zero them): the effective gradient grows with each step, eventually causing divergence. Monitor gradient norm at each step to diagnose
- **Parameters not changing:** the in-place operation may not be working. Likely used out-of-place subtraction instead of in-place. Snapshot parameter values before and after the update to verify

## Beyond Vanilla SGD: What Optimizers Add

Understanding the manual update makes clear what advanced optimizers contribute. Each modifies the basic update rule:

**SGD with momentum** maintains a velocity vector $v$ that accumulates past gradients with exponential decay:

$$
v_{t+1} = \beta v_t + \nabla_\theta \mathcal{L}, \quad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

This smooths out noisy gradients and helps accelerate through narrow valleys. Implementing manually would require a dictionary mapping each parameter to its velocity tensor.

**Adam** maintains both a first moment (mean of gradients) and second moment (mean of squared gradients):

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\begin{aligned}
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

Implementing Adam manually would require two state tensors per parameter plus a step counter for bias correction.

All optimizers share the same skeleton: read gradients, compute an update direction (possibly using state from previous steps), apply the update in-place under a no-gradient context, and zero gradients. The sophistication is entirely in computing the update direction.

## Common Pitfalls in Manual Training Loops

- **Forgetting the no-gradient context:** The update creates a new computational graph, wasting memory and potentially causing errors on the next backward pass
- **Using out-of-place operations:** Writing an out-of-place subtraction does not update the model's parameters; it only rebinds the local variable
- **Forgetting to zero gradients:** Gradients accumulate, leading to incorrect updates that grow in magnitude over time
- **Zeroing gradients between backward and update:** This erases the gradients you just computed, making the update do nothing
- **Not calling backward at all:** Gradients remain None, and the update step crashes when trying to multiply None by the learning rate

## Summary

The manual weight update strips training down to its essence. There is no optimizer object, no state dictionary, no learning rate scheduler: just the raw mechanics of gradient descent. The forward pass computes a loss. The backward pass computes gradients. The update step subtracts a scaled gradient from each parameter. Gradient zeroing prepares for the next step. Every optimizer in PyTorch is an elaboration on this theme, adding momentum, adaptive rates, or regularization, but the core loop remains the same. Understanding this loop at the manual level gives you the foundation to reason about training dynamics, debug optimization issues, and implement custom update rules.
