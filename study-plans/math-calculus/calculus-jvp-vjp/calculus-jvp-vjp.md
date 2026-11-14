# <span style="font-size: 20px;">JVP and VJP for a 2-Layer Network</span>

## From Derivatives to Pushforwards and Pullbacks

The derivative of a function $f: \mathbb{R}^d \to \mathbb{R}^p$ at a point $x$ is the Jacobian matrix $J \in \mathbb{R}^{p \times d}$, where $J_{ij} = \partial f_i / \partial x_j$. For a neural network with thousands or millions of parameters, forming the full Jacobian is prohibitively expensive - a network mapping $\mathbb{R}^{1000}$ to $\mathbb{R}^{10}$ has a Jacobian with 10,000 entries, and computing each entry requires a separate perturbation.

The key insight of automatic differentiation (AD) is that you never need to form the full Jacobian. Instead, you can efficiently compute the Jacobian's action on a vector:

- **Jacobian-Vector Product (JVP):** Given a tangent vector $\dot{x} \in \mathbb{R}^d$, compute $J\dot{x} \in \mathbb{R}^p$ without forming $J$. This is also called a "pushforward" because it pushes tangent vectors forward through the computation.

- **Vector-Jacobian Product (VJP):** Given a cotangent vector $\bar{v} \in \mathbb{R}^p$, compute $J^\top\bar{v} \in \mathbb{R}^d$ without forming $J$. This is also called a "pullback" because it pulls cotangent vectors backward through the computation.

Both operations cost roughly the same as one forward pass through the network - dramatically cheaper than forming the full Jacobian. The difference between them determines how efficiently you can compute full Jacobians, gradients, and higher-order derivatives.

## Forward-Mode AD and the JVP

### The Dual Number Perspective

Forward-mode AD can be understood through dual numbers. A dual number has the form $a + b\epsilon$ where $\epsilon^2 = 0$. When you evaluate $f(x + \dot{x}\epsilon)$, the result is $f(x) + J\dot{x}\epsilon$ - the coefficient of $\epsilon$ is exactly the JVP. This is not just a mathematical curiosity; it's the algorithmic principle behind forward-mode AD.

### Forward-Mode as Tangent Propagation

In forward-mode AD, you propagate tangent vectors forward alongside the primal (original) computation. For each intermediate variable $v_i$ in your computation graph, you simultaneously compute the tangent $\dot{v}_i$ using the chain rule:

$$\dot{v}_i = \sum_j \frac{\partial v_i}{\partial v_j} \dot{v}_j$$

The process starts with the input tangent $\dot{x}$ and ends with the output tangent $\dot{f} = J\dot{x}$. Each step applies the local Jacobian of one operation to the incoming tangent vector.

### JVP for Individual Operations

For common neural network operations, the JVP rules are:

**Linear layer** $f(x) = Wx + b$: The Jacobian is $W$, so $\text{JVP}(\dot{x}) = W\dot{x}$.

**ReLU** $f(z) = \max(0, z)$: The Jacobian is $\text{diag}(\mathbb{1}[z > 0])$, so $\text{JVP}(\dot{z}) = \dot{z} \odot \mathbb{1}[z > 0]$.

**Sigmoid** $f(z) = \sigma(z)$: $\text{JVP}(\dot{z}) = \sigma(z)(1 - \sigma(z)) \odot \dot{z}$.

**Matrix multiply** $f(X) = AX$: $\text{JVP}(\dot{X}) = A\dot{X}$.

**Element-wise multiply** $f(a, b) = a \odot b$: $\text{JVP}(\dot{a}, \dot{b}) = \dot{a} \odot b + a \odot \dot{b}$ (product rule).

### Composing JVPs: The Chain Rule

For a composition $f = g \circ h$, the JVP composes naturally:

$$J_f \dot{x} = J_g(J_h \dot{x}) = J_g \dot{h}$$

You first push the tangent through $h$, then push the result through $g$. This is evaluated left-to-right, innermost first - exactly the same order as the forward pass. This is why it's called "forward-mode": tangents flow in the same direction as the computation.

## Reverse-Mode AD and the VJP

### The Adjoint Method

Reverse-mode AD is based on the adjoint method from numerical analysis. Instead of propagating tangent vectors forward, you propagate cotangent vectors (also called adjoint vectors) backward through the computation graph. For each intermediate variable $v_i$, the adjoint $\bar{v}_i$ accumulates the total derivative of the output with respect to $v_i$:

$$\bar{v}_j = \sum_i \bar{v}_i \frac{\partial v_i}{\partial v_j}$$

### VJP for Individual Operations

The VJP rules are the transposes of the JVP rules:

**Linear layer** $f(x) = Wx + b$: $\text{VJP}(\bar{f}) = W^\top\bar{f}$ (and $\bar{b} = \bar{f}$, $\bar{W} = \bar{f}x^\top$).

**ReLU** $f(z) = \max(0, z)$: $\text{VJP}(\bar{f}) = \bar{f} \odot \mathbb{1}[z > 0]$ (same mask as JVP - ReLU is its own transpose).

**Sigmoid** $f(z) = \sigma(z)$: $\text{VJP}(\bar{f}) = \bar{f} \odot \sigma(z)(1 - \sigma(z))$.

**Matrix multiply** $f(X) = AX$: $\text{VJP}(\bar{f}) = A^\top\bar{f}$.

### Composing VJPs: The Reverse Chain Rule

For a composition $f = g \circ h$, the VJP composes in reverse:

$$J_f^\top \bar{v} = J_h^\top(J_g^\top \bar{v}) = J_h^\top \bar{g}$$

You first pull the cotangent back through $g$, then pull the result back through $h$. This is evaluated right-to-left - the reverse of the forward pass. This is why it's called "reverse-mode": cotangents flow opposite to the computation.

### The Memory Trade-off

Reverse-mode requires storing intermediate activations from the forward pass (the "tape") because the backward pass needs them to compute local Jacobians. For our ReLU network, we need to remember which neurons were active ($z > 0$). This is the fundamental memory-computation trade-off in neural network training: reverse-mode is computationally cheaper for gradient computation but requires more memory.

## The 2-Layer Network

### Architecture

Consider the network $f: \mathbb{R}^d \to \mathbb{R}^p$ defined by:

$$f(x) = W_2 \cdot \text{relu}(W_1 \cdot x)$$

where $W_1 \in \mathbb{R}^{m \times d}$ is the first layer weight matrix, $W_2 \in \mathbb{R}^{p \times m}$ is the second layer weight matrix, $x \in \mathbb{R}^d$ is the input, $m$ is the hidden dimension, $d$ is the input dimension, and $p$ is the output dimension.

The forward pass consists of three steps:

1. **First linear layer:** $z_1 = W_1 x$ (shape: $m$)
2. **ReLU activation:** $a_1 = \text{relu}(z_1)$ (shape: $m$)
3. **Second linear layer:** $f = W_2 a_1$ (shape: $p$)

### The Full Jacobian

The full Jacobian $J = \partial f / \partial x$ has shape $(p, d)$. Using the chain rule:

$$J = W_2 \cdot \text{diag}(\mathbb{1}[z_1 > 0]) \cdot W_1$$

This is a $p \times d$ matrix obtained by multiplying three matrices: $W_2$ ($p \times m$), a diagonal mask ($m \times m$), and $W_1$ ($m \times d$). Forming it explicitly requires $O(pmd)$ operations - we want to avoid this.

## Deriving the JVP

Given an input tangent vector $\dot{x} \in \mathbb{R}^d$, the JVP $J\dot{x} \in \mathbb{R}^p$ is computed by pushing tangents forward through each layer:

**Step 1 - Linear tangent:** The first layer is $z_1 = W_1 x$, so the tangent is:

$$\dot{z}_1 = W_1 \dot{x} \quad \text{(shape: } m\text{)}$$

This is a matrix-vector product costing $m \cdot d$ multiplications.

**Step 2 - ReLU tangent:** The activation is $a_1 = \text{relu}(z_1)$, so the tangent is:

$$\dot{a}_1 = \dot{z}_1 \odot \mathbb{1}[z_1 > 0] \quad \text{(shape: } m\text{)}$$

This is an element-wise product costing $m$ multiplications. The indicator function $\mathbb{1}[z_1 > 0]$ uses the pre-activation values from the forward pass.

**Step 3 - Output tangent:** The second layer is $f = W_2 a_1$, so the tangent is:

$$\dot{f} = W_2 \dot{a}_1 \quad \text{(shape: } p\text{)}$$

This is a matrix-vector product costing $p \cdot m$ multiplications.

The total JVP computation costs:

$$C_{\text{JVP}} = md + m + pm$$

## Deriving the VJP

Given an output cotangent vector $\bar{v} \in \mathbb{R}^p$, the VJP $J^\top\bar{v} \in \mathbb{R}^d$ is computed by pulling cotangents backward through each layer:

**Step 1 - Output pullback:** Starting from $\bar{v}$, we pull back through $f = W_2 a_1$:

$$\bar{a}_1 = W_2^\top \bar{v} \quad \text{(shape: } m\text{)}$$

This is a matrix-vector product costing $m \cdot p$ multiplications.

**Step 2 - ReLU pullback:** We pull back through $a_1 = \text{relu}(z_1)$:

$$\bar{z}_1 = \bar{a}_1 \odot \mathbb{1}[z_1 > 0] \quad \text{(shape: } m\text{)}$$

This is an element-wise product costing $m$ multiplications. Note: ReLU's VJP uses the same mask as its JVP.

**Step 3 - Input pullback:** We pull back through $z_1 = W_1 x$:

$$\bar{x} = W_1^\top \bar{z}_1 \quad \text{(shape: } d\text{)}$$

This is a matrix-vector product costing $d \cdot m$ multiplications.

The total VJP computation costs:

$$C_{\text{VJP}} = mp + m + dm$$

## Computational Cost Analysis

### Single-Pass Costs

For a single JVP or VJP, the costs are:

$$C_{\text{JVP}} = md + m + pm = m(d + 1 + p)$$

$$C_{\text{VJP}} = mp + m + dm = m(p + 1 + d)$$

These are identical! A single forward-mode pass and a single reverse-mode pass cost the same number of multiplications: $m(d + p + 1)$. This makes sense because one computes $J\dot{x}$ and the other computes $J^\top\bar{v}$ - both involve the same Jacobian, just applied to vectors of different dimensions.

### Full Jacobian Costs

The JVP $J\dot{x}$ produces one vector of length $p$ - it computes $J$ applied to one input direction. To build the full $p \times d$ Jacobian, you need $d$ JVP calls (one for each standard basis vector $e_1, \ldots, e_d$):

$$C_{\text{full-J via JVP}} = d \cdot m(d + p + 1)$$

The VJP $J^\top\bar{v}$ produces one vector of length $d$ - it computes one row of $J$ (or equivalently, the gradient with respect to all inputs for one output component). To build the full Jacobian, you need $p$ VJP calls:

$$C_{\text{full-J via VJP}} = p \cdot m(d + p + 1)$$

The ratio of costs is:

$$\frac{C_{\text{full-J via JVP}}}{C_{\text{full-J via VJP}}} = \frac{d}{p}$$

### When VJP Wins

VJP is cheaper for computing the full Jacobian whenever $p < d$, meaning the output dimension is smaller than the input dimension. The savings factor is $d/p$.

In neural network training, this is almost always the case:

- **Classification:** $d$ = number of parameters (millions), $p$ = 1 (scalar loss). VJP is millions of times cheaper.
- **Regression:** $d$ = number of parameters, $p$ = output dimension (usually small). VJP wins decisively.
- **Generative models:** Even with high-dimensional outputs, the loss is scalar ($p = 1$).

This is why backpropagation (which IS reverse-mode AD / VJP) is the standard algorithm for training neural networks. Computing the gradient of a scalar loss with respect to millions of parameters requires just ONE reverse pass, whereas forward-mode would require millions of forward passes.

### When JVP Wins

JVP is cheaper when $d < p$ - when there are fewer inputs than outputs. This occurs in:

- **Sensitivity analysis:** How does perturbing one input affect all outputs?
- **Forward-mode in JAX:** `jax.jvp` is preferred when computing directional derivatives.
- **ODE solvers:** Some numerical methods propagate tangent vectors forward.
- **Hessian-vector products:** Computing $Hv$ (Hessian times a vector) can be done with one JVP of the gradient, which is often cheaper than two VJPs.

## VJP is Backpropagation

### The Connection

The VJP of a neural network's loss function with respect to its parameters IS the backpropagation algorithm. When you call `loss.backward()` in PyTorch, you are computing:

$$\bar{\theta} = \left(\frac{\partial L}{\partial \theta}\right)^\top = J_L^\top \cdot 1$$

where $\bar{\theta}$ is the gradient of the scalar loss $L$ with respect to parameters $\theta$, and the cotangent vector is $\bar{v} = 1$ (since $\partial L / \partial L = 1$).

### Layer-by-Layer Correspondence

In our 2-layer network $f(x) = W_2 \cdot \text{relu}(W_1 \cdot x)$ with loss $L = \ell(f(x), y)$:

| Backprop step | VJP step |
|---|---|
| $\delta_2 = \nabla_f \ell$ | $\bar{v} = $ cotangent from loss |
| $\delta_1 = W_2^\top \delta_2$ | $\bar{a}_1 = W_2^\top \bar{v}$ |
| $\delta_1 = \delta_1 \odot \text{relu}'(z_1)$ | $\bar{z}_1 = \bar{a}_1 \odot \mathbb{1}[z_1 > 0]$ |
| $\nabla_{W_1} = \delta_1 x^\top$ | weight gradient (VJP w.r.t. $W_1$) |

Every step of backpropagation corresponds to a VJP through one layer. The "delta" values from the original backprop paper are exactly the cotangent vectors of reverse-mode AD.

### Historical Context

Backpropagation was developed in the 1970s-80s by researchers including Paul Werbos, David Rumelhart, Geoffrey Hinton, and Ronald Williams. The connection to reverse-mode AD (developed independently in the applied mathematics community by Seppo Linnainmaa and others) was recognized later. Today, they are understood as the same algorithm applied in different contexts - "backpropagation" for neural networks, "reverse-mode AD" for general differentiable programs.

## Implementation Details

### ReLU Derivative at Zero

The ReLU function $\text{relu}(z) = \max(0, z)$ is not differentiable at $z = 0$. By convention, we define $\text{relu}'(0) = 0$, which means $\mathbb{1}[z > 0]$ uses strict inequality. This is the standard convention in deep learning frameworks. Alternatively, some implementations use $z \geq 0$, but the choice rarely matters in practice since the probability of any pre-activation being exactly zero is negligible with continuous inputs.

### Numerical Considerations

When implementing JVP and VJP manually:

- Use `float64` for numerical stability in comparisons with reference implementations
- The ReLU mask should be computed from the forward pass pre-activations $z_1$, not the post-activations $a_1$
- Both JVP and VJP should produce results that match the full Jacobian computation: $\text{JVP}(\dot{x}) = J\dot{x}$ and $\text{VJP}(\bar{v}) = J^\top\bar{v}$

### Computing the FLOP Count

For counting multiply operations in our implementation:

- Matrix-vector product of shape $(a, b)$ times $(b,)$: costs $a \cdot b$ multiplies
- Element-wise product of two vectors of length $n$: costs $n$ multiplies
- Do not count additions - we count only multiplications as the dominant cost

The per-pass cost is $m(d + 1 + p)$ for both JVP and VJP. The full Jacobian cost via JVP is $d$ times this (one pass per input dimension), and via VJP is $p$ times this (one pass per output dimension).

## PyTorch and JAX: Two Approaches to AD

### PyTorch: VJP by Default

PyTorch's `autograd` engine implements reverse-mode AD (VJP). When you define a custom `torch.autograd.Function`, you implement:

- `forward(ctx, *inputs)`: Compute the output and save values needed for backward
- `backward(ctx, *grad_outputs)`: Given the output cotangent, return input cotangents

The `grad_outputs` are the $\bar{v}$ vectors, and `backward` returns the VJP $J^\top\bar{v}$. PyTorch also supports `torch.autograd.functional.jvp` and `torch.func.jvp` for forward-mode, but reverse-mode is the default because gradients of scalar losses are the most common need.

### JAX: Both Modes as First-Class

JAX provides both `jax.jvp` (forward-mode) and `jax.vjp` (reverse-mode) as core primitives. The `jax.grad` function is built on `jax.vjp`. JAX also supports:

- `jax.jacfwd`: Full Jacobian via forward-mode (uses `jax.vmap` over `jax.jvp`)
- `jax.jacrev`: Full Jacobian via reverse-mode (uses `jax.vmap` over `jax.vjp`)
- `jax.hessian`: Computed as the Jacobian of the gradient

The choice between `jacfwd` and `jacrev` depends on the function's shape: use `jacfwd` when $d < p$ and `jacrev` when $p < d$.

## Higher-Order Derivatives

### Hessian-Vector Products

The Hessian $H = \nabla^2 f$ can be computed as the Jacobian of the gradient. A Hessian-vector product $Hv$ can be computed efficiently by:

1. Compute the gradient $g(x) = \nabla f(x)$ using VJP
2. Compute $Hv = Jg \cdot v$ using JVP of the gradient

This "VJP then JVP" approach gives exact Hessian-vector products at the cost of one forward pass and one backward pass, without forming the full $d \times d$ Hessian. This is the basis of conjugate gradient methods for second-order optimization.

### Fisher Information Matrix

The Fisher Information Matrix $F = E[\nabla \log p \cdot \nabla \log p^\top]$ is related to the Hessian of the negative log-likelihood. Natural gradient methods use $F^{-1}\nabla L$ instead of $\nabla L$. Computing $F^{-1}v$ efficiently uses the same JVP/VJP machinery.

## Practical Implications for Deep Learning

### Why Networks Are Trained with Backprop

A typical neural network might have $d = 10^6$ parameters and a scalar loss ($p = 1$). Computing the gradient via:

- **VJP (backprop):** 1 reverse pass - cost $\sim m(d + p + 1) \approx md$
- **JVP (forward-mode):** $d = 10^6$ forward passes - cost $\sim d \cdot md = md^2$

VJP is $d = 10^6$ times cheaper. This is why backpropagation revolutionized neural network training - it makes gradient computation tractable for large networks.

### Per-Sample Gradients

Computing per-sample gradients (needed for differential privacy, influence functions, etc.) traditionally requires one backward pass per sample. Recent research on "per-sample gradient tricks" uses JVP combined with VJP to compute these more efficiently.

### Neural Tangent Kernels

The Neural Tangent Kernel $K(x, x') = J(x) J(x')^\top$ can be computed using JVPs and VJPs without forming the full Jacobian. This connects the infinite-width behavior of neural networks to kernel methods.

## Summary

JVP (forward-mode AD) pushes tangent vectors forward through the computation graph, computing $J\dot{x}$. VJP (reverse-mode AD) pulls cotangent vectors backward, computing $J^\top\bar{v}$. For a single pass, both cost the same: $m(d + p + 1)$ multiplications for our 2-layer network. The difference emerges when computing the full Jacobian: JVP needs $d$ passes while VJP needs $p$ passes. When $p \ll d$ (the typical case in deep learning where the loss is scalar), VJP is dramatically cheaper - and this is exactly why backpropagation is the cornerstone algorithm of modern machine learning.
