# <span style="font-size: 20px;">Jacobian of a Neural Network Layer</span>

## From Scalar Derivatives to Vector-Valued Functions

In single-variable calculus, the derivative $f'(x)$ tells us how a scalar output changes with respect to a scalar input. In multivariable calculus, the gradient $\nabla L(\mathbf{w})$ generalizes this to how a scalar output changes with respect to a vector input. But what happens when both the input and the output are vectors?

A neural network layer is a vector-valued function: it takes a vector input $\mathbf{x} \in \mathbb{R}^d$ and produces a vector output $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^m$. To describe how all outputs change with respect to all inputs, we need a matrix of partial derivatives - the **Jacobian matrix**.

## The Jacobian Matrix

Given a vector-valued function $\mathbf{f} : \mathbb{R}^d \to \mathbb{R}^m$ with components $f_1, f_2, \ldots, f_m$, the Jacobian matrix $\mathbf{J} \in \mathbb{R}^{m \times d}$ is defined as:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_d} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_d} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_d} \end{pmatrix}
$$

Each entry $J_{ij} = \frac{\partial f_i}{\partial x_j}$ measures how the $i$-th output changes when the $j$-th input is perturbed, holding all other inputs fixed. Row $i$ of $\mathbf{J}$ is the gradient of the $i$-th output component $\nabla f_i$, and column $j$ describes how all outputs respond to changes in $x_j$.

### The Jacobian as a Linear Map

The Jacobian encodes the best linear approximation of $\mathbf{f}$ near a point:

$$\mathbf{f}(\mathbf{x} + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J} \boldsymbol{\delta}$$

This first-order Taylor expansion shows that the Jacobian maps a small perturbation $\boldsymbol{\delta} \in \mathbb{R}^d$ in the input space to an approximate change $\mathbf{J}\boldsymbol{\delta} \in \mathbb{R}^m$ in the output space. The Jacobian is thus the natural generalization of the derivative to vector-valued functions.

### Relation to the Gradient

When $m = 1$ (scalar output), the Jacobian is a $1 \times d$ row vector, which is the transpose of the gradient:

$$\mathbf{J} = \nabla f^\top$$

When $d = 1$ (scalar input), the Jacobian is an $m \times 1$ column vector containing the ordinary derivatives of each output component.

## A Neural Network Layer

A single layer of a neural network computes:

$$\mathbf{f}(\mathbf{x}) = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

where:
- $\mathbf{x} \in \mathbb{R}^d$ is the input vector (with $d$ features)
- $\mathbf{W} \in \mathbb{R}^{m \times d}$ is the weight matrix ($m$ outputs, $d$ inputs)
- $\mathbf{b} \in \mathbb{R}^m$ is the bias vector
- $\sigma : \mathbb{R} \to \mathbb{R}$ is a scalar activation function applied element-wise

The computation proceeds in two stages:

1. **Linear transformation** (affine map): $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$, producing the pre-activation vector $\mathbf{z} \in \mathbb{R}^m$
2. **Element-wise activation**: $f_i = \sigma(z_i)$ for each $i = 1, \ldots, m$

For this problem, we use the specific dimensions $m = 2$ (outputs) and $d = 3$ (inputs), with the sigmoid activation:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

So $\mathbf{W}$ is $2 \times 3$, $\mathbf{b}$ is $2$-dimensional, $\mathbf{x}$ is $3$-dimensional, and the output $\mathbf{f}$ is $2$-dimensional.

## The Sigmoid Activation Function

The sigmoid function maps any real number to the interval $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Its derivative has a remarkably elegant form:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Proof**: Starting from $\sigma(z) = (1 + e^{-z})^{-1}$:

$$\sigma'(z) = \frac{d}{dz}(1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$

Now observe that:

$$\sigma(z)(1 - \sigma(z)) = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \frac{e^{-z}}{(1 + e^{-z})^2}$$

These expressions are identical, confirming $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

### Properties of $\sigma'$

- $\sigma'(z) > 0$ for all $z$: the sigmoid is strictly monotonically increasing
- Maximum at $z = 0$: $\sigma'(0) = 0.5 \cdot 0.5 = 0.25$
- Decays to 0 as $|z| \to \infty$: for large $|z|$, the sigmoid saturates and becomes nearly flat, so its derivative vanishes

The saturation of $\sigma'$ for large $|z|$ is the source of the "vanishing gradient" problem in deep networks. When pre-activations have large magnitudes, the gradient signal passed through sigmoid layers becomes negligibly small, making learning extremely slow.

### Behavior Across Regimes

The derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ is symmetric about $z = 0$ and achieves its global maximum there. As $|z|$ increases, both $\sigma(z)$ and $1 - \sigma(z)$ are pulled toward their respective extremes, causing $\sigma'(z)$ to decay rapidly toward zero. In practice, for $|z| > 5$, the sigmoid derivative is negligibly small, and the neuron is effectively saturated.

## Deriving the Jacobian of a Neural Network Layer

We want to compute $J_{ij} = \frac{\partial f_i}{\partial x_j}$ where $f_i = \sigma(z_i)$ and $z_i = \sum_{k=1}^d W_{ik} x_k + b_i$.

Applying the chain rule:

$$\frac{\partial f_i}{\partial x_j} = \frac{\partial \sigma(z_i)}{\partial z_i} \cdot \frac{\partial z_i}{\partial x_j} = \sigma'(z_i) \cdot W_{ij}$$

The second factor comes from $\frac{\partial z_i}{\partial x_j} = \frac{\partial}{\partial x_j}\left(\sum_{k} W_{ik} x_k + b_i\right) = W_{ij}$.

In matrix form, the Jacobian is:

$$\mathbf{J} = \text{diag}(\sigma'(\mathbf{z})) \cdot \mathbf{W}$$

where $\text{diag}(\sigma'(\mathbf{z}))$ is the $m \times m$ diagonal matrix with diagonal entries $\sigma'(z_1), \sigma'(z_2), \ldots, \sigma'(z_m)$.

Expanding this for our $2 \times 3$ case:

$$\mathbf{J} = \begin{pmatrix} \sigma'(z_1) & 0 \\ 0 & \sigma'(z_2) \end{pmatrix} \begin{pmatrix} W_{11} & W_{12} & W_{13} \\ W_{21} & W_{22} & W_{23} \end{pmatrix} = \begin{pmatrix} \sigma'(z_1) W_{11} & \sigma'(z_1) W_{12} & \sigma'(z_1) W_{13} \\ \sigma'(z_2) W_{21} & \sigma'(z_2) W_{22} & \sigma'(z_2) W_{23} \end{pmatrix}
$$

Each row of the Jacobian is the corresponding row of $\mathbf{W}$ scaled by the sigmoid derivative at that neuron's pre-activation. This scaling captures the key insight: the sensitivity of each output to each input is the product of the weight connecting them and the local slope of the activation function.

### Why the Jacobian is Not Simply $\mathbf{W}$

Without the activation function, $\mathbf{f}(\mathbf{x}) = \mathbf{W}\mathbf{x} + \mathbf{b}$ is a linear function, and its Jacobian would simply be $\mathbf{W}$ (constant everywhere). The nonlinear activation introduces input-dependent scaling: the Jacobian varies depending on $\mathbf{x}$ because $\sigma'(\mathbf{z})$ depends on $\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$.

Near the origin (where $z_i \approx 0$ and $\sigma'(z_i) \approx 0.25$), the Jacobian is approximately $0.25 \cdot \mathbf{W}$ - the layer behaves nearly linearly but with scaled weights. In the saturated regime (large $|z_i|$), $\sigma'(z_i) \approx 0$, and the corresponding rows of the Jacobian vanish, meaning those outputs become insensitive to input changes.

## Numerical Jacobian via Finite Differences

The numerical Jacobian provides an independent verification of the analytical computation. For each input component $x_j$, we perturb it and observe how each output changes.

Using central differences with step size $h$:

$$J_{ij}^{\text{num}} = \frac{f_i(\mathbf{x} + h\mathbf{e}_j) - f_i(\mathbf{x} - h\mathbf{e}_j)}{2h}$$

where $\mathbf{e}_j$ is the $j$-th standard basis vector. This requires $2d$ forward passes through the layer (two per input dimension). In our case with $d = 3$, that means 6 evaluations.

The numerical Jacobian is computed column by column: perturbing $x_j$ produces the $j$-th column of the Jacobian, containing the partial derivatives of all outputs with respect to $x_j$.

### Algorithm

```
function numerical_jacobian(f, x, h=1e-5):
    d = length(x)
    m = length(f(x))
    J = zeros(m, d)
    for j = 1 to d:
        x_plus = copy(x); x_plus[j] += h
        x_minus = copy(x); x_minus[j] -= h
        J[:, j] = (f(x_plus) - f(x_minus)) / (2 * h)
    return J
```

This is a general-purpose method that works for any differentiable function $\mathbf{f}$, regardless of its internal structure. It requires no knowledge of the function's derivatives, making it an invaluable debugging tool.

## The Jacobian and Backpropagation

The Jacobian is the mathematical foundation of backpropagation. In a multi-layer network:

$$\mathbf{x} \xrightarrow{\mathbf{f}_1} \mathbf{h}_1 \xrightarrow{\mathbf{f}_2} \mathbf{h}_2 \xrightarrow{\cdots} \mathbf{h}_L \xrightarrow{L} \ell$$

where each $\mathbf{f}_k$ is a layer and $\ell$ is a scalar loss. The chain rule for the total derivative gives:

$$\frac{\partial \ell}{\partial \mathbf{x}} = \frac{\partial \ell}{\partial \mathbf{h}_L} \cdot \mathbf{J}_L \cdot \mathbf{J}_{L-1} \cdots \mathbf{J}_1$$

where $\mathbf{J}_k = \frac{\partial \mathbf{f}_k}{\partial \mathbf{h}_{k-1}}$ is the Jacobian of layer $k$.

### Vector-Jacobian Products (VJPs)

In practice, backpropagation does not compute the full Jacobian matrices. Instead, it computes **vector-Jacobian products** (VJPs). Given a "upstream gradient" vector $\mathbf{g} = \frac{\partial \ell}{\partial \mathbf{f}}$, the VJP produces:

$$\frac{\partial \ell}{\partial \mathbf{x}} = \mathbf{g}^\top \mathbf{J} = \mathbf{g}^\top \text{diag}(\sigma'(\mathbf{z})) \mathbf{W}$$

This equals $(\mathbf{g} \odot \sigma'(\mathbf{z}))^\top \mathbf{W}$, where $\odot$ is element-wise multiplication. The VJP is computed in $O(md)$ time, the same cost as a single matrix-vector multiplication, without explicitly constructing the $m \times d$ Jacobian.

### Jacobian-Vector Products (JVPs)

The **Jacobian-vector product** (JVP) computes $\mathbf{J}\mathbf{v}$ for a given tangent vector $\mathbf{v}$:

$$\mathbf{J}\mathbf{v} = \text{diag}(\sigma'(\mathbf{z})) \cdot \mathbf{W}\mathbf{v} = \sigma'(\mathbf{z}) \odot (\mathbf{W}\mathbf{v})$$

This is used in forward-mode automatic differentiation and requires $O(md)$ time.

For a single layer, the full Jacobian is a $2 \times 3$ matrix (only 6 entries), so computing it explicitly is cheap. But for a layer with $m = 1000$ outputs and $d = 1000$ inputs, the Jacobian would have $10^6$ entries, and computing it explicitly would be wasteful when only VJPs or JVPs are needed.

## Worked Example

Let $\mathbf{W} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$, $\mathbf{b} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$, $\mathbf{x} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$

**Step 1: Compute the pre-activation.**

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

**Step 2: Compute the output.**

$$\mathbf{f} = \sigma(\mathbf{z}) = \begin{pmatrix} \sigma(0) \\ \sigma(0) \end{pmatrix} = \begin{pmatrix} 0.5 \\ 0.5 \end{pmatrix}
$$

**Step 3: Compute the sigmoid derivatives.**

$$\sigma'(0) = 0.5 \cdot (1 - 0.5) = 0.25$$

**Step 4: Compute the Jacobian.**

$$\mathbf{J} = \begin{pmatrix} 0.25 & 0 \\ 0 & 0.25 \end{pmatrix} \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix} = \begin{pmatrix} 0.25 & 0 & 0 \\ 0 & 0.25 & 0 \end{pmatrix}
$$

**Interpretation**: At $\mathbf{x} = \mathbf{0}$, the sigmoid is at its steepest point ($\sigma'(0) = 0.25$). The Jacobian shows that:
- $f_1$ depends only on $x_1$ (because $W$ has zeros in the off-diagonal positions), with sensitivity $0.25$
- $f_2$ depends only on $x_2$, also with sensitivity $0.25$
- Neither output depends on $x_3$ (it has zero weight in both rows)

## The Diagonal Structure of the Activation Derivative

The Jacobian $\mathbf{J} = \text{diag}(\sigma'(\mathbf{z})) \cdot \mathbf{W}$ has a particular structure: each row of $\mathbf{W}$ is scaled by a scalar factor $\sigma'(z_i)$. This diagonal scaling arises because the sigmoid is applied element-wise, meaning each output $f_i$ depends on the pre-activation $z_i$ alone, and $z_i$ in turn depends on all inputs through the weights $W_{i,:}$.

If the activation function were applied in a non-element-wise manner (e.g., the softmax, which couples all outputs), the derivative factor would be a full matrix rather than a diagonal one, and the Jacobian computation would be more complex.

For our sigmoid case, the diagonal structure means:
- Row $i$ of the Jacobian is independent of row $j$ for $i \neq j$: the sensitivity of output $i$ to the inputs does not depend on the behavior of output $j$.
- The effect of saturation is local: if $z_1$ is saturated (large $|z_1|$) but $z_2$ is in the linear regime, only the first row of the Jacobian is suppressed.

## Jacobians of Other Activation Functions

The formula $\mathbf{J} = \text{diag}(\phi'(\mathbf{z})) \cdot \mathbf{W}$ applies to any element-wise activation $\phi$:

**ReLU**: $\phi(z) = \max(0, z)$, $\phi'(z) = \mathbf{1}[z > 0]$

The Jacobian is $\mathbf{J} = \text{diag}(\mathbf{1}[\mathbf{z} > 0]) \cdot \mathbf{W}$, which simply zeros out the rows of $\mathbf{W}$ corresponding to negative pre-activations. This is sparse and efficient.

**Tanh**: $\phi(z) = \tanh(z)$, $\phi'(z) = 1 - \tanh^2(z)$

The Jacobian is $\mathbf{J} = \text{diag}(1 - \tanh^2(\mathbf{z})) \cdot \mathbf{W}$, similar in structure to sigmoid but with derivative values in $[0, 1]$ instead of $[0, 0.25]$.

**Linear (identity)**: $\phi(z) = z$, $\phi'(z) = 1$

The Jacobian is simply $\mathbf{J} = \mathbf{W}$, confirming that a linear layer has a constant Jacobian.

## The Chain Rule for Jacobians

For a composition of two functions $\mathbf{g} \circ \mathbf{f}$, the chain rule states:

$$
\begin{aligned}
\frac{\partial(\mathbf{g} \circ \mathbf{f})}{\partial \mathbf{x}} &= \frac{\partial \mathbf{g}}{\partial \mathbf{f}} \cdot \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \\
&= \mathbf{J}_g \cdot \mathbf{J}_f
\end{aligned}
$$

The Jacobian of the composition is the product of the individual Jacobians. This is the matrix generalization of the scalar chain rule $\frac{d}{dx}g(f(x)) = g'(f(x)) \cdot f'(x)$.

For a two-layer network $\mathbf{h} = \sigma(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)$:

$$
\begin{aligned}
\frac{\partial \mathbf{h}}{\partial \mathbf{x}} &= \text{diag}(\sigma'(\mathbf{z}_2)) \cdot \mathbf{W}_2 \\
&\quad \cdot \text{diag}(\sigma'(\mathbf{z}_1)) \cdot \mathbf{W}_1
\end{aligned}
$$

This product of matrices is exactly what backpropagation computes, working from the output layer backward to the input layer.

### The Vanishing Gradient Problem

Each Jacobian factor $\text{diag}(\sigma'(\mathbf{z}_k))$ has entries in $[0, 0.25]$. When we multiply $L$ such diagonal matrices together, the product can shrink exponentially:

$$\prod_{k=1}^L \sigma'(z_k) \leq 0.25^L$$

For $L = 10$ layers, the gradient signal is attenuated by a factor of at most $0.25^{10} \approx 10^{-6}$. For $L = 20$ layers, it is at most $0.25^{20} \approx 10^{-12}$. This exponential decay is the vanishing gradient problem, and it explains why deep networks with sigmoid activations are difficult to train.

ReLU activations ($\sigma'(z) \in \{0, 1\}$) avoid this issue because the non-zero derivatives are exactly 1, so the gradient signal is either passed through unchanged or blocked completely (but never shrunk continuously).

## Practical Applications of the Jacobian

### Gradient Checking in Deep Learning

When implementing a custom neural network layer, computing the analytical Jacobian and comparing it against the numerical Jacobian is a standard debugging technique. If they disagree, there is a bug in either the forward pass or the backward pass.

The typical workflow is:
1. Implement the forward pass $\mathbf{f}(\mathbf{x})$
2. Implement the analytical Jacobian (or VJP for backprop)
3. Compute the numerical Jacobian using finite differences
4. Compare: if $\|\mathbf{J}_{\text{analytical}} - \mathbf{J}_{\text{numerical}}\| / (\|\mathbf{J}_{\text{analytical}}\| + \|\mathbf{J}_{\text{numerical}}\|) < 10^{-5}$, the implementation is likely correct

### Sensitivity Analysis

The Jacobian reveals which inputs have the most influence on each output. For a trained neural network, large Jacobian entries indicate strong input-output dependencies, while small entries indicate weak or negligible dependencies. This can be used for feature importance analysis, pruning, and understanding model behavior.

### Condition Number and Numerical Stability

The singular values of the Jacobian determine how well-conditioned the layer transformation is. A large condition number (ratio of largest to smallest singular value) means that some input directions are amplified much more than others, which can cause numerical issues during optimization. Techniques like batch normalization, weight initialization (Xavier/He), and gradient clipping all aim to keep the Jacobian well-conditioned throughout the network.
