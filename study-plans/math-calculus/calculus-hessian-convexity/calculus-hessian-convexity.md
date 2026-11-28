# <span style="font-size: 20px;">Hessian Matrix & Convexity of Linear Regression Loss</span>

## From Gradients to Second Derivatives

In single-variable calculus, the first derivative $f'(x)$ gives the slope of a function, and the second derivative $f''(x)$ gives the curvature - how the slope itself changes. A function is convex (bowl-shaped) when $f''(x) \geq 0$ everywhere, and strictly convex when $f''(x) > 0$ everywhere.

For a multivariate function $L : \mathbb{R}^d \to \mathbb{R}$, the natural generalization of the second derivative is the **Hessian matrix** $\mathbf{H} \in \mathbb{R}^{d \times d}$:

$$\mathbf{H} = \nabla^2 L = \begin{pmatrix} \frac{\partial^2 L}{\partial w_1^2} & \frac{\partial^2 L}{\partial w_1 \partial w_2} & \cdots & \frac{\partial^2 L}{\partial w_1 \partial w_d} \\ \frac{\partial^2 L}{\partial w_2 \partial w_1} & \frac{\partial^2 L}{\partial w_2^2} & \cdots & \frac{\partial^2 L}{\partial w_2 \partial w_d} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 L}{\partial w_d \partial w_1} & \frac{\partial^2 L}{\partial w_d \partial w_2} & \cdots & \frac{\partial^2 L}{\partial w_d^2} \end{pmatrix}
$$

Each entry $H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$ measures how the partial derivative with respect to $w_i$ changes when $w_j$ is varied. For smooth functions, the mixed partial derivatives commute: $\frac{\partial^2 L}{\partial w_i \partial w_j} = \frac{\partial^2 L}{\partial w_j \partial w_i}$, so the Hessian is always a symmetric matrix.

### The Hessian and Curvature

The Hessian encodes the curvature of $L$ in all directions. The second-order Taylor expansion of $L$ around a point $\mathbf{w}$ is:

$$
\begin{aligned}
L(\mathbf{w} + \boldsymbol{\delta}) &\approx L(\mathbf{w}) + \nabla L(\mathbf{w})^\top \boldsymbol{\delta} \\
&\quad + \frac{1}{2}\boldsymbol{\delta}^\top \mathbf{H} \boldsymbol{\delta}
\end{aligned}
$$

The quadratic term $\frac{1}{2}\boldsymbol{\delta}^\top \mathbf{H} \boldsymbol{\delta}$ determines whether the function curves upward (positive curvature, like a bowl) or downward (negative curvature, like a dome) in the direction $\boldsymbol{\delta}$.

## The Hessian of Linear Regression Loss

Recall the sum-of-squared-errors loss for linear regression:

$$L(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 = (\mathbf{y} - \mathbf{X}\mathbf{w})^\top(\mathbf{y} - \mathbf{X}\mathbf{w})$$

We derived the gradient in the previous problem:

$$\nabla L(\mathbf{w}) = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\mathbf{w}) = 2\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{X}^\top\mathbf{y}$$

The Hessian is obtained by differentiating the gradient:

$$
\begin{aligned}
\mathbf{H} &= \frac{\partial}{\partial \mathbf{w}}(\nabla L) \\
&= \frac{\partial}{\partial \mathbf{w}}(2\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{X}^\top\mathbf{y})
\end{aligned}
$$

Since $2\mathbf{X}^\top\mathbf{y}$ is a constant (with respect to $\mathbf{w}$) and $\frac{\partial}{\partial \mathbf{w}}(\mathbf{A}\mathbf{w}) = \mathbf{A}$ for a constant matrix $\mathbf{A}$:

$$\boxed{\mathbf{H} = 2\mathbf{X}^\top\mathbf{X}}$$

This is a remarkably clean result: the Hessian of the SSE loss depends only on the data matrix $\mathbf{X}$, not on the targets $\mathbf{y}$ or the current weights $\mathbf{w}$. The Hessian is constant everywhere - the curvature of the loss surface is the same at every point in weight space. This is a defining property of quadratic functions.

### Component-wise Derivation

To verify, let us compute $H_{jk}$ directly:

$$\frac{\partial L}{\partial w_j} = -2\sum_{i=1}^n x_{ij}(y_i - \sum_{\ell} w_\ell x_{i\ell})$$

$$
\begin{aligned}
H_{jk} &= \frac{\partial^2 L}{\partial w_j \partial w_k} \\
&= -2\sum_{i=1}^n x_{ij} \cdot \frac{\partial}{\partial w_k}(y_i - \sum_\ell w_\ell x_{i\ell}) \\
&= -2\sum_{i=1}^n x_{ij}(-x_{ik}) = 2\sum_{i=1}^n x_{ij} x_{ik}
\end{aligned}
$$

In matrix notation, $H_{jk} = 2(\mathbf{X}^\top\mathbf{X})_{jk}$, confirming $\mathbf{H} = 2\mathbf{X}^\top\mathbf{X}$.

## Positive Semi-Definiteness and Convexity

### Definition of Positive Semi-Definiteness

A symmetric matrix $\mathbf{A} \in \mathbb{R}^{d \times d}$ is **positive semi-definite (PSD)** if:

$$\mathbf{v}^\top \mathbf{A} \mathbf{v} \geq 0 \quad \text{for all } \mathbf{v} \in \mathbb{R}^d$$

It is **positive definite (PD)** if the inequality is strict for all $\mathbf{v} \neq \mathbf{0}$:

$$\mathbf{v}^\top \mathbf{A} \mathbf{v} > 0 \quad \text{for all } \mathbf{v} \neq \mathbf{0}$$

### Eigenvalue Characterization

For a symmetric matrix, PSD and PD are equivalent to conditions on the eigenvalues:

- **PSD** $\iff$ all eigenvalues $\lambda_i \geq 0$
- **PD** $\iff$ all eigenvalues $\lambda_i > 0$

This follows from the spectral decomposition: $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ where $\mathbf{Q}$ is orthogonal and $\boldsymbol{\Lambda}$ is diagonal with eigenvalues. Then:

$$
\begin{aligned}
\mathbf{v}^\top \mathbf{A} \mathbf{v} &= \mathbf{v}^\top \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top\mathbf{v} \\
&= \mathbf{u}^\top\boldsymbol{\Lambda}\mathbf{u} = \sum_{i=1}^d \lambda_i u_i^2
\end{aligned}
$$

where $\mathbf{u} = \mathbf{Q}^\top\mathbf{v}$. Since $u_i^2 \geq 0$ for all $i$, the entire sum is non-negative if and only if all $\lambda_i \geq 0$.

### Convexity Theorem

**Theorem**: A twice-differentiable function $L : \mathbb{R}^d \to \mathbb{R}$ is convex if and only if its Hessian $\mathbf{H}(\mathbf{w})$ is positive semi-definite for all $\mathbf{w}$.

**Corollary**: If $\mathbf{H}(\mathbf{w})$ is positive definite for all $\mathbf{w}$, then $L$ is strictly convex, meaning it has at most one global minimum and no local minima.

### Proving Convexity of the SSE Loss

For the SSE loss, the Hessian is $\mathbf{H} = 2\mathbf{X}^\top\mathbf{X}$. We need to show this is PSD.

For any vector $\mathbf{v} \in \mathbb{R}^d$:

$$
\begin{aligned}
\mathbf{v}^\top \mathbf{H} \mathbf{v} &= 2\mathbf{v}^\top \mathbf{X}^\top\mathbf{X}\mathbf{v} \\
&= 2(\mathbf{X}\mathbf{v})^\top(\mathbf{X}\mathbf{v}) = 2\|\mathbf{X}\mathbf{v}\|^2 \geq 0
\end{aligned}
$$

Since the squared norm is always non-negative, $\mathbf{H}$ is PSD, and therefore the SSE loss is convex. This is a fundamental result: **linear regression with squared error loss is always a convex optimization problem**.

### When is the Loss Strictly Convex?

The Hessian $2\mathbf{X}^\top\mathbf{X}$ is PD (strictly convex) if and only if $\mathbf{X}\mathbf{v} \neq \mathbf{0}$ for all $\mathbf{v} \neq \mathbf{0}$, which means $\mathbf{X}$ has full column rank ($\text{rank}(\mathbf{X}) = d$). This requires $n \geq d$ (at least as many samples as features) and that the columns of $\mathbf{X}$ are linearly independent.

When $\mathbf{X}$ is rank-deficient (e.g., when features are collinear or $n < d$), some eigenvalues of $\mathbf{X}^\top\mathbf{X}$ are zero, the loss is convex but not strictly convex, and the minimum is not unique - there is a subspace of equally optimal solutions.

## Eigenvalues and the Loss Landscape

The eigenvalues of the Hessian $\mathbf{H} = 2\mathbf{X}^\top\mathbf{X}$ have direct geometric and computational significance.

### Geometric Interpretation

The eigenvectors of $\mathbf{H}$ define the **principal axes** of the loss surface's curvature. The corresponding eigenvalues $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_d$ give the curvature along each axis:

- **Large eigenvalue**: high curvature, the loss changes rapidly in this direction. Moving a small distance along this eigenvector produces a large change in loss.
- **Small eigenvalue**: low curvature, the loss changes slowly. The loss is nearly flat in this direction.
- **Zero eigenvalue**: zero curvature, the loss is perfectly flat. Moving along this eigenvector does not change the loss at all (the minimum is degenerate along this direction).

### The Condition Number

The condition number $\kappa$ of the Hessian is the ratio of the largest to the smallest positive eigenvalue:

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

The condition number measures how elongated the level curves of the loss function are:

- $\kappa = 1$: the level curves are circles (or hyperspheres in higher dimensions). The loss surface is equally curved in all directions, and gradient descent converges in one step with the optimal learning rate.
- $\kappa \gg 1$: the level curves are highly elongated ellipses. Gradient descent zigzags along the narrow direction and converges slowly. The convergence rate is proportional to $(\kappa - 1)/(\kappa + 1)$.

In practice, ill-conditioned problems (large $\kappa$) are common when features have very different scales (e.g., one feature ranges from 0 to 1 and another from 0 to 10000). Feature normalization reduces $\kappa$ and speeds up training.

## Newton's Method and the Hessian Inverse

Newton's method uses the Hessian to account for curvature when choosing the update direction:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}^{-1}\nabla L(\mathbf{w}_t)$$

For the SSE loss:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - (2\mathbf{X}^\top\mathbf{X})^{-1} \cdot 2(\mathbf{X}^\top\mathbf{X}\mathbf{w}_t - \mathbf{X}^\top\mathbf{y})$$

$$= \mathbf{w}_t - \mathbf{w}_t + (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

Newton's method reaches the exact optimum in a single step for quadratic losses. This is because the quadratic Taylor approximation is exact, so the Newton step perfectly accounts for the loss surface geometry.

For non-quadratic losses (logistic regression, neural networks), the Hessian varies with $\mathbf{w}$, and Newton's method requires multiple iterations. But each iteration uses curvature information to take better steps than gradient descent.

### Comparison: Gradient Descent vs Newton's Method

| Property | Gradient Descent | Newton's Method |
|----------|-----------------|-----------------|
| Update | $-\alpha \nabla L$ | $-\mathbf{H}^{-1}\nabla L$ |
| Per-step cost | $O(nd)$ | $O(nd + d^3)$ |
| Steps to converge | $O(\kappa \log(1/\epsilon))$ | $O(\log\log(1/\epsilon))$ |
| Hyperparameters | Learning rate $\alpha$ | None |

Newton's method converges quadratically (doubling digits of accuracy per step) compared to gradient descent's linear convergence. However, it requires computing and inverting the $d \times d$ Hessian, which costs $O(d^3)$ per step. For large $d$, this is prohibitive, motivating quasi-Newton methods (like L-BFGS) that approximate the Hessian inverse.

## The Structure of $\mathbf{X}^\top\mathbf{X}$

The Gram matrix $\mathbf{G} = \mathbf{X}^\top\mathbf{X}$ has entries:

$$G_{jk} = \sum_{i=1}^n x_{ij} x_{ik} = \mathbf{x}_{:,j}^\top \mathbf{x}_{:,k}$$

where $\mathbf{x}_{:,j}$ is the $j$-th column (feature) of $\mathbf{X}$. Each entry is the dot product of two feature columns.

- **Diagonal entries** $G_{jj} = \|\mathbf{x}_{:,j}\|^2$: the squared norm of feature $j$. Features with large magnitudes produce large diagonal entries.
- **Off-diagonal entries** $G_{jk}$: the inner product between features $j$ and $k$. If features are orthogonal, $G_{jk} = 0$. If features are correlated, $|G_{jk}|$ is large.

When the features are orthonormal ($\mathbf{X}^\top\mathbf{X} = \mathbf{I}$), the Hessian is $2\mathbf{I}$, all eigenvalues are 2, $\kappa = 1$, and the loss surface is perfectly isotropic. This is the ideal case for optimization.

When features are collinear (e.g., feature 2 = 2 times feature 1), $\mathbf{X}^\top\mathbf{X}$ becomes singular (rank-deficient), one eigenvalue is zero, and there are infinitely many optimal weight vectors.

## Connection to Regularization

When $\mathbf{X}$ is rank-deficient or ill-conditioned, L2 regularization (ridge regression) modifies the loss:

$$L_{\text{ridge}}(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$$

The Hessian becomes:

$$\mathbf{H}_{\text{ridge}} = 2\mathbf{X}^\top\mathbf{X} + 2\lambda\mathbf{I}$$

Adding $2\lambda\mathbf{I}$ shifts all eigenvalues by $2\lambda$: if $\mu_i$ are the eigenvalues of $2\mathbf{X}^\top\mathbf{X}$, the regularized eigenvalues are $\mu_i + 2\lambda$. Since $\lambda > 0$, all regularized eigenvalues are strictly positive, making $\mathbf{H}_{\text{ridge}}$ positive definite and the regularized loss strictly convex.

The condition number also improves:

$$\kappa_{\text{ridge}} = \frac{\mu_{\max} + 2\lambda}{\mu_{\min} + 2\lambda} \leq \frac{\mu_{\max}}{\mu_{\min}} = \kappa$$

Regularization simultaneously makes the problem well-posed (unique solution) and better-conditioned (easier to optimize). This dual benefit explains why regularization is so ubiquitous in machine learning.

## Computing Eigenvalues in Practice

For a symmetric matrix, eigenvalues can be computed using several methods:

**Characteristic polynomial**: Solve $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$. For $2 \times 2$ matrices, this is a quadratic equation. For larger matrices, this approach is numerically unstable and rarely used in practice.

**QR algorithm**: The standard method used by numpy's `np.linalg.eigvalsh`. It iteratively applies QR decomposition to converge to the eigenvalues. For a $d \times d$ matrix, this takes $O(d^3)$ time.

**Power iteration**: Finds the largest eigenvalue by repeatedly multiplying by the matrix. Useful for sparse or very large matrices where only a few eigenvalues are needed.

For our problem, we use `np.linalg.eigvalsh`, which is specialized for symmetric (Hermitian) matrices and guarantees real eigenvalues returned in ascending order. This is more numerically stable than the general `np.linalg.eigvals` for symmetric matrices.

## The Hessian in Neural Networks

For neural networks with nonlinear activations, the Hessian is no longer constant - it depends on both the data and the current weights. This has profound consequences:

1. **The loss is generally non-convex**: the Hessian has both positive and negative eigenvalues at some points, meaning the loss surface has saddle points and local minima.

2. **The Hessian changes during training**: as $\mathbf{w}$ evolves, the curvature landscape shifts, making optimization more challenging than the fixed-curvature case of linear regression.

3. **Computing the full Hessian is prohibitive**: for a network with $d$ parameters, the Hessian is $d \times d$. For modern networks with millions or billions of parameters, this is impossibly large.

Despite these challenges, the Hessian's eigenspectrum remains informative. Research has shown that the Hessian of trained neural networks often has a bulk of near-zero eigenvalues (corresponding to flat directions) and a few large positive eigenvalues (corresponding to a low-dimensional "important" subspace). This structure suggests that neural network loss landscapes, while non-convex, have special properties that make optimization tractable.

## Summary of Key Relationships

| Property | Condition | Implication |
|----------|-----------|-------------|
| All eigenvalues $\geq 0$ | PSD | Convex loss |
| All eigenvalues $> 0$ | PD | Strictly convex, unique minimum |
| Some eigenvalue $= 0$ | Singular | Flat directions, non-unique minimum |
| Some eigenvalue $< 0$ | Indefinite | Non-convex, saddle points possible |
| $\kappa$ close to 1 | Well-conditioned | GD converges quickly |
| $\kappa \gg 1$ | Ill-conditioned | GD converges slowly, zigzags |

For linear regression with SSE loss, the Hessian $2\mathbf{X}^\top\mathbf{X}$ is always PSD (convex), PD when $\mathbf{X}$ has full column rank, and the condition number depends on the feature scaling and correlations.

## Geometric Interpretation: Elliptical Level Curves

The second-order approximation of $L$ near the minimum $\mathbf{w}^*$ (where $\nabla L = 0$) is:

$$L(\mathbf{w}) \approx L(\mathbf{w}^*) + \frac{1}{2}(\mathbf{w} - \mathbf{w}^*)^\top \mathbf{H} (\mathbf{w} - \mathbf{w}^*)$$

The level curves of this quadratic are ellipses (in 2D) or ellipsoids (in higher dimensions) defined by:

$$(\mathbf{w} - \mathbf{w}^*)^\top \mathbf{H} (\mathbf{w} - \mathbf{w}^*) = c$$

The axes of these ellipses align with the eigenvectors of $\mathbf{H}$, and the lengths of the semi-axes are proportional to $1/\sqrt{\lambda_i}$. Large eigenvalues correspond to short axes (the loss rises steeply), and small eigenvalues correspond to long axes (the loss rises gently).

For our quadratic SSE loss, this approximation is exact (not just approximate), so the level curves are always perfect ellipses whose shape is determined entirely by the eigenvalues of $2\mathbf{X}^\top\mathbf{X}$.
