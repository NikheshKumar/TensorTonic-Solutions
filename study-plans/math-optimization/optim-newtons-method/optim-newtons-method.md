# <span style="font-size: 20px;">Newton's Method</span>

## Beyond First-Order Methods

Gradient descent uses only the gradient (first derivative) to determine the update direction. It treats the loss surface as if it has the same curvature in every direction, which is why it zigzags on ill-conditioned problems. Newton's method uses the Hessian (second derivative matrix) to account for curvature, leading to dramatically faster convergence.

## Second-Order Taylor Approximation

At any point $w_t$, the function can be approximated locally by a quadratic:

$$
f(w) \approx f(w_t) + \nabla f(w_t)^T (w - w_t) + \frac{1}{2} (w - w_t)^T H(w_t) (w - w_t)
$$

where $H(w_t)$ is the Hessian matrix of second partial derivatives. Setting the gradient of this quadratic to zero gives the minimizer:

$$
w_{t+1} = w_t - H(w_t)^{-1} \nabla f(w_t)
$$

This is the Newton update. Rather than taking a small step along the gradient, it jumps directly to the minimum of the local quadratic model.

## Quadratic Convergence

For a purely quadratic function, Newton's method converges in exactly one step (since the quadratic approximation is exact). For general smooth functions near a minimum, the error squares at each iteration:

$$
\|w_{t+1} - w^*\| \leq C \|w_t - w^*\|^2
$$

This means the number of correct digits roughly doubles each step. In practice, Newton's method often reaches machine precision in 3-5 iterations from a reasonable starting point, compared to thousands for gradient descent.

## Computing the Hessian

For a function $f: \mathbb{R}^d \to \mathbb{R}$, the Hessian is the $d \times d$ matrix:

$$
H_{ij} = \frac{\partial^2 f}{\partial w_i \partial w_j}
$$

For the Rosenbrock function $f(x,y) = (1-x)^2 + 100(y-x^2)^2$:

$$
H = \begin{bmatrix} 2 - 400(y - x^2) + 800x^2 & -400x \\ -400x & 200 \end{bmatrix}
$$

The Newton step solves $H \, \Delta w = -\nabla f$ for the update $\Delta w$, which is done via `np.linalg.solve(H, -grad)` rather than explicitly inverting $H$.

## The Tradeoff

Newton's method requires computing the $d \times d$ Hessian ($O(d^2)$ storage) and solving the linear system ($O(d^3)$ per step). For deep learning with millions of parameters, this is prohibitively expensive. But for low-dimensional optimization (2D-100D), Newton's method is extremely powerful. Quasi-Newton methods like L-BFGS approximate the Hessian to extend these ideas to higher dimensions.