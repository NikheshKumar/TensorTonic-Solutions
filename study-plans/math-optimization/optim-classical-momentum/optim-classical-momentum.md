# <span style="font-size: 20px;">Classical Momentum</span>

## The Problem with Vanilla GD

Vanilla gradient descent takes steps proportional to the local gradient. On loss surfaces where curvature differs across directions (ill-conditioned problems), this causes a characteristic zigzag pattern: the optimizer oscillates rapidly along the high-curvature direction while making slow progress along the low-curvature direction.

The Rosenbrock function is a classic example:

$$
f(x, y) = (1 - x)^2 + 100(y - x^2)^2
$$

Its minimum is at $(1, 1)$ with $f = 0$. The loss surface has a narrow, curved valley (the "banana") where the curvature across the valley is roughly 100 times steeper than along it. Vanilla GD bounces back and forth across this valley instead of sliding smoothly along it.

## The Momentum Update

Momentum adds a velocity term that accumulates past gradients:

$$
v_{t+1} = \beta \, v_t + \nabla f(w_t)
$$

$$
w_{t+1} = w_t - \alpha \, v_{t+1}
$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9) and $v_0 = 0$.

The velocity $v$ is an exponential moving average of past gradients. After many steps, the effective gradient contribution from $k$ steps ago is weighted by $\beta^k$, decaying geometrically.

## Why Momentum Helps

Consider what happens in a narrow valley:

- **Across the valley:** Gradients alternate in sign each step (oscillation). Momentum averages these opposing gradients, canceling much of the oscillation. The velocity component across the valley stays small.

- **Along the valley:** Gradients consistently point in the same direction. Momentum accumulates these consistent signals, building up speed. The velocity component along the valley grows up to $\frac{1}{1-\beta}$ times the gradient magnitude.

The net effect: momentum dampens oscillation in high-curvature directions and accelerates progress in low-curvature directions.

## The Rosenbrock Gradient

The partial derivatives of the Rosenbrock function are:

$$
\frac{\partial f}{\partial x} = -2(1 - x) - 400x(y - x^2)
$$

$$
\frac{\partial f}{\partial y} = 200(y - x^2)
$$

Near the valley floor (where $y \approx x^2$), the $y$-gradient is close to zero while the $x$-gradient is dominated by the $-2(1-x)$ term pushing toward $x = 1$.

## Effective Step Size

With momentum $\beta$, the effective step size along a consistent gradient direction is amplified by a factor of $\frac{1}{1-\beta}$. For $\beta = 0.9$, this is a 10x amplification. This explains why momentum reaches the Rosenbrock minimum much faster: it effectively uses a 10x larger step size along the valley while keeping the cross-valley step size small.

## Connection to Physics

The name "momentum" comes from the physical analogy: $v$ is a velocity, $\nabla f$ is a force, and $\beta$ is friction (1 minus the drag coefficient). A ball rolling downhill accumulates velocity in consistent directions and naturally dampens oscillation through inertia, mirroring exactly what the momentum update achieves.
