# <span style="font-size: 20px;">Nesterov Momentum</span>

## Motivation

Classical momentum builds up velocity by accumulating past gradients, then uses that velocity to update the parameters. The problem: the gradient is always computed at the current position, even though the momentum will carry the parameters to a different position. By the time the update lands, the gradient information is slightly stale.

Nesterov Accelerated Gradient (NAG) fixes this with a simple idea: compute the gradient at where momentum is about to take you, not where you currently are.

## Classical vs Nesterov Update

**Classical momentum:**

$$
v_t = \beta v_{t-1} + \nabla f(w_t)
$$

$$
w_{t+1} = w_t - \alpha v_t
$$

**Nesterov momentum:**

$$
v_t = \beta v_{t-1} + \nabla f(w_t - \alpha \beta v_{t-1})
$$

$$
w_{t+1} = w_t - \alpha v_t
$$

The only difference is where the gradient is evaluated. Nesterov computes $\nabla f$ at the "look-ahead" position $w_t - \alpha \beta v_{t-1}$ instead of the current position $w_t$.

## Why Look-Ahead Helps

Consider a ball rolling down a curved valley with momentum. Classical momentum computes the slope at the ball's current position, then applies both the slope and the accumulated velocity. If the velocity is about to overshoot, the gradient at the current position cannot anticipate this.

Nesterov first takes a provisional step in the direction of accumulated velocity, then computes the gradient there. If the ball is about to overshoot, the gradient at the look-ahead position points back more strongly, providing a corrective signal before the overshoot happens.

This gives Nesterov momentum a "course correction" ability that classical momentum lacks.

## Convergence Properties

For convex functions with Lipschitz-continuous gradients:
- Gradient descent converges at rate $O(1/t)$
- Classical momentum also converges at $O(1/t)$ (same asymptotic rate)
- Nesterov momentum converges at $O(1/t^2)$ - provably optimal for first-order methods

This quadratic speedup is why Nesterov momentum is the default in PyTorch's `SGD(nesterov=True)` and is widely used in practice.

## The Rosenbrock Function

The Rosenbrock function $f(x,y) = (1-x)^2 + 100(y - x^2)^2$ has a narrow curved valley that is especially challenging for momentum methods. The minimum is at $(1, 1)$. Nesterov's look-ahead helps navigate this valley more efficiently by anticipating the curvature.
