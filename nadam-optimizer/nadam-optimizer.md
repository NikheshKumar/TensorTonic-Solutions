## The Optimization Landscape

Training a neural network means finding the set of weights that minimizes a loss function. The loss surface is a high-dimensional landscape with hills, valleys, flat regions, and saddle points. The optimizer's job is to navigate this landscape efficiently.

The simplest approach is **vanilla gradient descent**: compute the gradient of the loss with respect to every parameter, then take a step in the opposite direction:

$$
w_t = w_{t-1} - \eta \cdot g_t
$$

where $\eta$ is the learning rate and $g_t$ is the gradient at step $t$.

This works, but it has problems:

- It treats all parameters the same, even though some gradients are consistently large and others are tiny
- It can oscillate back and forth in directions with steep curvature
- It makes no use of information from previous steps
- Choosing a single learning rate that works for all parameters is difficult

Modern optimizers fix these issues by adding **momentum** and **adaptive learning rates**.

---

## Momentum: Using the Past

Instead of relying only on the current gradient, momentum accumulates a running average of past gradients. This is called the **first moment** (the mean):

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

- $\beta_1$ is the decay rate, typically 0.9
- When $\beta_1 = 0.9$, the new moment is 90% of the old moment plus 10% of the current gradient
- This smooths out noisy gradients and builds up speed in consistent directions

Think of it like a ball rolling downhill. Instead of stopping and restarting at each point, it accumulates velocity. If the gradient keeps pointing the same way, the ball accelerates. If the gradient suddenly reverses, the accumulated momentum dampens the change.

---

## Adaptive Learning Rates: The Second Moment

Different parameters need different step sizes. A parameter whose gradient is always large probably needs a smaller learning rate. A parameter with tiny gradients needs a larger one.

The **second moment** tracks how large the gradients have been:

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

- $\beta_2$ is typically 0.999
- $v_t$ is a running average of **squared** gradients
- Large $v_t$ means the gradient for this parameter has been large recently
- Dividing by $\sqrt{v_t}$ gives each parameter its own adaptive learning rate

This is the core idea behind **RMSProp** and **Adam**.

---

## Adam: Combining Both

Adam (Adaptive Moment Estimation) combines momentum and adaptive rates:

1. Update first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. Update second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
3. Update parameters: $w_t = w_{t-1} - \eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon}$

The numerator $m_t$ gives direction and momentum. The denominator $\sqrt{v_t} + \epsilon$ scales each parameter's update by how large its gradients have been. The $\epsilon$ (typically $10^{-8}$) prevents division by zero.

---

## Nesterov Momentum: Looking Ahead

Standard momentum has a weakness: it computes the gradient at the **current** position, then applies momentum. But by the time the update is applied, the momentum has already carried the parameters further.

**Nesterov momentum** fixes this by effectively "looking ahead." Instead of computing the gradient where you are, it computes the gradient at where momentum is about to take you. This gives a better estimate of where you should actually go.

In the original formulation for plain SGD, Nesterov momentum works like this:

- First, take a "lookahead" step using the current momentum
- Then, compute the gradient at that lookahead position
- Use that gradient to update the momentum

The result is faster convergence and better responsiveness to changes in the loss surface. When the optimizer is heading toward a minimum and starts to overshoot, Nesterov momentum detects this sooner because it evaluates the gradient at the future position.

---

## Nadam: Nesterov + Adam

Nadam (Nesterov-accelerated Adaptive Moment Estimation) brings Nesterov's lookahead idea into Adam.

The first two steps are identical to Adam:

1. Update first moment: $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2. Update second moment: $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$

The difference is in the parameter update. Instead of using $m_t$ directly in the numerator, Nadam uses a **Nesterov-adjusted** combination:

$$
w_t = w_{t-1} - \eta \cdot \frac{\beta_1 m_t + (1 - \beta_1) g_t}{\sqrt{v_t} + \epsilon}
$$

Breaking down the numerator $\beta_1 m_t + (1 - \beta_1) g_t$:

- $\beta_1 m_t$: the momentum term, weighted by $\beta_1$. This represents where momentum is "about to take us"
- $(1 - \beta_1) g_t$: the current gradient, weighted by $(1 - \beta_1)$. This is the correction based on what we see right now
- Together, they approximate evaluating the gradient at the lookahead position

Compare this to standard Adam, which uses just $m_t$ in the numerator. Nadam's version incorporates the current gradient more directly into the update, giving it the Nesterov "lookahead" property.

---

## A Concrete Example

Parameters: $w = [1.0, -1.0]$, moments: $m = [0.1, -0.1]$, $v = [0.01, 0.01]$, gradient: $g = [0.2, -0.3]$, $\eta = 0.002$, $\beta_1 = 0.9$, $\beta_2 = 0.999$

**Step 1** (first moment):

- $m_1 = 0.9 \times 0.1 + 0.1 \times 0.2 = 0.09 + 0.02 = 0.11$
- $m_2 = 0.9 \times (-0.1) + 0.1 \times (-0.3) = -0.09 - 0.03 = -0.12$

**Step 2** (second moment):

- $v_1 = 0.999 \times 0.01 + 0.001 \times 0.04 = 0.00999 + 0.00004 = 0.01003$
- $v_2 = 0.999 \times 0.01 + 0.001 \times 0.09 = 0.00999 + 0.00009 = 0.01008$

**Step 3** (Nesterov update for first parameter):

- Nesterov numerator: $0.9 \times 0.11 + 0.1 \times 0.2 = 0.099 + 0.02 = 0.119$
- Denominator: $\sqrt{0.01003} + 10^{-8} \approx 0.10015$
- Update: $1.0 - 0.002 \times \frac{0.119}{0.10015} \approx 1.0 - 0.00238 \approx 0.998$

---

## When to Use Nadam

- Works well as a **drop-in replacement** for Adam, with the same hyperparameters
- Particularly effective on problems where momentum helps significantly (noisy gradients, saddle points)
- Common in NLP tasks, especially training RNNs and Transformers
- The learning rate default is often slightly higher than Adam's ($0.002$ vs $0.001$) because the Nesterov correction makes the optimizer slightly more aggressive