# <span style="font-size: 20px;">RMSProp</span>

## Motivation

AdaGrad accumulates squared gradients in a sum that only grows, causing effective learning rates to monotonically decay toward zero. This works for convex problems but stalls training on non-convex ones (like neural networks) because the optimizer eventually stops making progress.

RMSProp (Root Mean Square Propagation), proposed by Hinton in a Coursera lecture, fixes this by replacing the cumulative sum with an exponential moving average (EMA).

## The RMSProp Update

For each parameter $w_j$, RMSProp maintains a running average of squared gradients:

$$
E[g^2]_{t,j} = \rho \cdot E[g^2]_{t-1,j} + (1 - \rho) \cdot g_{t,j}^2
$$

where $\rho \in [0, 1)$ is the decay rate (typically 0.9). The update rule is:

$$
w_{t+1,j} = w_{t,j} - \frac{\alpha}{\sqrt{E[g^2]_{t,j} + \epsilon}} \cdot g_{t,j}
$$

In vector form:

$$
w_{t+1} = w_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t
$$

## Why the EMA Fixes AdaGrad

The key difference is that the EMA naturally "forgets" old squared gradients. If a parameter's recent gradients are small, the EMA shrinks, and the effective learning rate recovers. In AdaGrad, a burst of large early gradients permanently suppresses the learning rate; in RMSProp, the effect fades exponentially.

The effective learning rate at step $t$ is:

$$
\alpha_{\text{eff},j}^{(t)} = \frac{\alpha}{\sqrt{E[g^2]_{t,j} + \epsilon}}
$$

Unlike AdaGrad's monotonic decay, this quantity can increase or decrease depending on the recent gradient magnitudes.

## Role of the Decay Rate

- $\rho = 0$: no memory, equivalent to dividing by the current gradient magnitude (sign-based update)
- $\rho \to 1$: long memory, approaches AdaGrad-like behavior since old gradients are never forgotten
- $\rho = 0.9$ (default): averages over roughly the last $\frac{1}{1 - \rho} = 10$ steps

## Behavior on Sparse Data

Like AdaGrad, RMSProp applied with SGD on sparse data produces per-parameter adaptation:

- **Dense features**: receive gradients every step, so $E[g^2]_j$ stays large and the effective LR stays moderate
- **Sparse features**: receive gradients infrequently, so $E[g^2]_j$ decays between updates, giving a higher effective LR when a non-zero gradient finally arrives

The crucial difference is that RMSProp can sustain this adaptation indefinitely, while AdaGrad's sparse-feature advantage eventually gets overwhelmed by the cumulative sum.
