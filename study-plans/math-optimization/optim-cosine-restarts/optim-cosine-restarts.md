# <span style="font-size: 20px;">Cosine Annealing with Warm Restarts (SGDR)</span>

## Motivation

A fixed learning rate wastes optimization potential: too high causes oscillation near the minimum, too low wastes time in flat regions. Cosine annealing gradually decreases the LR following a cosine curve, spending more time at lower learning rates near convergence.

SGDR (Stochastic Gradient Descent with Warm Restarts), proposed by Loshchilov and Hutter (2017), extends this by periodically resetting the learning rate back to its maximum. Each restart gives the optimizer a chance to escape local minima and explore new regions of the loss landscape.

## The SGDR Schedule

Within each cycle, the learning rate follows a cosine curve:

$$
\begin{aligned}
\eta_t &= \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)
\end{aligned}
$$

where $T_{\text{cur}}$ is the number of epochs since the last restart, $T_i$ is the length of the current cycle, $\eta_{\max}$ and $\eta_{\min}$ are the upper and lower LR bounds.

## Cycle Length Growth

The cycle length can grow between restarts using a multiplier $T_{\text{mult}}$:

- **First cycle**: length $T_0$
- **Second cycle**: length $T_0 \cdot T_{\text{mult}}$
- **Third cycle**: length $T_0 \cdot T_{\text{mult}}^2$

With $T_{\text{mult}} = 1$, all cycles have the same length. With $T_{\text{mult}} = 2$, each cycle is twice as long as the previous one, allowing finer-grained optimization as training progresses.

## Why Restarts Help

- **Escaping local minima**: the sudden LR increase after a restart can push the optimizer out of a poor local minimum
- **Snapshot ensembling**: models saved at the end of each cycle (when LR is lowest and the model is most converged) can be averaged for better generalization
- **Exploration-exploitation tradeoff**: high LR explores, low LR exploits. Cycling between them balances both
