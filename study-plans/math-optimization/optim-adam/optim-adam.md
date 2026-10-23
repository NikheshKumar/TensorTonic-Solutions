# <span style="font-size: 20px;">Adam</span>

## Motivation

Momentum accelerates convergence by building up velocity in consistent gradient directions. RMSProp adapts per-parameter learning rates using an EMA of squared gradients. Adam (Adaptive Moment Estimation) combines both ideas into a single optimizer that tracks both the first moment (mean) and second moment (uncentered variance) of the gradients.

## The Adam Update

Adam maintains two exponential moving averages per parameter:

**First moment** (momentum-like):

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

**Second moment** (RMSProp-like):

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

## Bias Correction

Since $m_0 = 0$ and $v_0 = 0$, the estimates are biased toward zero in early steps. Adam corrects this:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

The correction is significant early in training (when $\beta^t$ is large) and becomes negligible as $t$ grows.

## Parameter Update

$$
w_{t+1} = w_t - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

The effective learning rate for parameter $j$ is approximately:

$$
\alpha_{\text{eff},j} \approx \frac{\alpha}{\sqrt{\hat{v}_{t,j}} + \epsilon}
$$

## Default Hyperparameters

The original paper recommends $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The high $\beta_2$ means the second moment averages over roughly the last 1000 steps, providing a stable denominator.

## Why Adam Works Well

- **From momentum**: gradient direction is smoothed, reducing oscillation and accelerating progress through narrow valleys
- **From RMSProp**: per-parameter scaling adapts to each parameter's gradient magnitude
- **Bias correction**: ensures reasonable step sizes from the very first iteration instead of taking tiny steps due to zero-initialized moments
