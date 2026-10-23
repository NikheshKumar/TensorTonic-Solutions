# <span style="font-size: 20px;">AdaGrad</span>

## Motivation

Vanilla gradient descent uses the same learning rate for every parameter. This is suboptimal when features have very different frequencies: in a text classification problem, common words like "the" generate gradients on nearly every sample, while rare words like "serendipity" produce gradients only occasionally. A single learning rate is either too large for frequent features (causing oscillation) or too small for rare features (learning too slowly).

AdaGrad (Adaptive Gradient) solves this by maintaining a per-parameter learning rate that adapts based on the history of that parameter's gradients.

## The AdaGrad Update

For each parameter $w_j$, AdaGrad accumulates the sum of squared gradients:

$$
G_{t,j} = G_{t-1,j} + g_{t,j}^2
$$

where $g_{t,j} = \frac{\partial L}{\partial w_j}\bigg|_{w_t}$ is the gradient of the loss with respect to $w_j$ at step $t$.

The update rule divides the base learning rate by the square root of this accumulator:

$$
w_{t+1,j} = w_{t,j} - \frac{\alpha}{\sqrt{G_{t,j} + \epsilon}} \cdot g_{t,j}
$$

where $\epsilon \approx 10^{-8}$ prevents division by zero. In vector form:

$$
w_{t+1} = w_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot g_t
$$

## Per-Parameter Effective Learning Rate

The effective learning rate for parameter $j$ at step $t$ is:

$$
\alpha_{\text{eff},j}^{(t)} = \frac{\alpha}{\sqrt{G_{t,j} + \epsilon}}
$$

This quantity decreases monotonically because $G_{t,j}$ can only increase (it accumulates squared values). Parameters that receive large or frequent gradients accumulate $G$ quickly, shrinking their effective learning rate. Parameters that receive small or infrequent gradients retain a higher effective learning rate.

## Behavior on Sparse Data

When data has sparse features (many zeros), SGD-based AdaGrad automatically adapts:

- **Dense features** (always present): every training sample contributes a non-zero gradient, so $G_j$ grows at every step. The effective learning rate decays quickly.

- **Sparse features** (mostly zeros): only the ~20% of samples where the feature is non-zero contribute to $G_j$. The accumulator grows roughly 5x slower, so the effective learning rate stays higher for longer.

This is exactly the right behavior: rare features need larger updates when they do appear, because each appearance carries proportionally more information.

## The Vanishing Learning Rate Problem

The main limitation of AdaGrad is that $G_t$ grows without bound. After enough training steps, every parameter's effective learning rate approaches zero:

$$
\lim_{t \to \infty} \frac{\alpha}{\sqrt{G_{t,j} + \epsilon}} = 0
$$

This means AdaGrad eventually stops making any progress, even if the optimum has not been reached. This is acceptable for convex problems (where the optimum is reached before the LR decays too much) but problematic for non-convex problems like neural network training.

RMSProp and Adam fix this by using an exponential moving average of squared gradients instead of a cumulative sum, preventing the unbounded growth of $G_t$.

## Logistic Regression

Binary logistic regression models the probability of the positive class as:

$$
\hat{y} = \sigma(x^T w + b) = \frac{1}{1 + e^{-(x^T w + b)}}
$$

The binary cross-entropy loss is:

$$
L = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]
$$

For a single sample, the gradient with respect to $w$ is:

$$
\nabla_w \ell = (\hat{y} - y) \cdot x
$$

When a feature $x_j = 0$ (sparse), the corresponding gradient component $g_j = 0$, so $G_j$ does not accumulate for that sample. This is what causes the differential learning rate behavior between dense and sparse features.
