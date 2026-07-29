# <span style="font-size: 20px;">Batch vs Stochastic vs Mini-Batch Gradient Descent</span>

## The Three Modes

Gradient descent has three variants that differ in how much data is used to compute each gradient:

- **Full-batch GD:** Uses the entire dataset for every gradient computation. One gradient step per epoch.
- **Stochastic GD (SGD):** Uses a single sample for each gradient step. $n$ gradient steps per epoch (where $n$ is the dataset size).
- **Mini-batch GD:** Uses a small batch of $B$ samples. $\lceil n/B \rceil$ gradient steps per epoch.

All three minimize the same objective, but they navigate the loss landscape very differently.

## Update Rules

For a dataset $\{(x_i, y_i)\}_{i=1}^n$ and loss $L(w) = \frac{1}{n}\sum_{i=1}^n \ell(w; x_i, y_i)$:

**Full-batch:**

$$
w_{t+1} = w_t - \alpha \nabla L(w_t) = w_t - \frac{\alpha}{n} \sum_{i=1}^n \nabla \ell(w_t; x_i, y_i)
$$

**Stochastic (single sample $i$):**

$$
w_{t+1} = w_t - \alpha \nabla \ell(w_t; x_i, y_i)
$$

**Mini-batch (batch $\mathcal{B}$ of size $B$):**

$$
w_{t+1} = w_t - \frac{\alpha}{B} \sum_{i \in \mathcal{B}} \nabla \ell(w_t; x_i, y_i)
$$

The mini-batch gradient is an unbiased estimator of the full gradient:

$$
\mathbb{E}_{\mathcal{B}}[\nabla L_{\mathcal{B}}] = \nabla L
$$

## Gradient Variance

The key tradeoff is between gradient quality and update frequency:

- **Full-batch:** Zero variance (exact gradient), but only one update per pass through the data
- **SGD:** High variance (noisy gradient from one sample), but $n$ updates per pass
- **Mini-batch:** Variance reduced by factor $1/B$ compared to SGD, with $n/B$ updates per pass

The variance of the mini-batch gradient estimator is:

$$
\text{Var}[\nabla L_{\mathcal{B}}] = \frac{\sigma^2}{B}
$$

where $\sigma^2$ is the per-sample gradient variance. Larger batches give lower variance but fewer updates per epoch.

## Convergence vs Wall-Clock Time

Full-batch GD converges smoothly but slowly (one step per epoch). SGD converges noisily but makes rapid per-epoch progress because each epoch contains $n$ weight updates. Mini-batch provides the practical sweet spot: enough noise to escape sharp minima, enough averaging to make reliable progress, and efficient use of vectorized hardware.

## Data Shuffling

Each epoch shuffles the data before splitting into batches. This ensures batches are not correlated across epochs and provides the stochastic variation needed for convergence. Without shuffling, the optimizer would cycle through the same batch ordering repeatedly, potentially getting stuck in suboptimal patterns.

## Implementation Details

The gradient norm for each batch quantifies how large the gradient signal is. The variance of gradient norms across batches within an epoch measures gradient noise: full-batch produces a single gradient (variance 0), while SGD and mini-batch produce many gradients whose norms vary depending on which samples are in each batch.
