# <span style="font-size: 20px;">Softmax + Cross-Entropy Gradient</span>

## The Most Important Derivation in Classification

If there is one calculus derivation that every machine learning practitioner should know by heart, it is this one. The result is strikingly elegant: the gradient of the cross-entropy loss with respect to the logits is simply the softmax output minus the one-hot target:

$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

This single formula powers every classification neural network, from simple logistic regression to GPT-scale transformers. Understanding where it comes from requires carefully working through the softmax function and the cross-entropy loss, applying the chain rule, and watching terms cancel beautifully.

## The Softmax Function

### Definition

Given a vector of logits $\mathbf{z} = (z_1, z_2, \ldots, z_K)$ for $K$ classes, the softmax function produces a probability distribution:

$$p_i = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Each output $p_i$ is positive and the outputs sum to 1: $\sum_{i=1}^K p_i = 1$.

### Interpretation

The softmax function converts raw scores (logits) into probabilities. It has several important properties:

- **Monotonic:** Larger logits produce larger probabilities. If $z_i > z_j$, then $p_i > p_j$.
- **Shift invariant:** $\text{softmax}(\mathbf{z}) = \text{softmax}(\mathbf{z} + c\mathbf{1})$ for any constant $c$. This is the basis of the numerical stability trick (subtracting $\max(\mathbf{z})$).
- **Temperature scaling:** $\text{softmax}(\mathbf{z}/T)$ approaches a one-hot distribution as $T \to 0$ (hard argmax) and a uniform distribution as $T \to \infty$ (maximum entropy).

### The Jacobian of Softmax

The softmax function maps $\mathbb{R}^K$ to the probability simplex, so its Jacobian is a $K \times K$ matrix. The entry $(i, j)$ is:

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}
$$

This can be written compactly as:

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

where $\delta_{ij}$ is the Kronecker delta.

### Deriving the Softmax Jacobian

For the diagonal case ($i = j$), using the quotient rule:

$$
\begin{aligned}
\frac{\partial p_i}{\partial z_i} &= \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} \\
&= \frac{e^{z_i}}{S} - \frac{e^{2z_i}}{S^2} = p_i - p_i^2 = p_i(1 - p_i)
\end{aligned}
$$

where $S = \sum_j e^{z_j}$.

For the off-diagonal case ($i \neq j$):

$$\frac{\partial p_i}{\partial z_j} = \frac{0 - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -p_i p_j$$

The numerator $e^{z_i}$ does not depend on $z_j$ when $i \neq j$, so the derivative of the numerator is 0.

## Cross-Entropy Loss

### Definition

Given the softmax probabilities $\mathbf{p}$ and a one-hot target $\mathbf{y}$ (where $y_k = 1$ for the correct class $k$ and $y_j = 0$ for all $j \neq k$), the cross-entropy loss is:

$$L = -\sum_{i=1}^{K} y_i \log(p_i)$$

Since $\mathbf{y}$ is one-hot with the true class at index $k$:

$$L = -\log(p_k)$$

The loss is simply the negative log-probability assigned to the correct class. It is zero when $p_k = 1$ (perfect prediction) and increases without bound as $p_k \to 0$ (confident wrong prediction).

### Gradient of Cross-Entropy with Respect to Softmax

The derivative of $L$ with respect to $p_i$ is:

$$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$$

For the correct class $k$: $\frac{\partial L}{\partial p_k} = -\frac{1}{p_k}$. For all other classes: $\frac{\partial L}{\partial p_j} = 0$ (since $y_j = 0$).

## The Chain Rule: Combining Softmax and Cross-Entropy

### Setting Up the Chain Rule

The gradient of the loss with respect to logit $z_j$ is:

$$\frac{\partial L}{\partial z_j} = \sum_{i=1}^{K} \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}$$

Substituting the known derivatives:

$$= \sum_{i=1}^{K} \left(-\frac{y_i}{p_i}\right) \cdot p_i(\delta_{ij} - p_j)$$

$$= -\sum_{i=1}^{K} y_i (\delta_{ij} - p_j)$$

$$= -\sum_{i=1}^{K} y_i \delta_{ij} + p_j \sum_{i=1}^{K} y_i$$

### The Beautiful Cancellation

Now observe:

- $\sum_{i=1}^{K} y_i \delta_{ij} = y_j$ (the Kronecker delta picks out the $j$-th element)
- $\sum_{i=1}^{K} y_i = 1$ (since $\mathbf{y}$ is a probability distribution / one-hot vector)

Therefore:

$$\frac{\partial L}{\partial z_j} = -y_j + p_j = p_j - y_j$$

This is the key result:

$$\boxed{\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}}$$

The gradient is simply the prediction minus the target. No logarithms, no exponentials, no divisions - just a subtraction.

## Why This Result Is Remarkable

### Simplicity

The softmax function involves exponentials and normalization. The cross-entropy involves logarithms and sums. Yet when composed and differentiated, everything cancels, leaving the simplest possible expression: $\mathbf{p} - \mathbf{y}$.

This cancellation is not a coincidence. Softmax + cross-entropy is the natural loss function for the categorical distribution in the exponential family. The gradient $\mathbf{p} - \mathbf{y}$ is the difference between the model's "sufficient statistics" and the data's sufficient statistics, which is the standard form for exponential family gradients.

### Numerical Stability

Because the gradient $\mathbf{p} - \mathbf{y}$ involves only the softmax probabilities (not their logarithms), it is numerically well-behaved. The individual terms $\frac{\partial L}{\partial p_i}$ and $\frac{\partial p_i}{\partial z_j}$ can be numerically problematic (division by small $p_i$, multiplication by small $p_ip_j$), but they cancel when composed. This is why PyTorch's `CrossEntropyLoss` takes raw logits rather than softmax probabilities: computing the loss and gradient together avoids numerical issues.

### Gradient Magnitude

The gradient $p_j - y_j$ has a natural interpretation:

- For the correct class $k$: $\frac{\partial L}{\partial z_k} = p_k - 1$. This is negative (pushing the logit up) with magnitude $1 - p_k$. When the model is confident ($p_k \approx 1$), the gradient is tiny. When the model is wrong ($p_k \approx 0$), the gradient is nearly $-1$.

- For incorrect classes $j \neq k$: $\frac{\partial L}{\partial z_j} = p_j$. This is positive (pushing the logit down) with magnitude $p_j$. Wrong classes that receive high probability get large negative updates.

The gradient automatically allocates update magnitude based on how wrong the prediction is. This is "self-calibrating" behavior.

## Implementing the Derivation Numerically

### Step-by-Step Computation

The numerical implementation computes the gradient in two ways:

**Method 1: Chain rule (step by step)**
1. Compute softmax: $p_i = e^{z_i} / \sum_j e^{z_j}$ (with numerical stability)
2. Compute the Jacobian of softmax: $J_{ij} = p_i(\delta_{ij} - p_j)$
3. Compute $\frac{\partial L}{\partial p_i} = -y_i/p_i$
4. Multiply: $\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot J_{ij}$

**Method 2: Direct formula**
1. Compute softmax: $\mathbf{p}$
2. Subtract: $\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}$

Both methods should give the same result, confirming the derivation.

### Numerical Stability

The softmax computation requires the log-sum-exp trick for stability:

$$p_i = \frac{e^{z_i - \max(\mathbf{z})}}{\sum_j e^{z_j - \max(\mathbf{z})}}$$

Subtracting the maximum prevents overflow in the exponentials. This was covered in Problem 4 (Softmax Numerical Stability).

## Connection to Logistic Regression

### The Binary Case

For $K = 2$ classes, softmax reduces to the sigmoid function:

$$p_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + e^{-(z_1 - z_2)}} = \sigma(z_1 - z_2)$$

And the cross-entropy gradient reduces to:

$$\frac{\partial L}{\partial z} = \sigma(z) - y$$

This is the familiar gradient for logistic regression. The multiclass result $\mathbf{p} - \mathbf{y}$ is the natural generalization.

### Maximum Likelihood Estimation

Cross-entropy loss is the negative log-likelihood of the categorical distribution. Minimizing cross-entropy is equivalent to maximum likelihood estimation of the softmax model parameters. The gradient $\mathbf{p} - \mathbf{y}$ is the score function of the categorical distribution.

## The Softmax-Cross-Entropy in Modern Architectures

### Transformers

In transformer language models, the final layer produces logits over the vocabulary (often 50,000+ tokens). The softmax + cross-entropy loss is applied at each position in the sequence. The gradient $\mathbf{p} - \mathbf{y}$ flows back through the entire transformer stack via backpropagation.

### Label Smoothing

Label smoothing replaces the hard one-hot target $\mathbf{y}$ with a smoothed version:

$$y_i^{\text{smooth}} = (1 - \varepsilon) y_i + \frac{\varepsilon}{K}$$

The gradient becomes $\mathbf{p} - \mathbf{y}^{\text{smooth}}$. This prevents the model from becoming overconfident and improves generalization.

### Temperature Scaling for Calibration

After training, the softmax temperature can be adjusted to calibrate the model's confidence. The gradient analysis shows that the temperature affects how strongly the loss penalizes wrong predictions: lower temperatures create sharper gradients, higher temperatures create softer ones.

## The General Exponential Family Connection

The result $\nabla_{\mathbf{z}} L = \mathbf{p} - \mathbf{y}$ is a special case of a general phenomenon. For any distribution in the exponential family:

$$p(\mathbf{x} | \boldsymbol{\theta}) = h(\mathbf{x}) \exp(\boldsymbol{\theta}^\top T(\mathbf{x}) - A(\boldsymbol{\theta}))$$

the gradient of the negative log-likelihood with respect to the natural parameters $\boldsymbol{\theta}$ is:

$$-\nabla_{\boldsymbol{\theta}} \log p = \nabla A(\boldsymbol{\theta}) - T(\mathbf{x})$$

where $\nabla A(\boldsymbol{\theta}) = \mathbb{E}[T(\mathbf{x})]$ is the expected sufficient statistic under the model. For the categorical distribution with softmax parameterization, $\nabla A(\boldsymbol{\theta}) = \mathbf{p}$ and $T(\mathbf{x}) = \mathbf{y}$, giving $\mathbf{p} - \mathbf{y}$.

This means the clean gradient formula is not a lucky accident - it is a fundamental property of the exponential family, which includes Gaussian, Bernoulli, Poisson, and many other distributions used in machine learning.

## Worked Example: Three Classes

Consider $K = 3$ classes with logits $\mathbf{z} = (2.0, 1.0, 0.1)$ and true class $k = 0$ (so $\mathbf{y} = (1, 0, 0)$).

### Forward Pass

First, apply the max-subtraction trick: $\mathbf{z} - \max(\mathbf{z}) = (0, -1, -1.9)$.

Then compute exponentials: $e^0 = 1.0$, $e^{-1} \approx 0.368$, $e^{-1.9} \approx 0.150$.

Sum: $S \approx 1.518$.

Softmax: $\mathbf{p} \approx (0.659, 0.242, 0.099)$.

Cross-entropy loss: $L = -\log(0.659) \approx 0.417$.

### Computing the Gradient via Chain Rule

The Jacobian of softmax is a $3 \times 3$ matrix:

$$J = \begin{pmatrix} p_0(1-p_0) & -p_0 p_1 & -p_0 p_2 \\ -p_1 p_0 & p_1(1-p_1) & -p_1 p_2 \\ -p_2 p_0 & -p_2 p_1 & p_2(1-p_2) \end{pmatrix}
$$

The gradient of loss w.r.t. softmax outputs: $\frac{\partial L}{\partial \mathbf{p}} = (-1/p_0, 0, 0) \approx (-1.518, 0, 0)$.

Multiplying: $\frac{\partial L}{\partial \mathbf{z}} = J^\top \cdot \frac{\partial L}{\partial \mathbf{p}}$.

After the matrix multiplication, all the terms simplify and we get:

$$\frac{\partial L}{\partial \mathbf{z}} \approx (-0.341, 0.242, 0.099)$$

### Computing the Gradient via the Direct Formula

$$\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y} = (0.659 - 1, 0.242 - 0, 0.099 - 0) = (-0.341, 0.242, 0.099)$$

The results match exactly. The direct formula gives the same answer as the full chain rule computation through the Jacobian, confirming the derivation.

### Interpreting the Gradient

The gradient tells the optimizer:
- **Class 0 (correct):** gradient $-0.341$, so increase this logit (the model is not confident enough)
- **Class 1:** gradient $+0.242$, so decrease this logit (the model assigns too much probability here)
- **Class 2:** gradient $+0.099$, so decrease this logit slightly

The magnitudes are proportional to how wrong each prediction is. Class 1 gets a larger correction than class 2 because it received more probability mass.

## Hessian of the Softmax-Cross-Entropy

### Second Derivatives

The Hessian of the cross-entropy loss with respect to the logits is:

$$H_{ij} = \frac{\partial^2 L}{\partial z_i \partial z_j} = \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

This is the same as the softmax Jacobian. The Hessian is positive semi-definite (since the loss is convex in the logits for fixed targets), which means gradient descent on the logits always converges to the global minimum of the cross-entropy loss.

### Newton's Method for Softmax Regression

Using the Hessian, the Newton update for softmax regression is:

$$\mathbf{z}^{(t+1)} = \mathbf{z}^{(t)} - H^{-1}(\mathbf{p} - \mathbf{y})$$

This is called Iteratively Reweighted Least Squares (IRLS) and converges quadratically. The Hessian $H = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ has a special structure that allows efficient inversion using the matrix inversion lemma.

## Practical Implementation Considerations

### Fused Kernels

In practice, softmax and cross-entropy are fused into a single operation (`log_softmax` + `nll_loss`) that computes the loss and gradient without materializing the full softmax probabilities. This reduces memory usage and improves numerical precision by using the log-sum-exp trick internally.

### Gradient Clipping at the Loss Level

Because the gradient $\mathbf{p} - \mathbf{y}$ is bounded (each component lies in $[-1, 1]$), the cross-entropy gradient is inherently well-behaved. Unlike MSE loss (where the gradient can be arbitrarily large), cross-entropy naturally prevents gradient explosion at the output layer. This is another reason it is preferred for classification.

### Multi-Label Classification

For multi-label classification (where multiple classes can be active), each class uses an independent sigmoid + binary cross-entropy. The gradient for class $i$ is $\sigma(z_i) - y_i$, which is the binary version of the softmax formula. The classes do not interact through normalization.

## Information-Theoretic Perspective

### Cross-Entropy and KL Divergence

Cross-entropy can be decomposed as:

$$H(\mathbf{y}, \mathbf{p}) = H(\mathbf{y}) + D_{\text{KL}}(\mathbf{y} \| \mathbf{p})$$

where $H(\mathbf{y})$ is the entropy of the target distribution (which is zero for one-hot targets) and $D_{\text{KL}}$ is the KL divergence. Minimizing cross-entropy is equivalent to minimizing KL divergence, which means finding the model distribution $\mathbf{p}$ closest to the target distribution $\mathbf{y}$ in the information-theoretic sense.

The gradient $\mathbf{p} - \mathbf{y}$ points in the direction that increases the KL divergence the fastest. Gradient descent moves in the opposite direction, reducing the KL divergence and bringing the model's predictions closer to the target.

### Bits and Nats

Cross-entropy measures the expected number of nats (when using natural logarithm) or bits (when using base-2 logarithm) needed to encode the target using the model's distribution. For a perfect model ($\mathbf{p} = \mathbf{y}$), the cross-entropy equals the entropy of the target. For one-hot targets, this is 0 nats.

A cross-entropy of $\log(K)$ (where $K$ is the number of classes) corresponds to the uniform distribution - no better than random guessing. Values below $\log(K)$ indicate the model has learned something useful.
