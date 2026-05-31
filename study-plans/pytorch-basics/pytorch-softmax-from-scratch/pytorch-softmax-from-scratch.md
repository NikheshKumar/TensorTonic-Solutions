<span style="font-size: 14px;">The softmax function converts a vector of raw scores (logits) into a valid probability distribution. It is one of the most important functions in deep learning, appearing in classification heads, attention mechanisms, and reinforcement learning policies.</span>

## <span style="font-size: 14px;">Definition</span>

<span style="font-size: 14px;">Given a vector</span> $z = (z_0, z_1, \ldots, z_{C-1})$ <span style="font-size: 14px;">of</span> $C$ <span style="font-size: 14px;">logits, softmax maps it to a probability vector</span> $p$ <span style="font-size: 14px;">where:</span>

$$
p_j = \frac{e^{z_j}}{\sum_{k=0}^{C-1} e^{z_k}}
$$

<span style="font-size: 14px;">Key properties of the output:</span>

* <span style="font-size: 14px;">Every element is positive:</span> $p_j > 0$ <span style="font-size: 14px;">for all</span> $j$
* <span style="font-size: 14px;">All elements sum to 1:</span> $\sum_j p_j = 1$
* <span style="font-size: 14px;">Larger logits get larger probabilities</span>
* <span style="font-size: 14px;">The relative ordering of values is preserved</span>

## <span style="font-size: 14px;">Why Not Just Normalize?</span>

<span style="font-size: 14px;">A simpler normalization like</span> $p_j = z_j / \sum_k z_k$ <span style="font-size: 14px;">does not work because logits can be negative, producing negative "probabilities." Another attempt,</span> $p_j = |z_j| / \sum_k |z_k|$<span style="font-size: 14px;">, loses information about which logit is largest. The exponential function solves both problems: it maps any real number to a positive number, and it preserves ordering (larger input gives larger output).</span>

## <span style="font-size: 14px;">The Numerical Stability Problem</span>

<span style="font-size: 14px;">The naive formula has a critical flaw. If any logit is large (e.g.,</span> $z_j = 1000$<span style="font-size: 14px;">), then</span> $e^{1000}$ <span style="font-size: 14px;">overflows to infinity in float32 (which can represent values up to about</span> $3.4 \times 10^{38}$<span style="font-size: 14px;">). The result is NaN (infinity divided by infinity).</span>

<span style="font-size: 14px;">Conversely, if all logits are very negative (e.g.,</span> $z_j = -1000$<span style="font-size: 14px;">), all exponentials underflow to 0, and we get 0/0 = NaN.</span>

<span style="font-size: 14px;">These are not edge cases. In practice, logits can easily reach hundreds or thousands during training, especially early on when weights are not yet tuned.</span>

## <span style="font-size: 14px;">The Max-Subtraction Trick</span>

<span style="font-size: 14px;">The fix is elegant: subtract the maximum logit from each element before exponentiating:</span>

$$
p_j = \frac{e^{z_j - m}}{\sum_{k} e^{z_k - m}}, \quad m = \max_k z_k
$$

<span style="font-size: 14px;">This is mathematically identical to the original formula. To see why, factor out</span> $e^m$ <span style="font-size: 14px;">from both numerator and denominator:</span>

$$
\begin{aligned}
\frac{e^{z_j}}{\sum_k e^{z_k}}
&= \frac{e^{-m} \cdot e^{z_j}}{e^{-m} \cdot \sum_k e^{z_k}} \\
&= \frac{e^{z_j - m}}{\sum_k e^{z_k - m}}
\end{aligned}
$$

<span style="font-size: 14px;">After subtracting</span> $m$<span style="font-size: 14px;">:</span>

* <span style="font-size: 14px;">The largest exponent becomes</span> $e^{m - m} = e^0 = 1$<span style="font-size: 14px;">, so nothing overflows</span>
* <span style="font-size: 14px;">Smaller values may underflow to 0, but that is harmless: it just means those classes get probability near 0</span>
* <span style="font-size: 14px;">The denominator is at least 1, so no division by zero</span>

<span style="font-size: 14px;">This trick costs almost nothing: one pass to find the max, then a subtraction. Every production softmax implementation uses it.</span>

## <span style="font-size: 14px;">Batched Softmax</span>

<span style="font-size: 14px;">In practice, softmax is applied row-wise to a 2-D tensor of shape</span> $(N, C)$ <span style="font-size: 14px;">where</span> $N$ <span style="font-size: 14px;">is the batch size and</span> $C$ <span style="font-size: 14px;">is the number of classes. Each row is an independent logit vector, and softmax normalizes each row independently. The max subtraction is also done per row:</span>

$$
m_i = \max_{k} z_{i,k}
$$

<span style="font-size: 14px;">In PyTorch, this means using</span> <code>dim=1</code> <span style="font-size: 14px;">and</span> <code>keepdim=True</code> <span style="font-size: 14px;">so the shapes broadcast correctly during subtraction and division.</span>

## <span style="font-size: 14px;">Softmax Temperature</span>

<span style="font-size: 14px;">A generalized version divides logits by a scalar temperature</span> $T$ <span style="font-size: 14px;">before applying softmax:</span>

$$
p_j = \frac{e^{z_j / T}}{\sum_k e^{z_k / T}}
$$

* $T = 1$<span style="font-size: 14px;">: standard softmax</span>
* $T \to 0$<span style="font-size: 14px;">: the distribution approaches a one-hot vector (argmax), with all probability on the largest logit</span>
* $T \to \infty$<span style="font-size: 14px;">: the distribution approaches uniform</span> $p_j = 1/C$

<span style="font-size: 14px;">Temperature scaling is used in knowledge distillation (training a small model to mimic a large one), calibration (making model confidence more accurate), and sampling from language models (controlling creativity vs. determinism).</span>

## <span style="font-size: 14px;">Derivative of Softmax</span>

<span style="font-size: 14px;">The Jacobian of softmax with respect to the logits is:</span>

$$
\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)
$$

<span style="font-size: 14px;">where</span> $\delta_{ij}$ <span style="font-size: 14px;">is the Kronecker delta (1 if</span> $i = j$<span style="font-size: 14px;">, 0 otherwise). This can be written in matrix form as:</span>

$$
J = \text{diag}(p) - p p^T
$$

<span style="font-size: 14px;">When combined with cross-entropy loss, the gradient simplifies dramatically to</span> $\hat{p} - y$ <span style="font-size: 14px;">(predicted minus true), which is why these two functions are almost always used together.</span>

## <span style="font-size: 14px;">Softmax in Attention Mechanisms</span>

<span style="font-size: 14px;">Beyond classification, softmax is a core component of the attention mechanism in transformers:</span>

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

<span style="font-size: 14px;">Here softmax is applied to the scaled dot products to produce attention weights, which are then used to compute a weighted sum of values. The numerical stability of softmax is critical here because the dot products</span> $QK^T$ <span style="font-size: 14px;">can have large magnitudes.</span>

## <span style="font-size: 14px;">Softmax vs Sigmoid</span>

<span style="font-size: 14px;">For binary classification (2 classes), softmax with 2 outputs is equivalent to sigmoid on the difference of the two logits:</span>

$$
\text{softmax}([z_0, z_1])_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}} = \frac{1}{1 + e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)
$$

<span style="font-size: 14px;">So for binary tasks, you can use either a single logit with sigmoid or two logits with softmax. In practice, the single-logit approach is preferred because it has fewer parameters.</span>

## <span style="font-size: 14px;">Common Mistakes</span>

* <span style="font-size: 14px;">**Forgetting the max subtraction**: Works on small logits, fails silently on large ones with NaN outputs</span>
* <span style="font-size: 14px;">**Wrong dimension**: Applying softmax along dim=0 (across the batch) instead of dim=1 (across classes) gives a completely wrong distribution</span>
* <span style="font-size: 14px;">**Not keeping dimensions**: When subtracting the max or dividing by the sum, failing to use</span> <code>keepdim=True</code> <span style="font-size: 14px;">causes shape mismatches due to broadcasting rules</span>
* <span style="font-size: 14px;">**Double softmax**: Applying softmax to values that are already probabilities (e.g., after a previous softmax) makes the distribution more uniform, losing information</span>

## <span style="font-size: 14px;">Where Softmax Appears</span>

* <span style="font-size: 14px;">**Classification output layers**: Converts final-layer logits to class probabilities</span>
* <span style="font-size: 14px;">**Attention weights**: Normalizes query-key scores in transformers</span>
* <span style="font-size: 14px;">**Mixture models**: Produces mixing coefficients from gating networks</span>
* <span style="font-size: 14px;">**Reinforcement learning**: Converts action logits to a policy distribution for sampling</span>
* <span style="font-size: 14px;">**Gumbel-softmax**: A differentiable approximation to categorical sampling, enabling gradient-based training of discrete choices</span>

## <span style="font-size: 14px;">Relationship to LogSoftmax</span>

<span style="font-size: 14px;">When you need log-probabilities (e.g., for cross-entropy or NLL loss), computing</span> $\log(\text{softmax}(z))$ <span style="font-size: 14px;">naively is unstable because softmax can produce values very close to 0. Instead, compute log-softmax directly:</span>

$$
\log p_j = z_j - m - \log \sum_k e^{z_k - m}
$$

<span style="font-size: 14px;">This avoids the double numerical hazard (overflow in exp, underflow in log) and is what PyTorch's</span> <code>F.log_softmax</code> <span style="font-size: 14px;">does internally.</span>