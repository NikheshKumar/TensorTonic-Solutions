## What Is Gaussian Naive Bayes?

Gaussian Naive Bayes is a classification algorithm for **continuous features**. It assumes each feature follows a Gaussian (normal) distribution within each class, and features are independent given the class label.

$$
P(x_i | y = c) = \frac{1}{\sqrt{2\pi\sigma_{ic}^2}} \exp\left(-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}\right)
$$

where $\mu_{ic}$ and $\sigma_{ic}^2$ are the mean and variance of feature $i$ for class $c$.

---

## The Classification Rule

Using Bayes' theorem:

$$
P(y = c | x) \propto P(y = c) \prod_{i=1}^{d} P(x_i | y = c)
$$

The predicted class is:

$$
\hat{y} = \arg\max_c P(y = c) \prod_{i=1}^{d} P(x_i | y = c)
$$

Substituting the Gaussian likelihood:

$$
\hat{y} = \arg\max_c P(y = c) \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi\sigma_{ic}^2}} \exp\left(-\frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}\right)
$$

---

## Log-Probability Form

To avoid numerical underflow, use log probabilities:

$$
\log P(y = c | x) = \log P(y = c) + \sum_{i=1}^{d} \log P(x_i | y = c)
$$

The Gaussian log-likelihood:

$$
\log P(x_i | y = c) = -\frac{1}{2}\log(2\pi\sigma_{ic}^2) - \frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}
$$

Simplifying:

$$
= -\frac{1}{2}\log(2\pi) - \log(\sigma_{ic}) - \frac{(x_i - \mu_{ic})^2}{2\sigma_{ic}^2}
$$

---

## Parameter Estimation

**Class prior:**

$$
P(y = c) = \frac{N_c}{N}
$$

**Mean for each feature per class:**

$$
\mu_{ic} = \frac{1}{N_c} \sum_{j: y_j = c} x_{ji}
$$

**Variance for each feature per class:**

$$
\sigma_{ic}^2 = \frac{1}{N_c} \sum_{j: y_j = c} (x_{ji} - \mu_{ic})^2
$$

Some implementations use $N_c - 1$ in the denominator (Bessel's correction) for unbiased variance estimation.

---

## Step-by-Step Worked Example

**Training data:** 6 samples, 2 classes, 2 features

Class A (3 samples):
- Sample 1: $[2, 3]$
- Sample 2: $[3, 4]$
- Sample 3: $[4, 5]$

Class B (3 samples):
- Sample 4: $[6, 2]$
- Sample 5: $[7, 1]$
- Sample 6: $[8, 3]$

---

**Step 1: Compute class priors**

$P(A) = 3/6 = 0.5$

$P(B) = 3/6 = 0.5$

---

**Step 2: Compute means**

Class A:
- $\mu_{1A} = (2 + 3 + 4)/3 = 3$
- $\mu_{2A} = (3 + 4 + 5)/3 = 4$

Class B:
- $\mu_{1B} = (6 + 7 + 8)/3 = 7$
- $\mu_{2B} = (2 + 1 + 3)/3 = 2$

---

**Step 3: Compute variances**

Class A:
- $\sigma_{1A}^2 = [(2-3)^2 + (3-3)^2 + (4-3)^2]/3 = (1 + 0 + 1)/3 = 0.667$
- $\sigma_{2A}^2 = [(3-4)^2 + (4-4)^2 + (5-4)^2]/3 = (1 + 0 + 1)/3 = 0.667$

Class B:
- $\sigma_{1B}^2 = [(6-7)^2 + (7-7)^2 + (8-7)^2]/3 = (1 + 0 + 1)/3 = 0.667$
- $\sigma_{2B}^2 = [(2-2)^2 + (1-2)^2 + (3-2)^2]/3 = (0 + 1 + 1)/3 = 0.667$

---

**Step 4: Classify a new sample**

New sample: $x = [5, 3]$

**For Class A:**

Feature 1: $P(x_1=5 | A) = \frac{1}{\sqrt{2\pi(0.667)}} \exp\left(-\frac{(5-3)^2}{2(0.667)}\right)$

$= \frac{1}{\sqrt{4.19}} \exp(-3) = 0.489 \times 0.0498 = 0.0243$

Feature 2: $P(x_2=3 | A) = \frac{1}{\sqrt{4.19}} \exp\left(-\frac{(3-4)^2}{1.333}\right)$

$= 0.489 \times \exp(-0.75) = 0.489 \times 0.472 = 0.231$

$P(A | x) \propto 0.5 \times 0.0243 \times 0.231 = 0.00281$

---

**For Class B:**

Feature 1: $P(x_1=5 | B) = \frac{1}{\sqrt{4.19}} \exp\left(-\frac{(5-7)^2}{1.333}\right)$

$= 0.489 \times \exp(-3) = 0.489 \times 0.0498 = 0.0243$

Feature 2: $P(x_2=3 | B) = \frac{1}{\sqrt{4.19}} \exp\left(-\frac{(3-2)^2}{1.333}\right)$

$= 0.489 \times \exp(-0.75) = 0.489 \times 0.472 = 0.231$

$P(B | x) \propto 0.5 \times 0.0243 \times 0.231 = 0.00281$

---

**Step 5: Normalize**

Both classes have equal probability, so this sample is on the decision boundary.

$P(A | x) = P(B | x) = 0.5$

In practice, you would break ties arbitrarily or examine the raw scores more carefully.

---

## The Naive Assumption

The "naive" assumption is that features are conditionally independent given the class:

$$
P(x_1, x_2, ..., x_d | y) = \prod_{i=1}^{d} P(x_i | y)
$$

This allows us to estimate each feature's distribution separately, dramatically reducing the number of parameters.

**Without independence:** Need to estimate a $d$-dimensional covariance matrix per class: $O(d^2)$ parameters

**With independence:** Only need mean and variance per feature per class: $O(d)$ parameters

---

## Decision Boundary

Gaussian Naive Bayes creates **quadratic decision boundaries** when variances differ between classes, and **linear boundaries** when variances are equal.

The decision boundary is where:

$$
P(y = c_1 | x) = P(y = c_2 | x)
$$

This involves comparing sums of squared terms, which is quadratic in $x$.

---

## Handling Zero Variance

If all samples of a class have the same value for a feature, variance is zero, causing division by zero.

**Solutions:**

**Add small epsilon:**
$\sigma_{ic}^2 = \max(\sigma_{ic}^2, \epsilon)$ where $\epsilon$ is small (e.g., $10^{-9}$)

**Use prior variance:**
Set a minimum variance based on domain knowledge

**Pool variances:**
Use a shared variance across classes for features with insufficient data

---

## Relationship to Linear/Quadratic Discriminant Analysis

**Gaussian Naive Bayes:**
- Diagonal covariance matrix (features independent)
- Different diagonal for each class
- Quadratic decision boundary

**Linear Discriminant Analysis (LDA):**
- Full covariance matrix (features correlated)
- Same covariance for all classes
- Linear decision boundary

**Quadratic Discriminant Analysis (QDA):**
- Full covariance matrix
- Different covariance for each class
- Quadratic decision boundary

Naive Bayes is a special case where off-diagonal covariance elements are zero.

---

## Advantages and Limitations

**Advantages:**
- Simple and fast
- Works well with small datasets
- Handles continuous features naturally
- No hyperparameters to tune
- Provides probability estimates

**Limitations:**
- Assumes Gaussian distribution (may not hold)
- Assumes feature independence (often violated)
- Sensitive to outliers (Gaussian assumption)
- May underperform when correlations are strong

---

## When Gaussian Assumption Fails

If features are not normally distributed:

**Log transform:** For right-skewed positive data

**Box-Cox transform:** General power transform to normality

**Use different Naive Bayes:** Multinomial for counts, Bernoulli for binary

**Kernel density estimation:** Non-parametric density estimate (computationally expensive)

---

## Computational Complexity

**Training:**
- Compute means: $O(N \times d)$
- Compute variances: $O(N \times d)$
- Total: $O(N \times d)$

**Prediction:**
- Compute Gaussian PDF for each feature and class: $O(C \times d)$
- Very fast inference

---

## Numerical Stability

**Log-sum-exp trick:** When computing posterior probabilities, use:

$$
P(y = c | x) = \frac{\exp(\log P(c, x))}{\sum_{c'} \exp(\log P(c', x))}
$$

Subtract the maximum log-probability before exponentiating to avoid overflow:

$$
\log P(c | x) = \log P(c, x) - \log \sum_{c'} \exp(\log P(c', x))
$$