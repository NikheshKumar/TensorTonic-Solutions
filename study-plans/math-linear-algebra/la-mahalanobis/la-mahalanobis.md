## Mahalanobis Distance

<span style="font-size: 14px;">The Mahalanobis distance measures how far a point $x$ is from a distribution $\mathcal{N}(\mu, \Sigma)$, accounting for the shape and orientation of the distribution:</span>

$$
d_M(x, \mu) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

<span style="font-size: 14px;">When $\Sigma = I$, this reduces to ordinary Euclidean distance $\|x - \mu\|$.</span>

---

## Why Not Euclidean?

<span style="font-size: 14px;">Euclidean distance treats all directions equally, but real data often has features with very different variances and correlations. Consider height (in cm) and weight (in kg): a 10-unit difference in height means something very different from a 10-unit difference in weight. The Mahalanobis distance normalizes by the covariance structure, measuring distances in units of standard deviations along the principal axes of the distribution.</span>

---

## Worked Example (2D)

<span style="font-size: 14px;">Suppose $\mu = (0, 0)^T$ and $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$ (features are positively correlated). The inverse is $\Sigma^{-1} = \frac{1}{8}\begin{pmatrix} 3 & -2 \\ -2 & 4 \end{pmatrix}$. For the point $x = (2, 1)^T$: $d_M^2 = (2, 1) \cdot \frac{1}{8} \begin{pmatrix} 3 & -2 \\ -2 & 4 \end{pmatrix} \begin{pmatrix} 2 \\ 1 \end{pmatrix} = \frac{1}{8}(2, 1)(4, 0)^T = 1$. So $d_M = 1$. Compare with Euclidean distance $\sqrt{4+1} = 2.24$. The Mahalanobis distance is smaller because the point lies along a high-variance direction of the data.</span>

---

## Geometric Interpretation

<span style="font-size: 14px;">The level sets $\{x : d_M(x, \mu) = c\}$ are ellipsoids centered at $\mu$. The axes of these ellipsoids align with the eigenvectors of $\Sigma$, and their lengths are proportional to the square roots of the eigenvalues. In contrast, Euclidean distance produces spherical level sets that ignore the distribution's shape. Eigendecomposing $\Sigma = V D V^T$, the Mahalanobis distance becomes:</span>

$$
d_M = \|D^{-1/2} V^T (x - \mu)\|
$$

<span style="font-size: 14px;">This is the Euclidean distance after whitening the data - transforming to a coordinate system where the distribution is spherical.</span>

---

## Connection to the Chi-Squared Distribution

<span style="font-size: 14px;">The probability density of a multivariate Gaussian is:</span>

$$
p(x) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} d_M^2\right)
$$

<span style="font-size: 14px;">So the Mahalanobis distance directly determines the log-likelihood. Under the assumption that $x \sim \mathcal{N}(\mu, \Sigma)$, the squared Mahalanobis distance follows a chi-squared distribution with $p$ degrees of freedom: $d_M^2 \sim \chi^2_p$. This provides a principled statistical threshold: for $p$ features, the 97.5th percentile of $\chi^2_p$ gives a cutoff above which points are flagged as outliers with 2.5% false positive rate.</span>

---

## Applications in ML

* <span style="font-size: 14px;">**Multivariate outlier detection**: flag points with $d_M^2 > \chi^2_{p, \alpha}$ as outliers. This generalizes the univariate rule of "more than $k$ standard deviations from the mean" to multiple dimensions while accounting for correlations.</span>
* <span style="font-size: 14px;">**Gaussian discriminant analysis**: classify $x$ to the class with smallest Mahalanobis distance (equivalent to maximum likelihood under equal priors). When each class has its own $\Sigma_k$, the decision boundary is quadratic (QDA); when all classes share $\Sigma$, it is linear (LDA).</span>
* <span style="font-size: 14px;">**Anomaly detection**: in production ML systems, monitoring the Mahalanobis distance of new inputs from the training distribution detects distribution shift and out-of-distribution samples.</span>
* <span style="font-size: 14px;">**k-NN with Mahalanobis**: using Mahalanobis distance instead of Euclidean in k-NN accounts for feature scaling and correlations, often improving accuracy without manual feature engineering.</span>
* <span style="font-size: 14px;">**Metric learning**: many metric learning algorithms (LMNN, NCA) learn a matrix $M = \Sigma^{-1}$ that defines a Mahalanobis distance optimized for a downstream task like classification or retrieval.</span>