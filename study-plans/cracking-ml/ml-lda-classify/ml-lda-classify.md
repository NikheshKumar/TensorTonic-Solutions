# <span style="font-size: 20px;">Linear Discriminant Analysis</span>

<span style="font-size: 14px;">Linear Discriminant Analysis is a generative classifier that models each class as a multivariate Gaussian with a shared covariance matrix. By assuming a common covariance structure across all classes, the decision boundaries between classes become linear hyperplanes, which is the defining property that gives LDA its name.</span>

---

## <span style="font-size: 16px;">Generative Model Assumption</span>

<span style="font-size: 14px;">LDA assumes the data in each class</span> $k$ <span style="font-size: 14px;">follows a multivariate Gaussian distribution:</span>

$$
P(x \mid y = k) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_k)^T \Sigma^{-1} (x - \mu_k)\right)
$$

- <span style="font-size: 14px;">Each class has its own mean</span> $\mu_k$
- <span style="font-size: 14px;">All classes share the same covariance</span> $\Sigma$ <span style="font-size: 14px;">(this is what makes it "linear")</span>
- <span style="font-size: 14px;">Class priors are</span> $\pi_k = P(y = k) = n_k / n$

<span style="font-size: 14px;">If each class had its own covariance</span> $\Sigma_k$<span style="font-size: 14px;">, the model becomes Quadratic Discriminant Analysis (QDA), where decision boundaries are quadratic surfaces.</span>

---

## <span style="font-size: 16px;">Deriving the Discriminant Function</span>

<span style="font-size: 14px;">To classify a new point</span> $x$<span style="font-size: 14px;">, we want the class with the highest posterior probability:</span>

$$
\hat{y} = \arg\max_k P(y = k \mid x) = \arg\max_k P(x \mid y = k) \cdot P(y = k)
$$

<span style="font-size: 14px;">Taking the log of the joint and dropping terms that do not depend on</span> $k$<span style="font-size: 14px;">:</span>

$$
\delta_k(x) = \log P(x \mid y=k) + \log P(y=k)
$$

<span style="font-size: 14px;">Expanding the Gaussian log-likelihood:</span>

$$
\log P(x \mid y=k) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(x - \mu_k)^T \Sigma^{-1}(x - \mu_k)
$$

<span style="font-size: 14px;">The first two terms are the same for all classes (shared</span> $\Sigma$<span style="font-size: 14px;">), so we can drop them. Expanding the quadratic form:</span>

$$
(x - \mu_k)^T \Sigma^{-1} (x - \mu_k) = x^T \Sigma^{-1} x - 2x^T \Sigma^{-1} \mu_k + \mu_k^T \Sigma^{-1} \mu_k
$$

<span style="font-size: 14px;">The term</span> $x^T \Sigma^{-1} x$ <span style="font-size: 14px;">is the same for all classes (it does not depend on</span> $k$<span style="font-size: 14px;">), so we drop it too. The resulting discriminant function is:</span>

$$
\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k
$$

<span style="font-size: 14px;">This is linear in</span> $x$<span style="font-size: 14px;">. We can write it as</span> $\delta_k(x) = w_k^T x + b_k$ <span style="font-size: 14px;">where:</span>

- $w_k = \Sigma^{-1} \mu_k$ <span style="font-size: 14px;">(weight vector for class</span> $k$<span style="font-size: 14px;">)</span>
- $b_k = -\frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k$ <span style="font-size: 14px;">(bias for class</span> $k$<span style="font-size: 14px;">)</span>

---

## <span style="font-size: 16px;">Why Decision Boundaries Are Linear</span>

<span style="font-size: 14px;">The decision boundary between class</span> $j$ <span style="font-size: 14px;">and class</span> $k$ <span style="font-size: 14px;">is the set of points where</span> $\delta_j(x) = \delta_k(x)$<span style="font-size: 14px;">:</span>

$$
(w_j - w_k)^T x + (b_j - b_k) = 0
$$

<span style="font-size: 14px;">This is a linear equation in</span> $x$<span style="font-size: 14px;">, defining a hyperplane. When the shared covariance assumption is violated (each class has a different</span> $\Sigma_k$<span style="font-size: 14px;">), the</span> $x^T \Sigma_k^{-1} x$ <span style="font-size: 14px;">terms no longer cancel, producing quadratic boundaries (QDA).</span>

---

## <span style="font-size: 16px;">Parameter Estimation</span>

- <span style="font-size: 14px;">**Class means**:</span> $\hat{\mu}_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i$
- <span style="font-size: 14px;">**Class priors**:</span> $\hat{\pi}_k = n_k / n$
- <span style="font-size: 14px;">**Pooled covariance** (unbiased estimate):</span>

$$
\hat{\Sigma} = \frac{1}{n - K} \sum_{k=1}^{K} \sum_{i: y_i = k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T
$$

<span style="font-size: 14px;">The denominator</span> $n - K$ <span style="font-size: 14px;">accounts for the K parameters (class means) estimated from the data, giving an unbiased estimate. Some implementations use</span> $n$ <span style="font-size: 14px;">instead, which does not affect the classification decision for equal-prior cases but matters when priors are unequal.</span>

---

## <span style="font-size: 16px;">Regularization</span>

<span style="font-size: 14px;">The covariance matrix</span> $\Sigma$ <span style="font-size: 14px;">can be singular or nearly singular when:</span>

- <span style="font-size: 14px;">The number of samples is less than the number of features (</span>$n < d$<span style="font-size: 14px;">)</span>
- <span style="font-size: 14px;">Features are linearly dependent or nearly so</span>
- <span style="font-size: 14px;">A class has very few samples</span>

<span style="font-size: 14px;">The standard fix is Tikhonov regularization:</span>

$$
\hat{\Sigma}_{\text{reg}} = \hat{\Sigma} + \epsilon I
$$

<span style="font-size: 14px;">where</span> $\epsilon$ <span style="font-size: 14px;">is a small constant (e.g.,</span> $10^{-6}$<span style="font-size: 14px;">). This ensures</span> $\Sigma$ <span style="font-size: 14px;">is positive definite and invertible. This approach is sometimes called Regularized Discriminant Analysis (RDA).</span>

---

## <span style="font-size: 16px;">Connection to Fisher's LDA</span>

<span style="font-size: 14px;">Fisher's Linear Discriminant takes a different perspective: find the projection direction(s) that maximize the ratio of between-class scatter to within-class scatter.</span>

- <span style="font-size: 14px;">**Within-class scatter**:</span> $S_W = \sum_k \sum_{i: y_i=k} (x_i - \mu_k)(x_i - \mu_k)^T$
- <span style="font-size: 14px;">**Between-class scatter**:</span> $S_B = \sum_k n_k (\mu_k - \mu)(\mu_k - \mu)^T$
- <span style="font-size: 14px;">**Optimal projection**: eigenvectors of</span> $S_W^{-1} S_B$ <span style="font-size: 14px;">with the largest eigenvalues</span>

<span style="font-size: 14px;">For two classes with equal priors, Fisher's approach and the Bayesian approach produce identical decision boundaries. The Bayesian formulation is more general because it naturally handles unequal priors through the</span> $\log \pi_k$ <span style="font-size: 14px;">term.</span>

---

## <span style="font-size: 16px;">LDA vs. Other Classifiers</span>

- <span style="font-size: 14px;">**LDA vs. Logistic Regression**: LDA is a generative model (models</span> $P(x \mid y)$<span style="font-size: 14px;">, then uses Bayes' rule). Logistic regression is discriminative (directly models</span> $P(y \mid x)$<span style="font-size: 14px;">). When the Gaussian assumption holds, LDA can be more sample-efficient. When it is violated, logistic regression is more robust.</span>
- <span style="font-size: 14px;">**LDA vs. QDA**: LDA uses a shared covariance (linear boundaries). QDA fits a separate covariance per class (quadratic boundaries). QDA has more parameters and can overfit with small data.</span>
- <span style="font-size: 14px;">**LDA vs. PCA**: PCA finds directions of maximum variance (unsupervised). LDA finds directions of maximum class separation (supervised). PCA can spread classes apart or compress them; LDA specifically optimizes for discrimination.</span>
- <span style="font-size: 14px;">**LDA vs. Naive Bayes**: Both are generative classifiers. Naive Bayes assumes feature independence (diagonal covariance). LDA models the full covariance structure. LDA captures feature correlations but needs more data to estimate</span> $\Sigma$ <span style="font-size: 14px;">reliably.</span>

---

## <span style="font-size: 16px;">Computational Considerations</span>

- <span style="font-size: 14px;">**Training**: Computing</span> $\Sigma$ <span style="font-size: 14px;">is</span> $O(n d^2)$<span style="font-size: 14px;">, inverting it is</span> $O(d^3)$<span style="font-size: 14px;">. Total training cost is</span> $O(n d^2 + d^3)$
- <span style="font-size: 14px;">**Prediction**: Each test point requires</span> $O(d K)$ <span style="font-size: 14px;">operations (one dot product per class)</span>
- <span style="font-size: 14px;">**Storage**: The main objects are</span> $\Sigma^{-1}$ <span style="font-size: 14px;">(</span>$d \times d$<span style="font-size: 14px;">), class means (</span>$K \times d$<span style="font-size: 14px;">), and priors (</span>$K$<span style="font-size: 14px;">)</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why is it called "linear"?**</span>
  <span style="font-size: 14px;">A: Because the decision boundaries between any two classes are hyperplanes (linear functions of $x$). This results from the shared covariance assumption, which causes the quadratic terms in the log-likelihood to cancel.</span>

- <span style="font-size: 14px;">**Q: When does LDA fail?**</span>
  <span style="font-size: 14px;">A: When class distributions are highly non-Gaussian, when classes have very different covariance structures, or when decision boundaries are inherently nonlinear.</span>

- <span style="font-size: 14px;">**Q: How does the prior affect classification?**</span>
  <span style="font-size: 14px;">A: The $\log \pi_k$ term shifts the discriminant function. A class with a higher prior gets a larger bias, making the classifier more likely to predict that class when the likelihoods are similar.</span>

- <span style="font-size: 14px;">**Q: What is the relationship between the within-class scatter and the covariance?**</span>
  <span style="font-size: 14px;">A: The within-class scatter $S_W$ equals $(n - K) \Sigma$. They differ only by a scaling factor.</span>

- <span style="font-size: 14px;">**Q: Can LDA be used for dimensionality reduction?**</span>
  <span style="font-size: 14px;">A: Yes. The eigenvectors of $S_W^{-1} S_B$ define the most discriminative projection directions. With $K$ classes, there are at most $K - 1$ meaningful directions.</span>

---