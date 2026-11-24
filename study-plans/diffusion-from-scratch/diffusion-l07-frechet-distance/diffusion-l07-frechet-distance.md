# Frechet Distance Between Feature Sets

The Frechet distance gives one number for how different two large collections of feature vectors are. It is like comparing two crowds by their average height and by how spread out they are, instead of matching people one by one. Diffusion models need a measure like this because a generated image has no single correct answer, so quality is judged by how closely many generated samples resemble many real ones as a group. This page builds the distance one piece at a time, starting from two small sets of single numbers and ending with the full formula for D-dimensional features, including the matrix square root that has to be computed with care.

## Why generated samples are judged as a group

A generator can only be judged by comparing many of its samples with many real samples. Comparing raw pixels is a poor proxy for quality, so the standard recipe first passes every image through a fixed pretrained network and keeps one feature vector per image. Good samples should produce feature vectors that are spread out like the real ones, with the same typical values and the same correlations between features.

The cheapest summary of a cloud of vectors that captures both properties is its mean and its covariance. The Frechet distance treats each cloud as the Gaussian with those two statistics and measures how far apart the two Gaussians are. Applied to features from the Inception network, it is the Frechet Inception Distance, or FID, the standard number reported for image generators including diffusion models (Heusel et al., 2017).

## Summarizing a set of single numbers

Start with $D = 1$, so every row of a feature set is a single number. Take set a with values $0, 2, 4$ and set b with values $2, 3, 4$. The mean of a set is its average, so $\mu_a = 2$ and $\mu_b = 3$.

The spread is measured by the sample variance, the sum of squared distances from the mean divided by $N - 1$, where $N$ is the number of rows. For set a the squared distances are $4, 0, 4$, so its variance is $8 / 2 = 4$ and its standard deviation is $\sigma_a = 2$. For set b the squared distances are $1, 0, 1$, so its variance is $1$ and $\sigma_b = 1$.

The divisor $N - 1$ corrects a bias. The mean is estimated from the same rows, which makes the squared distances slightly too small on average, and dividing by one fewer row compensates. Dividing by $N$ would give a variance of about $2.6667$ for set a instead of 4, and with only a few rows that difference is large. The original FID code uses $N - 1$, and so does this problem.

## The distance between two Gaussians in one dimension

With a mean and a standard deviation for each set, the squared Frechet distance between the two one-dimensional Gaussians is

$$
d^2 = (\mu_a - \mu_b)^2 + (\sigma_a - \sigma_b)^2 .
$$

The first term penalizes a shift of the whole set, and the second penalizes a difference in spread. For the example sets this gives $(2 - 3)^2 + (2 - 1)^2 = 2$. With the biased divisor the standard deviations would be about $1.633$ and $0.8165$, and the distance would drop to about $1.6667$.

Expanding the square shows the form that generalizes to many features.

$$
d^2 = (\mu_a - \mu_b)^2 + \sigma_a^2 + \sigma_b^2 - 2 \sqrt{\sigma_a^2\, \sigma_b^2}
$$

Here the spread appears through the two variances and a cross term, the square root of their product. In higher dimensions each variance becomes a covariance matrix and the cross term becomes the square root of a matrix product.

## Many features bring covariance matrices

Each row of features_a and features_b is now a vector of length D. The mean $\mu_a$ is the average row, a vector of shape $(D)$. The covariance $\Sigma_a$ is a $(D, D)$ matrix whose diagonal holds the variance of each feature and whose off-diagonal entries hold how pairs of features vary together, all with the divisor $N_a - 1$ where $N_a$ is the number of rows of set a. The same definitions give $\mu_b$ and $\Sigma_b$, and the two sets may have different numbers of rows. The squared distance is

$$
d^2 = \lVert \mu_a - \mu_b \rVert^2 + \operatorname{Tr}(\Sigma_a) + \operatorname{Tr}(\Sigma_b) - 2 \operatorname{Tr}\big( (\Sigma_a \Sigma_b)^{1/2} \big),
$$

where $\operatorname{Tr}$ is the trace, the sum of the diagonal entries. This is also the squared 2-Wasserstein distance between the two Gaussians. The result is zero only when the means and covariances agree.

When both covariances are diagonal, every term splits feature by feature, and $d^2$ is the sum of one-dimensional distances like the one above. Correlations are what make the general case harder. Two clouds can have the same variance on every feature and still be tilted in different directions, and only the cross term notices.

## The cross term needs a true matrix square root

A matrix square root of $A$ is a matrix $R$ with $R R = A$. It is unrelated to taking the square root of each entry. For a symmetric matrix with eigendecomposition $A = V \Lambda V^\top$, where $V$ holds orthonormal eigenvectors and $\Lambda$ is the diagonal of eigenvalues, the symmetric square root is $V \Lambda^{1/2} V^\top$. The trace of a square root is the sum of the square roots of the eigenvalues.

Each covariance is symmetric with eigenvalues that are zero or positive, but the product $\Sigma_a \Sigma_b$ is generally not symmetric. A general eigenvalue or square root routine applied to a non-symmetric matrix works in complex arithmetic, and rounding then leaves tiny imaginary parts or small negative eigenvalues, even though the exact answer is real. A similar symmetric matrix avoids this.

1. Let $\Sigma_a^{1/2}$ be the symmetric square root of $\Sigma_a$, and define $M = \Sigma_a^{1/2} \Sigma_b \Sigma_a^{1/2}$.
2. The matrix $M$ is symmetric, and $x^\top M x = (\Sigma_a^{1/2} x)^\top \Sigma_b (\Sigma_a^{1/2} x) \ge 0$ for every $x$, so its eigenvalues are real and nonnegative.
3. For any two square matrices, $A B$ and $B A$ have the same eigenvalues. Choosing $A = \Sigma_a^{1/2}$ and $B = \Sigma_a^{1/2} \Sigma_b$ gives $A B = \Sigma_a \Sigma_b$ and $B A = M$.
4. The cross term is therefore $\sum_i \sqrt{\lambda_i}$, where the $\lambda_i$ are the eigenvalues of $M$.

This holds even when $\Sigma_a$ is singular, since step 3 never needs an inverse. A symmetric eigenvalue routine can still return a value such as $-10^{-17}$ where the exact value is 0, so negative eigenvalues are clipped to 0 before the square root.

## A two-dimensional worked example

Take $\Sigma_a = \operatorname{diag}(2, 0.5)$, a $\Sigma_b$ with 1 on the diagonal and $0.5$ off the diagonal, and a mean difference of $(1, 0)$.

1. The product $\Sigma_a \Sigma_b$ has rows $(2, 1)$ and $(0.25, 0.5)$, which is visibly not symmetric.
2. Since $\Sigma_a$ is diagonal, its square root is $\operatorname{diag}(\sqrt 2, \sqrt{0.5})$, and $M$ has rows $(2, 0.5)$ and $(0.5, 0.5)$.
3. The eigenvalues of $M$ are about $0.348612$ and $2.151388$, the same as those of the product.
4. Their square roots are about $0.590434$ and $1.466761$, so the cross term is about $2.057195$.
5. With $\lVert \mu_a - \mu_b \rVert^2 = 1$, $\operatorname{Tr}(\Sigma_a) = 2.5$ and $\operatorname{Tr}(\Sigma_b) = 2$, the result is $1 + 2.5 + 2 - 2 \cdot 2.057195 \approx 1.385610$.

Taking the square root of each entry of the product instead would give a trace of about $2.1213$ and the wrong distance.

## Rounding, singular sets and zero distance

Two identical sets give a distance of zero up to rounding, so a result like $10^{-15}$ or a tiny negative number is normal. A set whose rows are all equal has a zero covariance, its square root is the zero matrix, and only the mean term and the other trace survive. A constant feature or two identical features make a covariance singular, which the symmetric route handles because it never inverts anything.

The estimate is biased even with the right divisor, because the square root is a nonlinear function of a noisy covariance. Published FID numbers therefore state the number of samples, commonly 50,000, and scores computed from different sample counts are not comparable.

## Where FID sits in the diffusion pipeline

FID is computed after training, on samples drawn with a full sampler such as DDPM or DDIM. It is used to compare architectures, noise schedules, step counts and guidance scales, and the DiT paper reports it for every model size (Peebles and Xie, 2023). A lower score means the generated feature statistics are closer to the real ones.

## Common mistakes to avoid

- **Dividing the covariance by N instead of N - 1.** Every covariance shrinks by a factor of $(N - 1) / N$, which shifts the score noticeably for small sets.
- **Taking the eigenvalues of the non-symmetric product directly.** Rounding produces complex values or negative eigenvalues, which give a complex number or NaN in place of a real distance.
- **Using an elementwise square root.** The square root of each entry is unrelated to the matrix square root whenever the matrix is not diagonal.
- **Keeping only the variance of each feature.** Dropping the off-diagonal covariances ignores correlations, so two clouds with the same variances and different orientations look identical.
- **Skipping the clip before the square root.** A rounding error of $-10^{-17}$ in one eigenvalue turns the whole result into NaN.
- **Returning the square root of the result.** The FID convention reports the squared distance, so a final square root changes every score.