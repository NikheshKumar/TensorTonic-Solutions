# Diagonal Gaussian KL Divergence

The KL divergence measures how surprised you would be, on average, if you expected data from one bell curve and it actually came from another. Diffusion models and variational autoencoders are trained by keeping surprises like this small, and many terms of their training objectives compare two bell curves in exactly this way. This page builds the divergence step by step, starting from two one-dimensional Gaussians that differ only in their means, then letting their spreads differ, and finally summing over many independent coordinates for every row of a batch, which is what you implement here.

## Surprise as a log ratio of densities

Suppose a model believes data follows a density $q$, but the data really follows a density $p$. A natural measure of how wrong the model is at one point $x$ is the log ratio $\log p(x) - \log q(x)$. It is positive where $p$ gives the point more probability than $q$ does, and negative where $q$ gives it more.

Averaging this log ratio over points drawn from $p$ gives the KL divergence from $p$ to $q$.

$$
\mathrm{KL}(p \,\|\, q) = \mathbb E_{x \sim p}\left[\log p(x) - \log q(x)\right]
$$

The divergence is zero exactly when the two distributions are equal and positive otherwise. It can be read as the extra cost of describing samples from $p$ with a model built for $q$. The average is taken under $p$ alone, which makes the divergence asymmetric. A region where $p$ has mass and $q$ has almost none is penalized heavily, while the reverse situation costs little.

## Two Gaussians that differ only in their means

Start with the simplest pair. Let $p = \mathcal N(\mu_p, \sigma^2)$ and $q = \mathcal N(\mu_q, \sigma^2)$ share the same variance $\sigma^2$ and differ only in their means $\mu_p$ and $\mu_q$. The log density of a Gaussian is a normalizing constant minus $(x - \mu)^2 / (2\sigma^2)$, and with equal variances the constants cancel in the log ratio.

What remains is the difference of two squared distances, and averaging it under $p$ leaves only the mean gap.

$$
\mathrm{KL}(p \,\|\, q) = \frac{(\mu_p - \mu_q)^2}{2\sigma^2}
$$

The gap is measured in units of the variance. Take $\mu_p = 0.6$, $\mu_q = 0$ and $\sigma^2 = 0.36$. The squared gap is $0.36$, and dividing by $2 \cdot 0.36 = 0.72$ gives a divergence of $0.5$. The same gap between two much narrower Gaussians would give a far larger divergence, because narrow curves barely overlap when their centres are apart.

## Letting the spreads differ

Now let the variances differ, so $p = \mathcal N(\mu_p, \sigma_p^2)$ and $q = \mathcal N(\mu_q, \sigma_q^2)$. The log density of a Gaussian is $-\tfrac{1}{2}\log(2\pi\sigma^2) - (x - \mu)^2 / (2\sigma^2)$, and the average under $p$ can be taken term by term.

1. The constants $\log 2\pi$ cancel, and the normalizers leave $\tfrac{1}{2}(\log \sigma_q^2 - \log \sigma_p^2)$.
2. Under $p$, the average of $(x - \mu_p)^2$ is $\sigma_p^2$, so the squared term of $\log p$ contributes exactly $-\tfrac{1}{2}$.
3. Writing $x - \mu_q = (x - \mu_p) + (\mu_p - \mu_q)$ and averaging under $p$ gives $\sigma_p^2 + (\mu_p - \mu_q)^2$, so the squared term of $\log q$ contributes $(\sigma_p^2 + (\mu_p - \mu_q)^2) / (2\sigma_q^2)$.

Adding the three pieces gives the one-dimensional formula.

$$
\mathrm{KL}(p \,\|\, q) = \frac{1}{2}\left(\log \sigma_q^2 - \log \sigma_p^2 + \frac{\sigma_p^2 + (\mu_p - \mu_q)^2}{\sigma_q^2} - 1\right)
$$

Each term has a job. The log difference and the variance ratio compare the two spreads, the mean gap is measured in units of the variance of $q$, and the minus one makes the whole expression vanish when the Gaussians coincide. Setting $\sigma_p^2 = \sigma_q^2$ makes the log difference zero and the variance ratio one, so the formula falls back to the mean-only case from the previous section.

## A worked example that shows the asymmetry

Take $p = \mathcal N(1, 4)$ and $q = \mathcal N(0, 1)$, so $p$ is wide and $q$ is narrow. The log difference is $\log 1 - \log 4 \approx -1.3863$, the variance ratio is $4 / 1 = 4$, the scaled mean gap is $1 / 1 = 1$, and the constant is $-1$. Their sum is about $2.6137$, so $\mathrm{KL}(p \,\|\, q) \approx 1.3069$.

Swap the roles and compute $\mathrm{KL}(q \,\|\, p)$. The log difference is now $\log 4 \approx 1.3863$, the variance ratio is $1 / 4 = 0.25$, the scaled mean gap is $1 / 4 = 0.25$, and the constant is $-1$. Their sum is about $0.8863$, so the divergence is about $0.4431$, roughly a third of the other direction.

The wide $p$ puts plenty of mass in regions where the narrow $q$ is almost zero, and the divergence punishes that strongly. The narrow $q$ stays well inside the wide $p$, so measuring in that direction costs much less. This is why the order of the two arguments always matters.

## Summing over independent coordinates

A diagonal Gaussian in $D$ dimensions has a mean vector of length $D$ and a variance for each coordinate, with no correlation between coordinates. Its density is a product of $D$ one-dimensional densities, so its log density is a sum, and the divergence between two diagonal Gaussians is the sum of the $D$ one-dimensional divergences. Writing $d$ for the coordinate index,

$$
\mathrm{KL}(p \,\|\, q) = \sum_{d=1}^{D} \mathrm{KL}\big(\mathcal N(\mu_{p,d}, \sigma_{p,d}^2) \,\|\, \mathcal N(\mu_{q,d}, \sigma_{q,d}^2)\big)
$$

Extending the example, suppose $p$ and $q$ gain a second coordinate in which both are the standard Gaussian $\mathcal N(0, 1)$. That coordinate contributes zero, so the two-dimensional divergence is still about $1.3069$. If the second coordinate instead used the first pair again, the total would double to about $2.6137$.

In a batch, the inputs have shape $(\dots, D)$. Every leading index holds a separate pair of Gaussians, and only the last axis is summed, so inputs of shape $(N, D)$ give $N$ divergences.

## Working directly with log variances

The inputs give log variances, $\log \sigma_p^2$ and $\log \sigma_q^2$, and the formula fits that form well. The log term is already a plain difference of the inputs. The variance ratio $\sigma_p^2 / \sigma_q^2$ equals $\exp(\log \sigma_p^2 - \log \sigma_q^2)$, which avoids dividing two numbers that might both be very large or very small. Only the mean term needs $\sigma_q^2$ on its own, which is $\exp(\log \sigma_q^2)$.

## Where KL terms appear in diffusion training

Ho et al. (2020) train a diffusion model by maximizing a variational lower bound on the log likelihood. The bound splits into one KL divergence per timestep, $\mathrm{KL}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))$. Here $q(x_{t-1} \mid x_t, x_0)$ is the exact Gaussian posterior of the forward process, $p_\theta$ is the Gaussian reverse step predicted by the network, and both are diagonal Gaussians, so every term is an instance of this formula.

When both Gaussians share a fixed variance $\sigma_t^2$, only the mean term survives, as in the mean-only section. A mean gap of $(0.2, -0.4)$ with $\sigma_t^2 = 0.04$ gives $(0.04 + 0.16) / 0.08 = 2.5$. This is how the bound turns into the familiar squared error losses on predicted means or predicted noise.

The last term of the bound compares $q(x_T \mid x_0)$ with the standard Gaussian prior. With $\bar\alpha = 4 \times 10^{-5}$ at the final step and a pixel value $x_0 = 1$, that term is only about $2.0 \times 10^{-5}$, which shows why a good schedule makes the final state indistinguishable from pure noise. Models that learn their variances, such as Nichol and Dhariwal (2021), keep the full formula, because the variance terms no longer cancel.

## Numerical behaviour

All terms are finite for finite log variances, and the exact result is never negative. When $p$ and $q$ are nearly identical, the four terms almost cancel, so the computed value can be a tiny negative number of order $10^{-16}$ from rounding. A very small $\sigma_q^2$ makes the mean term large, because a confident $q$ that misses the mean of $p$ is strongly penalized.

## Common mistakes to avoid

- **Swapping p and q.** The divergence is asymmetric, so dividing by the variance of p instead of q measures the reverse direction and gives a different number.
- **Averaging over coordinates instead of summing.** Independent coordinates add their divergences, and a mean shrinks the result by a factor of D.
- **Dropping the factor one half.** Every term doubles, which doubles the loss and upsets its balance with the other terms of a bound.
- **Forgetting the constant minus one.** Identical Gaussians then report a divergence of D/2 instead of zero.
- **Leaving the squared mean gap unscaled.** The gap must be divided by the variance of q, otherwise a confident q that misses the mean is barely penalized.
- **Treating log variances as variances.** Plugging log_var_p straight into the ratio gives negative or meaningless spreads, so each variance must be recovered with an exponential first.
- **Reducing over every axis.** Summing the whole array returns one number for the batch instead of one divergence per leading index.