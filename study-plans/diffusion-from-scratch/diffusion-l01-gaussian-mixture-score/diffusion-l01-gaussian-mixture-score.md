# Gaussian Mixture Score

A score attaches an arrow to every point in space, and each arrow points towards where data is more likely, much like a compass that always points uphill on a map of probability. Diffusion models generate data by following arrows like these from noise back to realistic samples, so learning them is the core of the method. A mixture of a few bell curves is the simplest case where the arrows can be written down exactly. This page builds the score step by step, from one Gaussian in one dimension to a weighted mixture evaluated far from every cluster without breaking the arithmetic, which is what you implement here.

## The score of one Gaussian on a line

For a density $p(x)$, the score is the gradient of its log density with respect to the point, $\nabla_x \log p(x)$. In one dimension it is just the slope of $\log p$. Moving a small step along the score increases the log density as fast as possible, so following it leads uphill towards the most likely values.

Start with a single Gaussian with mean $\mu$ and variance $\sigma^2$. Its log density is a constant minus $(x - \mu)^2 / (2\sigma^2)$, and differentiating with respect to $x$ gives

$$
\nabla_x \log \mathcal N(x; \mu, \sigma^2) = \frac{\mu - x}{\sigma^2}
$$

Take $\mu = 2$ and $\sigma^2 = 0.5$. At $x = 1$ the score is $(2 - 1) / 0.5 = 2$, a positive arrow pointing right towards the mean. At $x = 2$ it is 0, because the point already sits on the peak. At $x = 4$ it is $-4$, pointing left and twice as long, since the pull grows linearly with the distance. A narrower Gaussian pulls harder, because the variance divides the distance.

## Moving to many dimensions

In $D$ dimensions the point $x$ and the mean $\mu$ are vectors of length $D$, and the score is a vector of length $D$ as well. For an isotropic Gaussian $\mathcal N(\mu, \sigma^2 I)$, every coordinate has the same variance $\sigma^2$ and the coordinates are independent, where $I$ is the identity matrix. The log density is a constant minus $\lVert x - \mu \rVert^2 / (2\sigma^2)$, so the score is $(\mu - x) / \sigma^2$, a straight pull from the point towards the mean.

## Why the normalizing constant disappears

Many models know a density only up to a constant, $p(x) = \tilde p(x) / Z$, where $Z$ is an integral over the whole space that is usually impossible to compute. Taking the log turns the division into a subtraction of $\log Z$, and $Z$ does not depend on $x$, so its gradient is zero. The score of $p$ therefore equals the score of $\tilde p$.

This is the main reason score based modelling is attractive. A network that outputs a vector field never needs to be normalized, and Hyvarinen (2005) showed that such a field can still be fitted to data through score matching. The normalizer will come back later on this page, though, because inside a mixture each component has its own one.

## A mixture pulls with weighted arrows

A mixture with $K$ components has density $p(x) = \sum_k \pi_k\, \mathcal N(x; \mu_k, \sigma_k^2 I)$. The weights $\pi_k$ are positive and sum to 1, each mean $\mu_k$ is a vector of length $D$, and each variance $\sigma_k^2$ is a single number shared by every coordinate of that component. Differentiating the log of a sum takes three short steps.

1. The gradient of $\log p(x)$ equals $\nabla_x p(x) / p(x)$.
2. Each component contributes $\pi_k \nabla_x \mathcal N_k$, and $\nabla_x \mathcal N_k = \mathcal N_k \cdot (\mu_k - x) / \sigma_k^2$, where $\mathcal N_k$ is short for $\mathcal N(x; \mu_k, \sigma_k^2 I)$.
3. Dividing by $p(x)$ turns $\pi_k \mathcal N_k / p(x)$ into a number $r_k(x)$ called the responsibility of component $k$.

The responsibility is the posterior probability that component $k$ produced the point $x$, so the responsibilities at one point are positive and sum to 1. The score becomes

$$
\nabla_x \log p(x) = \sum_{k=1}^{K} r_k(x)\, \frac{\mu_k - x}{\sigma_k^2}
$$

Each component pulls the point towards its own mean with its own single-Gaussian score, and the pulls are averaged with weights that say how plausibly each component explains the point.

## A two-component example

Take a one-dimensional mixture with weights $0.25$ and $0.75$, means $-1$ and $3$, and variances 1 and 4, and evaluate it at $x = 1$, exactly halfway between the means. Write $\ell_k$ for the log of the weighted density $\pi_k \mathcal N_k$. It has three parts.

1. The log weights are $\log 0.25 \approx -1.3863$ and $\log 0.75 \approx -0.2877$.
2. The normalizers $-\tfrac{1}{2}\log(2\pi\sigma_k^2)$ are about $-0.9189$ and $-1.6121$.
3. The distance terms $-(x - \mu_k)^2 / (2\sigma_k^2)$ are $-2$ and $-0.5$.

The totals are $\ell_1 \approx -4.3052$ and $\ell_2 \approx -2.3998$. Normalizing their exponentials gives responsibilities of about $0.1295$ and $0.8705$. The single-component scores are $(-1 - 1) / 1 = -2$ and $(3 - 1) / 4 = 0.5$, so the mixture score is $0.1295 \cdot (-2) + 0.8705 \cdot 0.5 \approx 0.1763$.

The point is equally far from both means, yet the wide and more common second component claims most of the responsibility, and the arrow points gently towards it.

## Why each component keeps its normalizer

The normalizer did not matter for one Gaussian, because it was a constant. In a mixture it changes the responsibilities whenever the components have different variances. A narrow component has a taller peak, and its normalizer is what records that extra height.

Dropping the normalizers in the example above changes the log terms to about $-3.3863$ and $-0.7877$, which gives responsibilities of about $0.0692$ and $0.9308$ and a score of about $0.3269$, nearly double the correct value. The normalizer also scales with the dimension. A $D$ dimensional isotropic Gaussian is a product of $D$ independent one-dimensional factors, each with its own normalizer, so the full term is $-\tfrac{D}{2}\log(2\pi\sigma_k^2)$.

## Computing responsibilities without underflow

Far from every mean the densities are tiny. For the same mixture at $x = 100$, the log terms are about $-5102.81$ and $-1178.02$. The exponential of anything below about $-745$ is exactly 0 in float64, so both weighted densities become 0, and the responsibility is the undefined ratio 0/0.

The cure is the log-sum-exp trick. Subtracting the largest log term $m$ from every term leaves the responsibilities unchanged, because the factor $e^{-m}$ appears in both the numerator and the denominator and cancels.

$$
r_k(x) = \frac{\exp(\ell_k - m)}{\sum_j \exp(\ell_j - m)}, \qquad m = \max_j \ell_j
$$

After the shift the largest term is 0, so at least one exponential equals 1 and the denominator can never be zero. At $x = 100$ the shifted terms are about $-3924.78$ and $0$, the responsibilities are 0 and 1, and the score is the pull of the second component alone, $(3 - 100) / 4 = -24.25$.

## Where this sits in diffusion models

Song and Ermon (2019) and Song et al. (2021) frame generation as learning the score of the data after it has been blurred by Gaussian noise of many sizes. A noise prediction network trained as in Ho et al. (2020) is a scaled score estimate, since $\nabla_x \log p_t(x_t) \approx -\hat\epsilon / \sqrt{1 - \bar\alpha_t}$. Here $\hat\epsilon$ is the predicted noise and $\bar\alpha_t$ is the fraction of signal variance kept up to timestep $t$.

When the data is a Gaussian mixture, every noisy version of it is again a Gaussian mixture, with each mean scaled to $\sqrt{\bar\alpha_t}\, \mu_k$ and each variance replaced by $\bar\alpha_t \sigma_k^2 + 1 - \bar\alpha_t$. The exact score from this page therefore serves as ground truth for checking samplers and training losses on toy data.

## Common mistakes to avoid

- **Exponentiating the log terms directly.** Far from the means every density underflows to zero, and the responsibilities become 0/0, which is NaN.
- **Dropping the normalizing term of each component.** With different variances, a narrow component loses the extra height of its peak and receives too little responsibility near its own mean.
- **Using the normalizer of one coordinate.** The term must be multiplied by D, otherwise wide components are penalized too little in more than one dimension.
- **Ignoring the mixture weights.** A rare component then receives as much responsibility as a common one, which shifts the score towards it.
- **Using x minus the mean instead of the mean minus x.** The resulting field points away from the data, so following it drives samples out to infinity.
- **Forgetting to divide by each component's variance.** The pull of each component then ignores its width, so narrow and wide clusters attract equally hard.