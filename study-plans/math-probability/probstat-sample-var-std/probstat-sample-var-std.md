## <span style="font-size: 20px;">Sample Variance and Standard Deviation</span>

Understanding the distinction between population and sample statistics is fundamental to statistical inference.

---

## Definitions and Formulas

**Population variance** uses all $N$ data points and divides by $N$:

$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

**Sample variance** corrects for bias using Bessel's correction, dividing by $n-1$:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Sample standard deviation** is the square root of the sample variance:

$$s = \sqrt{s^2}$$

---

## Why Bessel's Correction?

When estimating population variance from a sample, using $n$ as the divisor systematically underestimates the true variance. This happens because the sample mean $\bar{x}$ is closer to the sample points than the true population mean $\mu$.

Dividing by $n-1$ instead of $n$ produces an **unbiased estimator** - meaning the expected value of the sample variance equals the population variance:

$$E[s^2] = \sigma^2$$

The quantity $n-1$ is called the **degrees of freedom**: once we compute the sample mean from $n$ observations, only $n-1$ of the deviations from the mean are independent.

---

## Key Properties

- Sample variance is always non-negative: $s^2 \geq 0$
- For a constant dataset (all values equal), $s^2 = 0$
- Standard deviation has the same units as the original data
- Variance has squared units, making it less interpretable but mathematically convenient