# <span style="font-size: 20px;">Per-Channel Mean and Std</span>

<span style="font-size: 14px;">Per-channel statistics summarize an image dataset by reducing it to one mean and one standard deviation per colour channel. These two numbers per channel are exactly the constants that drive **input normalization**: the ImageNet mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$ were produced by this very computation run over the full ImageNet training set.</span>

---

## <span style="font-size: 16px;">What Is Being Computed</span>

<span style="font-size: 14px;">Given a batch of images stored as a tensor $X$ of shape $(N, H, W, C)$, the goal is, for each channel $c$, the mean and standard deviation of **every pixel value in that channel across the entire batch**. The reduction runs over three axes at once - the batch index $n$, the row $h$, and the column $w$ - leaving a result of length $C$.</span>

<span style="font-size: 14px;">Concretely, channel $c$ contains $N \cdot H \cdot W$ scalar values (one per pixel per image). The mean is their average and the std is the spread of those same values. The spatial layout is irrelevant for this particular statistic: the pixels are treated as an unordered bag of numbers per channel.</span>

---

## <span style="font-size: 16px;">The Formulas</span>

<span style="font-size: 14px;">Let $M = N \cdot H \cdot W$ be the number of pixels contributing to each channel. The per-channel mean is:</span>

$$
\mu_c = \frac{1}{M} \sum_{n, h, w} X[n, h, w, c]
$$

<span style="font-size: 14px;">and the per-channel population standard deviation is the square root of the average squared deviation from that mean:</span>

$$
\sigma_c = \sqrt{\frac{1}{M} \sum_{n, h, w} \bigl(X[n, h, w, c] - \mu_c\bigr)^2}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$M = N \cdot H \cdot W$ is the count of values per channel</span>
* <span style="font-size: 14px;">the inner expression $X[n,h,w,c] - \mu_c$ is the deviation of one pixel from its channel mean</span>
* <span style="font-size: 14px;">the variance $\sigma_c^2$ is the average of those squared deviations; the std is its square root</span>

<span style="font-size: 14px;">The result is two length-$C$ vectors, returned here as a dict `{"mean": [...], "std": [...]}` with one entry per channel and each value rounded to 4 decimals.</span>

---

## <span style="font-size: 16px;">Population vs Sample (the divisor)</span>

<span style="font-size: 14px;">There are two conventions for the denominator of the variance, and the difference is the single most common source of mismatched answers:</span>

* <span style="font-size: 14px;">**Population variance** divides the sum of squared deviations by $M$. This treats the data as the entire population of interest.</span>
* <span style="font-size: 14px;">**Sample variance** divides by $M - 1$ (Bessel's correction). This is an unbiased estimator of the variance of a larger population from which the data is a sample.</span>

<span style="font-size: 14px;">This problem requires the **population** estimate: divide by $M = N \cdot H \cdot W$, not $M - 1$. The distinction is enormous for tiny $M$ (dividing by $1$ vs $2$ doubles the variance) but negligible for the millions of pixels in a real dataset. The reason to fix the convention explicitly is that libraries disagree by default: NumPy's `np.std` uses `ddof=0` (population), but pandas `.std()` and PyTorch `torch.std` default to `ddof=1` (sample). To match the population formula in PyTorch, pass `unbiased=False`.</span>

---

## <span style="font-size: 16px;">The Reduction in Detail</span>

<span style="font-size: 14px;">For a channels-last tensor of shape $(N, H, W, C)$, the reduction collapses axes $0, 1, 2$ and keeps axis $3$. In NumPy: `mean = X.mean(axis=(0,1,2))` gives a length-$C$ vector, and `std = X.std(axis=(0,1,2))` gives the population std because NumPy defaults to `ddof=0`.</span>

<span style="font-size: 14px;">In PyTorch with the common channels-first layout $(N, C, H, W)$, the reduction keeps axis $1$ instead: `X.mean(dim=(0,2,3))` and `X.std(dim=(0,2,3), unbiased=False)`. The axis bookkeeping is the part most likely to go wrong; the arithmetic is simple. Whatever the layout, the channel axis is the one axis that is **not** reduced.</span>

---

## <span style="font-size: 16px;">What the Mean and Std Tell You</span>

<span style="font-size: 14px;">The two statistics capture complementary aspects of a channel. The mean $\mu_c$ measures the channel's average brightness across the dataset: a red channel mean near $0.49$ means natural images are, on average, slightly less than half-saturated in red. The standard deviation $\sigma_c$ measures contrast or spread: how far typical pixels stray from that average. A channel with high $\sigma_c$ has strong light-dark variation, while a low $\sigma_c$ channel is flat and washed out.</span>

<span style="font-size: 14px;">For ImageNet the three means cluster around $0.45$ and the three stds around $0.22$, which says natural photographs are moderately bright and have similar contrast in all three colour channels. These numbers are not universal: a dataset of night-time scenes would have far lower means, and a dataset of high-contrast line drawings would have larger stds. That is precisely why custom datasets need their own statistics rather than borrowing ImageNet's.</span>

---

## <span style="font-size: 16px;">Worked Example (one channel, tiny batch)</span>

<span style="font-size: 14px;">Take a single channel ($C = 1$) over one $2 \times 2$ image, so $M = 1 \cdot 2 \cdot 2 = 4$, with pixel values $\begin{pmatrix} 2 & 4 \\ 6 & 8 \end{pmatrix}$.</span>

<span style="font-size: 14px;">1. **Mean**: $\mu = (2 + 4 + 6 + 8)/4 = 20/4 = 5$</span>

<span style="font-size: 14px;">2. **Squared deviations**: $(2-5)^2 = 9$, $(4-5)^2 = 1$, $(6-5)^2 = 1$, $(8-5)^2 = 9$</span>

<span style="font-size: 14px;">3. **Variance (population, divide by 4)**: $\sigma^2 = (9 + 1 + 1 + 9)/4 = 20/4 = 5$</span>

<span style="font-size: 14px;">4. **Std**: $\sigma = \sqrt{5} = 2.2361$</span>

<span style="font-size: 14px;">If sample variance had been used by mistake, step 3 would divide by $M - 1 = 3$, giving $\sigma^2 = 20/3 = 6.667$ and $\sigma = 2.582$. The gap between $2.2361$ and $2.582$ is the entire difference between population and sample, and it is large here only because $M = 4$ is tiny. For a real dataset with $M$ in the millions the two estimates would agree to far more than 4 decimal places, but the rounding requested here can still surface the discrepancy on small test inputs, so matching the population convention exactly is what makes the answer correct.</span>

---

## <span style="font-size: 16px;">Worked Example (two channels)</span>

<span style="font-size: 14px;">Now a batch of $N = 1$ image, shape $(1, 2, 2, 2)$, so $M = 4$ per channel. Suppose channel 0 has values $\{0.0, 0.2, 0.4, 0.6\}$ and channel 1 has values $\{0.5, 0.5, 0.7, 0.9\}$ across the four spatial locations.</span>

* <span style="font-size: 14px;">**Channel 0 mean**: $(0 + 0.2 + 0.4 + 0.6)/4 = 0.3$</span>
* <span style="font-size: 14px;">**Channel 0 variance**: $((0.09) + (0.01) + (0.01) + (0.09))/4 = 0.2/4 = 0.05$, so $\sigma_0 = \sqrt{0.05} = 0.2236$</span>
* <span style="font-size: 14px;">**Channel 1 mean**: $(0.5 + 0.5 + 0.7 + 0.9)/4 = 2.6/4 = 0.65$</span>
* <span style="font-size: 14px;">**Channel 1 variance**: $((0.0225) + (0.0225) + (0.0025) + (0.0625))/4 = 0.11/4 = 0.0275$, so $\sigma_1 = \sqrt{0.0275} = 0.1658$</span>

<span style="font-size: 14px;">The returned dict is `{"mean": [0.3, 0.65], "std": [0.2236, 0.1658]}`. Each channel is reduced independently; channel 1's values are bunched more tightly, so its std is smaller despite its higher mean. This independence is the whole point: a red-heavy dataset and a blue-heavy dataset can share the same overall brightness yet need very different per-channel constants, and only a per-channel reduction captures that.</span>

---

## <span style="font-size: 16px;">Two-Pass vs One-Pass Computation</span>

<span style="font-size: 14px;">The formulas above describe a **two-pass** algorithm: pass one computes $\mu_c$, pass two accumulates squared deviations from that mean. There is also a **one-pass** identity using the mean of squares:</span>

$$
\sigma_c^2 = \frac{1}{M}\sum X^2 - \mu_c^2 = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

<span style="font-size: 14px;">The one-pass form accumulates $\sum X$ and $\sum X^2$ together, then combines them at the end. It is convenient for streaming over a dataset too large to hold in memory, since it never needs the data twice. The catch is numerical: subtracting two large nearly-equal quantities ($\mathbb{E}[X^2]$ and $\mu_c^2$) can lose precision and even produce a tiny negative variance for low-variance channels. The two-pass method, or Welford's online algorithm, is more numerically stable and preferred when accuracy matters.</span>

---

## <span style="font-size: 16px;">Streaming Over a Large Dataset</span>

<span style="font-size: 14px;">Real datasets are far too large to load into memory at once, so the statistics are accumulated incrementally. The simplest streaming scheme keeps three running totals per channel: the count $M$, the sum $\sum X$, and the sum of squares $\sum X^2$. Each batch of images adds its contribution to all three, and at the end $\mu_c = (\sum X)/M$ and $\sigma_c^2 = (\sum X^2)/M - \mu_c^2$.</span>

<span style="font-size: 14px;">This works but inherits the cancellation problem noted above. Welford's algorithm avoids it by maintaining the running mean and a running sum of squared deviations from that evolving mean, updating both in a numerically stable way as each value arrives. For batched image data, a common compromise is to compute exact per-batch means and variances and combine them with the parallel (Chan et al.) variance-merging formula, which is both accurate and easy to vectorize. The arithmetic identity is the same; only the order and grouping of operations change to preserve precision.</span>

---

## <span style="font-size: 16px;">Why This Is Computed Over the Whole Dataset</span>

<span style="font-size: 14px;">The purpose of these statistics is to feed input normalization $(x - \mu_c)/\sigma_c$. For that to be meaningful, $\mu_c$ and $\sigma_c$ must describe the dataset the model will see, so they are computed once over the entire training set and then frozen. This is why the formula reduces over the batch axis $N$ as well as the spatial axes: a single image's statistics are too noisy, and the goal is the population-level brightness and contrast of each channel.</span>

<span style="font-size: 14px;">Computing them on the training split only is essential; deriving them from validation or test data leaks information about the evaluation set into preprocessing. Once computed, the same two vectors are reused unchanged at train, validation, and test time.</span>

<span style="font-size: 14px;">There is a chicken-and-egg relationship between this problem and the normalization problem. Per-channel statistics are the **producer** of the constants; normalization is the **consumer**. The output dict `{"mean": [...], "std": [...]}` here is exactly the pair of lists that would be passed to `transforms.Normalize(mean, std)`. Understanding the producer side clarifies why the consumer constants look the way they do and why they must never be recomputed per image at inference time.</span>

<span style="font-size: 14px;">A practical detail: the statistics should be computed on the same value scale the model will use. If the pipeline divides pixels by $255$ to reach $[0, 1]$ before training, then the statistics must also be computed on $[0, 1]$ values, not raw $[0, 255]$ ones. Computing on one scale and normalizing on another is a silent and common bug.</span>

---

## <span style="font-size: 16px;">Complexity and Variants</span>

<span style="font-size: 14px;">The computation touches every pixel a constant number of times, so the cost is $O(N \cdot H \cdot W \cdot C)$ for both mean and variance, with $O(C)$ memory for the running totals. This is the cheapest possible non-trivial reduction over the data and is trivially parallel: each channel and each spatial location can be summed independently before combining.</span>

<span style="font-size: 14px;">Related statistics build on the same reduction. **Per-channel min and max** are used for min-max scaling instead of standardization. **Per-channel histograms** support histogram equalization and exposure analysis. The **full channel covariance matrix** $\Sigma$, an extension that does not reduce away cross-channel structure, is the basis for whitening. All of these share the pattern of reducing over $(N, H, W)$ while preserving channel identity; the mean and std are simply the first two moments and by far the most commonly needed.</span>

<span style="font-size: 14px;">When images in a dataset have different spatial sizes, $M$ is not constant per image, and the correct global mean is the total sum of all pixels divided by the total pixel count, not the average of per-image means. Averaging per-image means over-weights small images. The streaming sum-and-count approach handles this automatically because it accumulates raw counts rather than per-image averages.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Wrong divisor (population vs sample).** Using $M - 1$ when $M$ is required, or relying on a library default that disagrees with the spec, yields a std that is slightly too large. In PyTorch remember to pass `unbiased=False`; in NumPy the default `ddof=0` is already correct.</span>
* <span style="font-size: 14px;">**Reducing the wrong axes.** The channel axis must be preserved while batch and spatial axes are collapsed. Accidentally including the channel axis in the reduction produces a single scalar instead of a length-$C$ vector; mixing up channels-first and channels-last reduces over the wrong dimensions.</span>
* <span style="font-size: 14px;">**Integer dtype overflow in the sum.** Summing millions of `uint8` pixels overflows the byte range and wraps around to a wrong mean. Accumulate in `float64` (or at least `float32`) before dividing, since the running total can vastly exceed the per-pixel range.</span>
* <span style="font-size: 14px;">**Negative variance from the one-pass identity.** $\mathbb{E}[X^2] - \mu_c^2$ can return a small negative number due to floating-point cancellation, and taking its square root yields NaN. Clamp the variance at zero or use a stable two-pass method.</span>
* <span style="font-size: 14px;">**Averaging per-image statistics instead of pooling pixels.** When images differ in size, taking the mean of each image's mean over-weights small images. The correct global statistic pools all pixels with their true counts, which the sum-and-count accumulation does automatically.</span>

---