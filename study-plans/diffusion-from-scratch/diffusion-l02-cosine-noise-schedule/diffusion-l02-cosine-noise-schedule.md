# Cosine Noise Schedule

A noise schedule decides how quickly a diffusion model's training images dissolve into static. The linear schedule dissolves them too early, so many late steps see almost nothing but static and teach the model very little. The cosine schedule of Nichol and Dhariwal (2021) fades the image along a smooth curve that is gentle at both ends, like a dimmer switch that starts and finishes slowly. This page builds it step by step, from the idea of designing the fade curve directly, to reading per-step noise amounts off that curve, to the offset and the clipping that keep the first and last steps well behaved, which is what you implement here.

## What a schedule controls

The forward process adds noise in steps numbered $t = 0, 1, \dots, T - 1$, where step $t$ adds noise of variance $\beta_t$. The quantity that matters most is the cumulative signal fraction $\bar\alpha_t$, the share of the original signal variance that survives after steps 0 through $t$. It decides the noisy sample directly, since $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ for clean data $x_0$ and standard Gaussian noise $\epsilon$. Timesteps are 0-based, so the clean data sits before step 0, where $\bar\alpha_{-1} = 1$.

A good schedule makes $\bar\alpha_t$ fall at a pace that keeps every timestep informative. Some timesteps should look slightly noisy, some half noisy, and some almost pure noise, with no long stretch where nothing changes.

## What the linear schedule gets wrong

The linear schedule of Ho et al. (2020) spaces the betas evenly and multiplies the per-step fractions $1 - \beta_t$ together. Over 1000 steps, its $\bar\alpha_t$ drops below one half at $t = 259$ and below $0.01$ at $t = 673$. The last third of the steps therefore see samples that are almost pure noise, and training spends a large share of its effort where there is little left to learn.

The cosine schedule over the same 1000 steps crosses one half at $t = 496$ and drops below $0.01$ only at $t = 935$. Noise arrives slowly near the start, quickly in the middle and slowly again at the end, so every part of the trajectory carries useful information.

## Designing the fade curve first

The linear schedule chooses the betas and lets $\bar\alpha_t$ fall wherever the product sends it. The cosine schedule turns this around and chooses the shape of $\bar\alpha$ first. It needs a curve that starts at 1, ends at 0, and is flat at both ends.

A squared cosine has exactly that shape. As its angle moves from 0 to $\pi/2$, the value $\cos^2$ falls from 1 to 0, slowly at first, fastest in the middle, and slowly again near the end. The schedule maps the time grid onto that angle range. With $T$ steps it uses $T + 1$ grid points $u = 0, 1, \dots, T$, where $u = 0$ is the clean data and $u = T$ is the end of the last step.

$$
f(u) = \cos^2\!\left(\frac{u / T + s}{1 + s} \cdot \frac{\pi}{2}\right)
$$

Here $s$ is a small offset, explained later, which can be ignored for now by setting $s = 0$. Then the angle is simply $(u / T) \cdot \pi / 2$. The curve $f$ is turned into a signal fraction by dividing by its starting value, $\bar\alpha(u) = f(u) / f(0)$, so the clean data always has a fraction of exactly 1.

## Reading the betas off the curve

Models and samplers still need per-step betas. Because $\bar\alpha$ is a running product of per-step fractions, the fraction kept by one step is the ratio of two neighbouring cumulative values. The normalizer $f(0)$ appears in both and cancels.

$$
1 - \beta_t = \frac{f(t + 1)}{f(t)}
$$

Step $t$ moves the process from grid point $t$ to grid point $t + 1$, so the signal fraction after step $t$ corresponds to grid point $t + 1$. This is why the grid has one more point than there are steps.

## A three-step example

Take $T = 3$ and $s = 0$. The angles are $0$, $\pi/6$, $\pi/3$ and $\pi/2$, so the grid values are $f(0) = 1$, $f(1) = \cos^2(\pi/6) = 0.75$, $f(2) = \cos^2(\pi/3) = 0.25$, and $f(3) \approx 3.7 \times 10^{-33}$. The last value would be exactly 0 in exact arithmetic, and rounding leaves a tiny positive number.

1. The first ratio is $0.75 / 1 = 0.75$, so $\beta_0 = 0.25$.
2. The second ratio is $0.25 / 0.75 \approx 0.3333$, so $\beta_1 \approx 0.6667$.
3. The third ratio is essentially 0, so the raw $\beta_2$ is essentially 1.

That last value is a problem, and the next two sections deal with the two ends of the curve.

## Why the offset s exists

At the start, a curve that is perfectly flat adds almost no noise. For $T = 1000$ and $s = 0$, the first beta is about $2.47 \times 10^{-6}$, a noise standard deviation of about $0.00157$. Nichol and Dhariwal found that the network struggles to predict noise this faint, so they shifted the angle slightly away from 0 with the offset $s$. Dividing by $1 + s$ keeps the final angle at exactly $\pi/2$, so the curve still ends at 0.

With $s = 0.008$ the first beta becomes about $4.13 \times 10^{-5}$, and the first noise standard deviation is about $0.0064$. That is just below $1 / 127.5 \approx 0.0078$, the size of one intensity level for pixels scaled to the range from $-1$ to 1, so the first step blurs the data by less than one quantisation level.

## Why the last betas must be clipped

At the final grid point the cosine angle is exactly $\pi/2$, so the last raw beta equals 1, as in the example. A beta of 1 means the step keeps nothing, and a reverse sampler that divides by the square root of the kept fraction would divide by zero. Betas just below 1 near the end cause similar instability. Every beta is therefore capped from above at $\beta_{\max}$, the argument max_beta, usually $0.999$.

$$
\beta_t = \min\!\left(1 - \frac{f(t + 1)}{f(t)},\ \beta_{\max}\right)
$$

## Rebuilding alpha_bar from the clipped betas

Clipping changes the betas, so the cumulative product has to be rebuilt from the clipped values, as $\bar\alpha_t = \prod_{k=0}^{t} (1 - \beta_k)$. If alpha_bars were read straight from the cosine curve instead, the two arrays would describe different processes. Formulas that mix them, such as posterior means and sampler steps, would then be inconsistent.

Finishing the three-step example with max_beta $= 0.999$, the clipped betas are $0.25$, about $0.6667$ and $0.999$. The rebuilt products are $\bar\alpha_0 = 0.75$, then $\bar\alpha_1 = 0.75 \cdot 0.3333 = 0.25$, then $\bar\alpha_2 = 0.25 \cdot 0.001 = 0.00025$. The first two match the curve, because nothing was clipped yet, and the last stays small but positive instead of collapsing to zero.

Over 1000 steps with $s = 0.008$ and max_beta $= 0.999$, only the final beta reaches the cap. The four betas before it are about $0.36$, $0.4375$, $0.5556$ and $0.75$, which shows how sharply the curve steepens at the very end.

## Where the schedule sits in the pipeline

The schedule is computed once and shared by training and sampling. Training reads alpha_bars to build noisy samples at random timesteps, and the sampler reads both betas and alpha_bars at every reverse step. Swapping a linear schedule for a cosine one changes nothing else in the code, which is why schedules are usually a single configuration option.

## Common mistakes to avoid

- **Taking alpha_bars from the cosine curve after clipping the betas.** The two arrays then disagree at every clipped step and at every step after it.
- **Forgetting to square the cosine.** The curve then lies above the intended alpha_bar at every interior timestep, so the samples keep too much signal and every beta changes.
- **Dropping the division by one plus s.** The angle then overshoots $\pi/2$ at the end, so the curve misses zero at the final grid point.
- **Evaluating f on only num_steps points.** The grid then lacks its endpoint, which shifts every step by one and makes timestep 0 claim to be perfectly clean.
- **Clipping in the wrong direction.** Taking the larger of the raw beta and max_beta raises every beta to at least max_beta, so the data is destroyed in the very first step.
- **Leaving the final beta unclipped.** A beta of 1 makes the kept fraction zero, and the reverse process divides by its square root.