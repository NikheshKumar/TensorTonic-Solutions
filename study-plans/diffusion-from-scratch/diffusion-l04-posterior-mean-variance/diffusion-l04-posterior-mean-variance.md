# Posterior Mean and Variance

A diffusion model creates data by running the noising process backwards, one small step at a time. Going back one step is hard in general, because many earlier samples could have led to the same noisy sample. If you also knew the clean sample, the question would become easy, much like guessing where a hiker was an hour ago when you know both the trailhead and where the hiker stands now. That easy case has an exact answer, a bell-shaped spread of likely positions with a known centre and width, and it is the target that every learned reverse step imitates. This page builds that answer from the simple idea of combining two noisy clues, and ends with the mean and variance you compute in this problem.

## Combining two noisy clues

Start away from diffusion. Two thermometers measure the same room. The first reads 20 degrees and has an error variance of 1, and the second reads 23 degrees with an error variance of 2. The best combined estimate trusts each reading in proportion to its precision, which is one over its variance.

The precisions are 1 and 0.5, so the combined estimate is $(20 \cdot 1 + 23 \cdot 0.5) / 1.5 = 21$. The combined precision is the sum $1.5$, so the combined variance is $1 / 1.5 \approx 0.667$. The estimate leans towards the more reliable thermometer, and its variance is smaller than either reading's variance, because two clues always say more than one.

This is exactly what happens when two Gaussian beliefs about the same quantity are multiplied together. The product is another Gaussian, its precision is the sum of the two precisions, and its mean is the precision-weighted average of the two means. The rest of the page applies this rule to diffusion.

## The two clues about the previous sample

The quantity we want is the sample one step earlier, $x_{t-1}$, for a single number to begin with. Two facts constrain it.

The first clue comes from the clean sample. The forward formula says that $x_{t-1}$ is $\sqrt{\bar\alpha_{t-1}}\, x_0$ plus noise of variance $1 - \bar\alpha_{t-1}$, where $\bar\alpha_{t-1}$ is the cumulative signal fraction one step earlier. So the clean sample points to a centre of $\sqrt{\bar\alpha_{t-1}}\, x_0$ with variance $1 - \bar\alpha_{t-1}$.

The second clue comes from the noisy sample. A single forward step makes $x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t$, where $\beta_t$ is the variance added by step $t$ (betas at index $t$) and $\alpha_t = 1 - \beta_t$ is the fraction that step keeps. Undoing the scale, $x_{t-1}$ is about $x_t / \sqrt{\alpha_t}$, with variance $\beta_t / \alpha_t$ from the unknown step noise.

Both clues are Gaussian beliefs about the same number, so the thermometer rule combines them. This combined belief is called the forward posterior, written $q(x_{t-1} \mid x_t, x_0)$, and Ho et al. (2020) derived it in closed form.

## The posterior variance

Adding the two precisions gives the precision of the posterior.

$$
\frac{1}{1 - \bar\alpha_{t-1}} + \frac{\alpha_t}{\beta_t}
$$

Over a common denominator, the numerator is $\beta_t + \alpha_t (1 - \bar\alpha_{t-1})$. Since $\alpha_t \bar\alpha_{t-1} = \bar\alpha_t$ and $\beta_t + \alpha_t = 1$, this numerator simplifies to $1 - \bar\alpha_t$. Inverting gives the posterior variance, written $\tilde\beta_t$.

$$
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\, \beta_t
$$

The factor in front of $\beta_t$ is below 1, because $\bar\alpha_{t-1}$ is larger than $\bar\alpha_t$. So the posterior is always narrower than a single forward step, which matches the thermometer lesson that two clues beat one. Late in the schedule both cumulative values are close to 0, the factor approaches 1, and $\tilde\beta_t$ is almost $\beta_t$.

## The posterior mean

The mean is the precision-weighted average of the two clue centres. Multiplying each centre by its precision, adding them and multiplying by the posterior variance gives a weighted sum of the two inputs.

$$
\tilde\mu_t = \frac{\beta_t \sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_t}\, x_0 + \frac{(1 - \bar\alpha_{t-1}) \sqrt{\alpha_t}}{1 - \bar\alpha_t}\, x_t
$$

The two coefficients use different alphas, and this is where most errors happen. The coefficient on $x_0$ contains the cumulative $\bar\alpha_{t-1}$, because that clue came from the closed form jump from the clean sample. The coefficient on $x_t$ contains the single-step $\alpha_t$, because that clue came from undoing one step.

The coefficients also say which clue dominates. Near the end of the schedule $\bar\alpha_{t-1}$ is tiny, so the clean clue is weak and the mean follows $x_t$. Near the start the clean clue is very precise and the mean moves towards it.

## A one-number example

Take two steps with betas $= (0.2, 0.5)$, so the alphas are $(0.8, 0.5)$ and the cumulative values are $\bar\alpha_0 = 0.8$ and $\bar\alpha_1 = 0.4$. Work at $t = 1$ with a clean value $x_0 = 1$ and a noisy value $x_t = 0.4$.

1. The clean clue has centre $\sqrt{0.8} \cdot 1 \approx 0.8944$ and variance $1 - 0.8 = 0.2$, so its precision is 5.
2. The noisy clue has centre $0.4 / \sqrt{0.5} \approx 0.5657$ and variance $0.5 / 0.5 = 1$, so its precision is 1.
3. The posterior precision is 6, so $\tilde\beta_1 = 1/6 \approx 0.1667$, and the closed form agrees, since $0.2 \cdot 0.5 / 0.6 = 1/6$.
4. The precision-weighted mean is $(5 \cdot 0.8944 + 0.5657) / 6 \approx 0.8396$.
5. Through the closed form, the coefficients are $0.5 \sqrt{0.8} / 0.6 \approx 0.7454$ and $0.2 \sqrt{0.5} / 0.6 \approx 0.2357$, and $0.7454 + 0.2357 \cdot 0.4 \approx 0.8396$.

Both routes give the same mean, and the mean sits much closer to the clean clue because that clue is five times more precise.

## Moving to a batch

In a real batch, $x_0$ and $x_t$ have shape $(B, \dots)$, for example $(B, C, H, W)$, and timesteps gives every sample its own $t$. The two coefficients and $\tilde\beta_t$ depend only on the schedule and $t$, so each sample has its own three scalars, and the two mean coefficients scale every element of that sample in the same way. Because these values depend only on the schedule, it is natural to build them once for every timestep and then look them up per sample.

## The boundary at t = 0

Timesteps are 0-based, so the step before $t = 0$ is the clean sample itself and the plan defines $\bar\alpha_{-1} = 1$. Then $1 - \bar\alpha_{-1} = 0$ and $\tilde\beta_0 = 0$. The coefficient on $x_t$ vanishes, and the coefficient on $x_0$ becomes $\beta_0 / (1 - \bar\alpha_0) = 1$, since $1 - \bar\alpha_0 = \beta_0$. The posterior is a spike at the clean sample, which is correct, because the step from $t = 0$ lands on $x_0$ with certainty.

That spike creates a numerical problem. Losses and samplers often work with $\log \tilde\beta_t$, for example inside a Gaussian KL divergence or when a network learns to interpolate between log variances. The log of 0 is minus infinity, and one infinite entry turns a whole batch loss into NaN.

The original DDPM code of Ho et al. (2020) avoids this by using the variance at $t = 1$ inside the log at $t = 0$. The reported variance stays exactly 0, while the log variance stays finite and close to the values around it. In the example above, the clipped log variance at $t = 0$ would be $\log(1/6) \approx -1.7918$. This convention is why the schedule needs at least two steps.

## Precision of the schedule

The smallest denominator $1 - \bar\alpha_t$ appears at $t = 0$, where it equals $\beta_0$. With very small betas in single precision, rounding in the cumulative product is large compared with this small difference, so diffusion code often builds schedule tables in double precision before casting.

## Common mistakes to avoid

- **Taking the log of the raw variance.** The variance at $t = 0$ is exactly 0, so its log is minus infinity and poisons every loss it enters.
- **Clamping the variance to a tiny floor before the log.** This keeps the value finite but gives a log variance near $-46$, far from the conventional clipped value.
- **Using the cumulative alpha in the x_t coefficient.** That coefficient needs the single-step $\sqrt{\alpha_t}$, and the cumulative value shrinks it far too much at late steps.
- **Indexing the previous cumulative alpha with t minus 1 at t = 0.** The index wraps to the last entry of the schedule instead of using the value 1.
- **Swapping the two mean coefficients.** The coefficient containing $\beta_t$ belongs to $x_0$, and swapping them pulls the mean towards the wrong input.
- **Reporting the forward variance.** The posterior variance is $\tilde\beta_t$, and $\beta_t$ is only an upper bound on it.