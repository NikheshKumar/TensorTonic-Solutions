# Rescale to Zero Terminal SNR

Most noise schedules stop just short of pure static, so a faint ghost of the training image survives even at the noisiest step. The model learns to lean on that ghost, but at generation time it starts from real static with no ghost in it, and Lin et al. (2024) showed that this mismatch limits what the model can generate. Their fix gently reshapes any schedule so that the last step holds no signal at all. This page builds that fix step by step, from a way to measure leftover signal, to a simple shift and stretch of the signal factors, to the rebuilt betas and signal to noise ratios, which is what you implement here.

## Measuring how much signal is left

A schedule is given by its betas $\beta_t$, the noise variance added at step $t$ for $t = 0, \dots, T - 1$. Their cumulative signal fraction is $\bar\alpha_t = \prod_{k=0}^{t} (1 - \beta_k)$, the share of the clean signal's variance that survives after steps 0 through $t$. It decides the noisy sample, $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, where $x_0$ is the clean data and $\epsilon$ is standard Gaussian noise.

A convenient summary of one timestep is its signal to noise ratio, the variance of the signal part divided by the variance of the noise part.

$$
\mathrm{snr}_t = \frac{\bar\alpha_t}{1 - \bar\alpha_t}
$$

With $\bar\alpha_t = 0.8$ the ratio is $0.8 / 0.2 = 4$, so the signal is four times stronger than the noise. With $\bar\alpha_t = 0.5$ the ratio is 1. The ratio falls towards 0 as the noise takes over, and it equals 0 only when $\bar\alpha_t = 0$, which means no signal is left at all.

## Why a small leftover signal matters

Typical schedules never quite reach 0. With the linear DDPM schedule over 1000 steps, the final signal factor $\sqrt{\bar\alpha_{T-1}}$ is about $0.0064$. The scaled linear schedule used by latent diffusion models ends with $\bar\alpha_{T-1} \approx 0.0047$, a signal factor of about $0.068$ and a terminal SNR of about $0.0047$.

That leftover signal is small per pixel, but it is strongest in the lowest frequencies, such as the mean brightness of an image. During training the network always sees a final sample that still carries this hint, and it learns to rely on it. At generation time the sampler starts from pure Gaussian noise, which carries no such hint, so the network keeps roughly the average brightness of its starting noise. Lin et al. (2024), in "Common Diffusion Noise Schedules and Sample Steps are Flawed", found that models trained this way cannot produce very bright or very dark images even when asked to.

## Working with signal factors

The fix should make the last timestep pure noise while moving every earlier timestep as little as possible. It acts on the signal factors, the square roots of the cumulative fractions,

$$
a_t = \sqrt{\bar\alpha_t}
$$

because these are the coefficients that actually multiply $x_0$ in the forward process. A signal factor of 0 at the last timestep is exactly what pure noise means.

## A shift that removes the leftover

Start with the simplest change that makes the last factor zero. Subtracting the last factor $a_{T-1}$ from every factor gives $a_t - a_{T-1}$, which is 0 at $t = T - 1$ and still decreases over time.

The shift alone has a side effect. It lowers every factor by the same amount, including the first one, so even the cleanest timestep would suddenly receive more noise than before. Take a schedule whose signal factors are $0.9$, $0.6$ and $0.2$. Shifting gives $0.7$, $0.4$ and $0$, and the first timestep has lost more than a fifth of its signal factor.

## A stretch that restores the first step

The fix for that side effect is a stretch. Multiplying the shifted factors by $a_0 / (a_0 - a_{T-1})$ maps the first shifted value $a_0 - a_{T-1}$ back onto $a_0$, while the last value stays at 0. In the small example the stretch factor is $0.9 / 0.7 \approx 1.2857$, which turns $0.7$, $0.4$ and $0$ into $0.9$, about $0.5143$, and $0$.

Combining the shift and the stretch gives the rescaled signal factors $\tilde a_t$, and squaring them turns them back into cumulative fractions.

$$
\tilde a_t = \left(a_t - a_{T-1}\right) \frac{a_0}{a_0 - a_{T-1}}, \qquad \tilde{\bar\alpha}_t = \tilde a_t^{\,2}
$$

This is a linear map of the signal factors, so their ordering and the overall shape of the curve are preserved. Applying the same map to $\bar\alpha_t$ itself would bend the curve differently and would not match the method that the published models use.

## Rebuilding the betas and the SNR

Samplers and training code read betas as well as alpha_bars, so the new betas have to be recovered from the new products. The fraction kept by one step is the ratio of neighbouring cumulative values, $\tilde{\bar\alpha}_t / \tilde{\bar\alpha}_{t-1}$. Timesteps are 0-based, and the clean data sits before step 0, so the value before the first step is $\tilde{\bar\alpha}_{-1} = 1$.

$$
\tilde\beta_t = 1 - \frac{\tilde{\bar\alpha}_t}{\tilde{\bar\alpha}_{t-1}}
$$

The first beta is unchanged, since the first alpha_bar is unchanged. The last new alpha_bar is 0, so the last step keeps nothing and the last beta is exactly 1. The new signal to noise ratio $\mathrm{snr}_t = \tilde{\bar\alpha}_t / (1 - \tilde{\bar\alpha}_t)$ is exactly 0 at the last timestep, which is what the name of the method refers to.

## A three-step example

Take betas $0.2$, $0.3$ and $0.4$.

1. The original alpha_bars are $0.8$, $0.8 \cdot 0.7 = 0.56$ and $0.56 \cdot 0.6 = 0.336$.
2. The signal factors are about $0.8944$, $0.7483$ and $0.5797$.
3. Shifting by the last factor gives about $0.3148$, $0.1687$ and $0$.
4. The stretch factor $0.8944 / 0.3148$ is about $2.8415$, which restores the first value to $0.8944$ and turns the middle one into about $0.4793$.
5. Squaring gives the new alpha_bars $0.8$, about $0.2297$ and $0$.
6. The new betas are $1 - 0.8 = 0.2$, then $1 - 0.2297 / 0.8 \approx 0.7128$, then $1$.
7. The new SNR values are $4$, about $0.2982$ and $0$.

The middle step now adds far more noise than before, because the whole remaining signal has to be removed by the end. Over 1000 steps of the scaled linear schedule the change is much gentler. The first beta stays at $0.00085$, and the alpha_bar at $t = 499$ moves only from about $0.2777$ to about $0.2424$, while the last few betas rise sharply towards 1.

## Consequences for the rest of the pipeline

A zero terminal SNR breaks one common assumption. Recovering a clean sample from a noise prediction divides by $\sqrt{\bar\alpha_t}$, which is now 0 at the last step, so predicting the noise tells the model nothing about the clean data there. Lin et al. therefore pair the rescaled schedule with v prediction, and they also make the sampler start exactly at the last timestep. Loss weightings that divide by the SNR need the same care, since the last SNR is now 0.

## Common mistakes to avoid

- **Rescaling the alpha_bars instead of their square roots.** The resulting schedule differs at every interior timestep from the one the method defines.
- **Shifting without stretching.** The first alpha_bar then drops below its original value, so even the cleanest timestep receives extra noise.
- **Forgetting to square the rescaled values.** Signal factors and variance fractions get mixed up, and every downstream formula reads the wrong quantity.
- **Treating the value before step 0 as the first alpha_bar.** The cumulative value before step 0 is 1, so the first kept fraction equals the first alpha_bar and the first beta keeps its original value.
- **Computing the SNR as a ratio of standard deviations.** The SNR compares variances, so it uses the alpha_bars without square roots.
- **Overwriting the input betas.** The original schedule is often still needed for comparison, and changing it in place silently corrupts it.