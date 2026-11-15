# Linear Beta Schedule

A diffusion model ruins its training data on purpose, a little at a time, much like turning up the static on a radio until the song is gone. The noise schedule is the dial that says how much static to add at each step, and the model needs it so that it always knows how noisy its input is. The linear schedule from Ho et al. (2020) turns that dial up at a steady rate. This page builds it step by step, from the noise added by one step, to a list of evenly spaced noise amounts, to the two arrays derived from that list, which together are what you implement here.

## How much noise one step adds

The forward process of a diffusion model corrupts a clean sample $x_0$ in a chain of small steps. Each step shrinks the current sample slightly and adds a little fresh Gaussian noise $\epsilon_t$, which has mean 0 and variance 1. The amount is set by one number $\beta_t$ between 0 and 1, the variance of the noise added at step $t$.

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t
$$

Take one pixel with value $1$ and a step with $\beta_t = 0.04$. The signal is multiplied by $\sqrt{0.96} \approx 0.9798$, and the noise is multiplied by $\sqrt{0.04} = 0.2$. With a noise draw of $\epsilon_t = 0.5$, the pixel becomes about $0.9798 + 0.1 = 1.0798$.

The two weights are chosen so their squares add to 1. If the sample has variance 1 before the step, it still has variance $(1 - \beta_t) + \beta_t = 1$ after it, so the step swaps a little signal for noise without growing or shrinking the data.

## Why each step must be small

The reverse process learned later undoes one step at a time. A single step is close to a simple Gaussian only when $\beta_t$ is small, and then the network can learn to reverse it reliably. A large step would ask the network to undo a big, complicated jump in one go, which it cannot model well. Real schedules therefore use many steps, usually hundreds or a thousand, each adding only a sliver of noise.

## A schedule is the list of all betas

With $T$ steps, numbered $t = 0, 1, \dots, T - 1$, the list $\beta_0, \dots, \beta_{T-1}$ is called the variance schedule. It fixes the noise level of every timestep once and for all. Because of that, the denoising network can be told the timestep and know exactly how corrupted its input is.

Timesteps are 0-based. The clean sample $x_0$ sits before step 0, so the noisy sample at timestep $t$ has passed through steps $0$ to $t$.

## Why the betas grow over time

Early in the process the sample still carries fine detail, and even a little noise destroys a lot of it. Late in the process the sample is already mostly noise, so adding more barely changes what the network sees. A schedule that adds very little noise at first and more noise later spends its steps where they matter.

The linear schedule does this in the plainest possible way. It places $T$ evenly spaced values between $\beta_{\text{start}}$ and $\beta_{\text{end}}$, the arguments beta_start and beta_end, including both endpoints. With three steps from $0.02$ to $0.06$, the spacing is $0.02$ and the betas are $0.02$, $0.04$ and $0.06$. In general there are $T - 1$ gaps between $T$ points, which gives

$$
\beta_t = \beta_{\text{start}} + \frac{t}{T - 1}\left(\beta_{\text{end}} - \beta_{\text{start}}\right), \qquad t = 0, \dots, T - 1
$$

The first beta is exactly beta_start, and the last is exactly beta_end. With a single step there are no gaps, and the schedule holds only beta_start.

## The fraction each step keeps

Most diffusion formulas talk about what a step keeps instead of what it adds. The per-step alpha is

$$
\alpha_t = 1 - \beta_t
$$

which is the fraction of the variance that comes from the previous sample after step $t$. For the three betas above, the alphas are $0.98$, $0.96$ and $0.94$.

## Stacking steps into a cumulative fraction

What a training step really needs is how much of the original clean signal survives at timestep $t$, after all the steps so far. When two steps run in a row, the signal is scaled by $\sqrt{\alpha_0}$ and then by $\sqrt{\alpha_1}$, so its surviving variance fraction is $\alpha_0 \alpha_1$. The noise added along the way combines into a single Gaussian whose variance makes up the rest. Repeating the argument for every step gives the cumulative product

$$
\bar\alpha_t = \prod_{s=0}^{t} \alpha_s
$$

which is the fraction of the original signal variance left after steps 0 through $t$. Every alpha lies between 0 and 1, so the sequence $\bar\alpha_t$ starts just below 1 and falls steadily towards 0.

This product is what makes training fast. Unrolling the one-step rule shows that $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ for a single standard Gaussian $\epsilon$, so any timestep can be reached in one jump once $\bar\alpha_t$ is known.

The product is inclusive. The empty product before the first step is $\bar\alpha_{-1} = 1$, which describes the clean data, and $\bar\alpha_0 = \alpha_0$ already contains the first step of noise.

## A three-step example

Return to $T = 3$ with beta_start $= 0.02$ and beta_end $= 0.06$. The betas are $0.02$, $0.04$ and $0.06$, and the alphas are $0.98$, $0.96$ and $0.94$.

1. The first cumulative value is $\bar\alpha_0 = 0.98$.
2. The second is $\bar\alpha_1 = 0.98 \cdot 0.96 = 0.9408$.
3. The third is $\bar\alpha_2 = 0.9408 \cdot 0.94 = 0.884352$.

At the last step the signal factor is $\sqrt{0.884352} \approx 0.9404$ and the noise factor is $\sqrt{1 - 0.884352} \approx 0.3401$. Three small steps leave most of the signal intact, which is why real schedules need hundreds of them to reach pure noise.

## The standard DDPM setting

Ho et al. (2020) used $T = 1000$ with beta_start $= 10^{-4}$ and beta_end $= 0.02$. The first step keeps a fraction $0.9999$ of the variance, so its noise factor is only $0.01$. The last value of $\bar\alpha_t$ is about $4.04 \times 10^{-5}$, which gives a signal factor of about $0.0064$ and a noise factor of about $0.99998$. The final sample is almost pure noise, which is the state the sampler starts from.

The same numbers reveal the weakness of the linear schedule. The signal fraction drops below one half at $t = 259$ and below $0.01$ at $t = 673$, so roughly the last third of all timesteps hold samples that are nearly indistinguishable from noise. Nichol and Dhariwal (2021) proposed the cosine schedule to spread the destruction more evenly, and later work argued that the final value should be exactly 0 instead of merely small.

## Numerical care

A product of a thousand numbers below 1 is the kind of quantity where precision matters. In float64 the running product keeps about sixteen significant digits, while float32 keeps only about seven and adds a rounding error with every factor. Schedules are computed once and reused at every training and sampling step, so building them in float64 costs nothing and keeps those errors out of every later formula.

## Where the schedule sits in the pipeline

The schedule is built once before training starts. The training loop reads alpha_bars to make noisy samples, and loss weightings often read it to compute signal to noise ratios. The sampler reads betas, alphas and alpha_bars at every reverse step, so a single wrong entry here corrupts both training and generation.

## Common mistakes to avoid

- **Dividing by num_steps instead of num_steps minus one.** The betas then stop short of beta_end, so the last timestep is less noisy than intended.
- **Using the per-step alphas where the cumulative product is needed.** Each alpha describes one small step, so treating it as alpha_bar makes every timestep look almost clean.
- **Starting the running product at 1.** An exclusive product shifts the whole curve by one step, so timestep 0 claims to hold perfectly clean data.
- **Spacing the square roots of the betas evenly.** That variant is the scaled linear schedule from latent diffusion models, and it produces different middle values.
- **Dividing by zero when there is one step.** With num_steps equal to 1 the spacing formula has no gaps, and the schedule should hold beta_start alone.
- **Building the arrays in float32.** The running product loses precision over many steps, and every formula that reads it inherits the error.