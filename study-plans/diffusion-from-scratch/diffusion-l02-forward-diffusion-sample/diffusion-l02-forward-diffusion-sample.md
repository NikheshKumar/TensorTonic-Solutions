# Forward Diffusion Sample

A diffusion model learns to remove noise, so during training it needs a steady supply of clean examples that have been mixed with a known amount of noise. Think of photos taken through fog, where some photos have a thin mist and others are almost completely grey. This page builds the idea one piece at a time, starting from a single number and ending with the formula that adds the right amount of noise to a whole batch in one step. That formula is what you implement in this problem.

## Mixing one number with noise

Start with a single clean value, say a pixel with value $x_0 = 1.5$. To make it noisy, you blend it with a random number $\epsilon$ drawn from a standard Gaussian, which has mean 0 and variance 1. The blend uses two weights, one for the clean value and one for the noise.

$$
\text{noisy value} = a \cdot x_0 + b \cdot \epsilon
$$

A large $a$ and a small $b$ give a lightly noisy value that still looks like the original. A small $a$ and a large $b$ give a value that is mostly noise. Suppose $a = 0.6$, $b = 0.8$ and the noise draw is $\epsilon = 0.5$. The noisy value is $0.6 \cdot 1.5 + 0.8 \cdot 0.5 = 0.9 + 0.4 = 1.3$.

Everything else on this page is about choosing $a$ and $b$ well and doing this for many values at once.

## Why the two weights must satisfy one rule

You could pick any two weights, but diffusion models pick them so that $a^2 + b^2 = 1$. The reason is the size of the numbers. Assume the clean data has been scaled to variance 1, which is the usual practice. The variance of a weighted sum of two independent random quantities is the sum of their squared weights times their variances, so the noisy value has variance $a^2 \cdot 1 + b^2 \cdot 1$.

With $a = 0.6$ and $b = 0.8$, that variance is $0.36 + 0.64 = 1$, the same as the clean data. The value changes from signal to noise without growing or shrinking. If you used weights that did not follow the rule, say $0.36$ and $0.64$, the variance would be $0.1296 + 0.4096 \approx 0.54$, and the data would quietly shrink as noise is added. A network that sees inputs of a different size at every noise level has a much harder time learning.

So the natural way to describe a noise level is by one number, the fraction of variance that still comes from the clean data. Call that fraction $\bar\alpha$. The rule then fixes both weights, because $a = \sqrt{\bar\alpha}$ and $b = \sqrt{1 - \bar\alpha}$. In the example above, $\bar\alpha = 0.36$. The square roots are there because $\bar\alpha$ measures variance and the weights multiply values.

## Adding noise in many small steps

Diffusion models do not add all the noise at once in their definition. They describe a chain of many small steps, numbered $t = 0, 1, \dots, T - 1$, where each step mixes in a little fresh noise. Step $t$ has a small number $\beta_t$ between 0 and 1, called the variance schedule, that says how much noise that step adds.

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t
$$

Here $x_{t-1}$ is the sample before the step, $x_t$ is the sample after it, and $\epsilon_t$ is fresh standard Gaussian noise drawn for that step alone. This is the same mixing rule as before, with $a^2 = 1 - \beta_t$ and $b^2 = \beta_t$, so each step keeps the variance at 1. With $\beta_t = 0.1$, one step keeps the signal with weight $\sqrt{0.9} \approx 0.9487$ and adds noise with weight $\sqrt{0.1} \approx 0.3162$.

It helps to name the fraction of variance a single step keeps, $\alpha_t = 1 - \beta_t$. The timesteps are 0-based, so the clean data $x_0$ sits before step 0, and the sample at timestep $t$ has gone through steps $0$ to $t$.

## Two steps collapse into one

Running hundreds of these steps one after another for every training example would be slow. Fortunately, two Gaussian steps in a row behave like a single Gaussian step. A sum of independent Gaussian noise terms is again Gaussian, and its variance is the sum of the individual variances.

Take a schedule with $\beta_0 = 0.1$ and $\beta_1 = 0.2$, so $\alpha_0 = 0.9$ and $\alpha_1 = 0.8$. After both steps, the clean value has been scaled by $\sqrt{0.9}$ and then by $\sqrt{0.8}$, a total weight of $\sqrt{0.72} \approx 0.8485$. The noise from step 0 has been scaled by $\sqrt{0.1}$ and then by $\sqrt{0.8}$, which gives it variance $0.08$. The noise from step 1 has variance $0.2$. Together the two noise terms have variance $0.08 + 0.2 = 0.28$, which is exactly $1 - 0.72$.

So the two steps are equivalent to one mix with $\bar\alpha = 0.72$, where the clean data keeps 72 percent of the variance and a single Gaussian noise draw supplies the other 28 percent.

## The shortcut for any timestep

The same argument works for any number of steps. Multiply together the fractions kept by every step up to and including $t$, and call the result the cumulative fraction.

$$
\bar\alpha_t = \alpha_0 \, \alpha_1 \cdots \alpha_t
$$

Then the sample at timestep $t$ can be written directly in terms of the clean data, with one fresh noise tensor $\epsilon$ of the same shape as $x_0$.

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

This is the mixing rule from the first section, with the weights set by the cumulative fraction. It is the formula you implement. Because $\bar\alpha_t$ is a product of numbers below 1, it starts close to 1 and falls towards 0 as $t$ grows, so later timesteps are noisier. A schedule stores all of these values in one array, alpha_bars, so the value for timestep $t$ is simply the entry at index $t$.

## What the noise level looks like along the schedule

The value of $\bar\alpha_t$ tells you the whole story of a timestep.

- **Near the start.** With $\bar\alpha_t = 0.9999$, the signal weight is about $0.99995$ and the noise weight is only $0.01$, so the sample is almost clean.
- **In the middle.** With $\bar\alpha_t = 0.5$, both weights are about $0.7071$, and signal and noise contribute equal variance.
- **At the end.** A good schedule drives $\bar\alpha_t$ very close to 0, so the last sample is almost pure noise. Generation starts from pure noise, so training has to cover that state too.

## Every sample in a batch has its own noise level

During training, a batch holds many clean samples, and each one is given its own randomly chosen timestep. The network has to learn every noise level, and mixing them in one batch gives it a balanced view.

This means the two weights are different for every sample. For one sample they are single numbers, and they must scale every pixel and channel of that sample in the same way. If the batch has shape $(B, C, H, W)$, the $B$ per-sample weights have to line up with the batch axis, so each weight multiplies only its own sample. The same idea holds for flat data of shape $(B, D)$ or for any other number of dimensions after the batch axis.

## Where this step sits in training

Each training step of a diffusion model uses this formula. It takes a batch of clean samples, picks a random timestep and a fresh noise tensor for each one, builds the noisy samples with the shortcut, and asks the network to predict the noise that was added. The same formula appears again later in the plan, when samplers turn a predicted noise back into a predicted clean sample.

## Common mistakes to avoid

- **Forgetting the square roots.** The schedule stores variance fractions, so using $\bar\alpha_t$ directly as a weight gives the wrong balance and shrinks the data.
- **Using the single-step alpha instead of the cumulative one.** One step's $\alpha_t$ describes only that small step, while the shortcut needs the product of all steps up to $t$.
- **Shifting the timestep index.** Timesteps are 0-based, so the sample at timestep $t$ reads alpha_bars at index $t$, never at $t - 1$.
- **Lining the weights up with the wrong axis.** A vector of $B$ weights lines up with the last axis of a tensor by default, which silently mixes samples together when the sizes happen to match.
- **Using one timestep for the whole batch.** Each sample needs its own noise level, otherwise every sample except one gets the wrong weights.
- **Changing the clean batch in place.** The clean samples are reused for the loss, so scaling them in place corrupts the rest of the training step.