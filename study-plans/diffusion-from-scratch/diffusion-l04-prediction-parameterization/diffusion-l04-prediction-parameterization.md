# Prediction Parameterization

A diffusion network looks at a noisy sample and makes a guess, and there are three common ways to phrase that guess. It can name the noise that was added, the clean sample underneath, or a blend of the two called the velocity. Once the noisy sample is known, any one of these answers fixes the other two, in the same way that knowing a total and one of its two parts tells you the other part. Different models are trained on different targets, while samplers and losses usually expect a particular one, so converting between them is a routine job. This page builds the three targets from a single noisy number and ends with the batch conversion you implement in this problem.

## One noisy number, three descriptions

Recall the forward formula and give its two weights short names.

$$
x_t = a\, x_0 + b\, \epsilon, \qquad a = \sqrt{\bar\alpha_t}, \quad b = \sqrt{1 - \bar\alpha_t}
$$

Here $x_0$ is the clean value, $\epsilon$ is standard Gaussian noise, and $\bar\alpha_t$ is alpha_bars at the sample's timestep $t$. The two weights satisfy $a^2 + b^2 = 1$, which keeps the variance of $x_t$ at 1.

Take $\bar\alpha_t = 0.9216$, so $a = 0.96$ and $b = 0.28$. With $x_0 = 0.5$ and $\epsilon = -1$ the noisy value is $x_t = 0.48 - 0.28 = 0.2$. The network sees only $0.2$ and the timestep.

If the network says the noise is $-1$, then the clean part must be whatever is left of $0.2$ after removing $0.28 \cdot (-1)$, which is $0.48$, and dividing by $a$ gives $x_0 = 0.5$. If it says the clean value is $0.5$ instead, the same equation run the other way gives the noise. One equation with one unknown always has one answer, so a noise guess and a clean guess carry exactly the same information once $x_t$ is fixed.

## Why the target changes what the network learns

Equal information does not mean equal training. The network is trained with a squared error on whatever it outputs, so the target decides which mistakes are punished, and converting an error from one target to another rescales it.

Rearranging the forward equation shows the rescaling. An error in a clean value guess becomes an error in the implied noise multiplied by $a / b$, and a noise error becomes a clean value error multiplied by $b / a$. In the example, $a / b \approx 3.43$, so a clean value error of $0.1$ becomes a noise error of about $0.343$, while a noise error of $0.1$ becomes a clean value error of only about $0.029$.

At early timesteps, like this one, $b$ is small. Predicting $x_0$ is easy there because $x_t$ is almost clean, but turning that guess into noise inflates its error. At late timesteps the roles swap. The noise is easy to read off $x_t$, but turning a noise guess into a clean value divides by a tiny $a$, so noise prediction gives poor clean estimates near pure noise. Ho et al. (2020) trained on the noise, and some models predict $x_0$ directly, which is handy when the clean sample should be clipped to a valid range.

## The velocity as a third target

Salimans and Ho (2022) proposed a target that avoids both blow-ups. It mixes the same two ingredients with the same two weights, arranged differently.

$$
v = a\, \epsilon - b\, x_0
$$

In the example, $v = 0.96 \cdot (-1) - 0.28 \cdot 0.5 = -0.96 - 0.14 = -1.1$. When the data has unit variance, the velocity has variance $a^2 + b^2 = 1$ at every timestep, just like $x_t$, so its scale never collapses at either end of the schedule. Many image and video models predict $v$, and it is the usual choice for schedules that reach zero signal at the last step, where $a = 0$ and noise prediction breaks down completely.

## The rotation behind the velocity

Because $a^2 + b^2 = 1$, the two weights can be written as the cosine and sine of one angle, $a = \cos\phi$ and $b = \sin\phi$, with $\phi$ between 0 and $\pi / 2$. The angle grows from 0 for clean data to $\pi / 2$ for pure noise.

Now look at the pair $(x_t, v)$ next to the pair $(x_0, \epsilon)$. The rule that builds the first pair from the second has the pattern of a rotation in the plane, with $\cos\phi$ on the diagonal and $\pm\sin\phi$ off it. So $(x_t, v)$ is the point $(x_0, \epsilon)$ turned by the angle $\phi$. The velocity is also the rate at which $x_t$ moves as $\phi$ increases, which is where its name comes from.

A rotation keeps lengths. In the example, $x_0^2 + \epsilon^2 = 0.25 + 1 = 1.25$, and $x_t^2 + v^2 = 0.04 + 1.21 = 1.25$ as well. Undoing a rotation is another rotation by the opposite angle, so it never stretches anything either. That is why recovering the clean value and the noise from a velocity has no amplifying division, in contrast with the other two conversions. Working out the exact form of the inverse rotation is the heart of the v branch in this problem, and the identity $a^2 + b^2 = 1$ is what makes the algebra collapse.

A correct v conversion on the example must take $x_t = 0.2$ and $v = -1.1$ and return $x_0 = 0.5$ and $\epsilon = -1$, so that pair is a handy check for your own derivation.

## From one number to a batch

In a batch, x_t and model_output have shape $(B, \dots)$, for example $(B, C, H, W)$, and timesteps gives each sample its own $t$. The weights $a$ and $b$ are scalars per sample, so each sample uses its own pair for every one of its elements, and two samples in the same batch can sit at very different points on the circle. Timesteps are 0-based, so sample $i$ reads alpha_bars at position $t_i$.

The argument prediction_type says what model_output holds. Whichever target it names is returned as given, and the other two follow from the two linear relations above. The function always produces all three, which lets a sampler written for one target use a model trained on another.

## Edge cases and numerical stability

The conversions that start from a noise guess divide by $a$, and the ones that start from a clean guess divide by $b$. Both are safe while $\bar\alpha_t$ stays strictly between 0 and 1, but rounding errors grow as either end is approached. The velocity conversions need no division at all.

A string that names none of the three targets should never fall through to a default branch. A silent wrong conversion produces plausible looking but meaningless samples, while raising a ValueError makes a misconfigured model fail on its first call.

## Where conversion is used

Samplers such as DDPM and DDIM are written in terms of a predicted clean sample and a predicted noise, so a model trained with any target is converted to those two first. Loss weighting schemes such as Min-SNR also need to know which target the network outputs, because the same squared error means different things for each one.

## Common mistakes to avoid

- **Flipping the sign of the velocity.** With $v = b\, x_0 - a\, \epsilon$ the recovered clean sample and noise both come out wrong whenever the model predicts v.
- **Forgetting the square roots.** The schedule stores variance fractions, so using $\bar\alpha_t$ itself as a weight breaks the identity $a^2 + b^2 = 1$.
- **Dividing in the v branch.** The inverse of a rotation needs no division, so a division there is a sign that the two relations were combined incorrectly.
- **Falling through to a default branch.** An unknown prediction_type then quietly uses the wrong formulas instead of raising a ValueError.
- **Raising a KeyError for an unknown type.** A dictionary lookup on a missing key raises the wrong exception, which callers that catch ValueError will not handle.
- **Using one weight pair for the whole batch.** Each sample has its own timestep, so a shared pair converts every other sample with the wrong noise level.
- **Editing model_output in place.** The caller often keeps the raw network output for the loss, and an in-place update corrupts it.