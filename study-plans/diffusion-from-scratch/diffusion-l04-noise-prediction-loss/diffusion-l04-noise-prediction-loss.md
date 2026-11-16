# Noise Prediction Loss

A diffusion model learns by playing a guessing game with itself. You take a clean training example, mix in random noise that you drew yourself, and ask the network which noise you added. Because you made the noise, you always know the right answer, and the gap between the guess and the truth becomes the training signal. Think of a teacher who smudges a page with a known stain and asks a student to point out exactly where the stain is. This page builds that loss one piece at a time, starting from one number and ending with the batch loss of a small linear denoiser, which is what you implement in this problem.

## One noisy number and one guess

Recall the forward noising formula. A clean value $x_0$ and a standard Gaussian noise value $\epsilon$ are blended with a signal weight $\sqrt{\bar\alpha_t}$ and a noise weight $\sqrt{1 - \bar\alpha_t}$, where $\bar\alpha_t$ is the fraction of variance that still comes from the clean data at timestep $t$.

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon
$$

Take $x_0 = 2$, $\epsilon = -0.5$ and $\bar\alpha_t = 0.64$, so the two weights are $0.8$ and $0.6$. The noisy value is $0.8 \cdot 2 + 0.6 \cdot (-0.5) = 1.6 - 0.3 = 1.3$. The network only sees $1.3$ and the timestep, and it has to say how much of that value is noise.

Suppose it guesses $\hat\epsilon = -0.2$. The true noise was $-0.5$, so the error is $0.3$ and the squared error is $0.09$. That squared error is the whole idea of the loss in its smallest form. A perfect guess gives 0, and a guess that is far from the drawn noise gives a large number that the optimizer pushes down.

## Why the network guesses the noise

A diffusion model generates data by undoing the noise step by step, and each undo step needs an estimate of the clean sample hidden inside $x_t$. The forward formula ties the three quantities together, so once $x_t$ is known, a guess for $\epsilon$ is also a guess for $x_0$. Predicting the noise is therefore a complete answer to the denoising question.

The noise is also a convenient target. It is always standard Gaussian, so the regression target has variance 1 at every timestep. A network asked for $x_0$ directly would see a target that is trivial at low noise and nearly impossible at high noise, which makes the loss jump around in scale.

Ho et al. (2020) derived the full variational training objective of a diffusion model and found that it reduces, up to one weight per timestep, to a squared error between the true and predicted noise. They then dropped those weights and trained on the plain mean squared error, which they called the simple loss. It gave better samples than the exact bound and became the default starting point for diffusion training.

## The same noise plays two roles

The detail that makes this game work is that one noise tensor is used twice. It is first mixed into the clean sample to build $x_t$, and it is then used again as the label that the prediction is compared with. If the label came from a fresh draw, the network would be asked to predict random numbers that have nothing to do with its input, and the best it could do is predict 0 everywhere, which leaves a loss of about 1 per feature.

During training each sample in a batch gets its own timestep drawn at random and its own noise. The timesteps are 0-based, so a schedule with $T$ steps stores $\bar\alpha_0$ to $\bar\alpha_{T-1}$ in alpha_bars, and the value for sample $i$ is the entry at its own timestep $t_i$. Different samples in one batch therefore use different signal and noise weights, and each pair of weights scales every element of its own sample.

## A toy denoiser

A real denoiser is a U-Net or a transformer, but the loss does not care what the network looks like. It only needs a prediction with the shape of the noise. This problem uses the smallest network that still has the right ingredients, a linear map on the flattened sample plus a time feature and a bias.

$$
\hat\epsilon = x_t W + \frac{t}{T}\, w_t + b
$$

Each sample is first flattened to $D$ features, so $x_t$ has shape $(B, D)$. The matrix $W$ (weight) has shape $(D, D)$ and multiplies $x_t$ from the right. The vectors $w_t$ (time_weight) and $b$ (bias) both have shape $(D)$.

The term $(t / T)\, w_t$ is the simplest possible way to tell the network the noise level. The time feature $t / T$ is 0 at the first timestep and grows towards 1 at the last, and dividing by $T$ keeps it in the same range whatever the schedule length. Real models use a sinusoidal timestep embedding instead, but its job is the same. The network has to know the timestep, because the same noisy input means different things at different noise levels.

## From one sample to a batch loss

For one flattened sample $i$, the squared error is a vector of $D$ numbers, one per feature. Averaging those $D$ numbers gives one loss value for the sample.

$$
\ell_i = \frac{1}{D} \sum_{d=1}^{D} \left(\hat\epsilon_{i,d} - \epsilon_{i,d}\right)^2
$$

The batch loss $L$ is then the mean of the $B$ per-sample values. Because every sample has the same $D$, this equals the mean over all $B \cdot D$ entries, so the order of the two means does not change $L$ itself.

Keeping the per-sample values is still useful. Loss weighting schemes such as Min-SNR multiply each $\ell_i$ by a weight that depends on its timestep, and training dashboards often group $\ell_i$ by timestep to show which noise levels the model finds hard. A mean over features also keeps the loss on the same scale when the image resolution changes, while a sum over features would multiply it by $D$ and change the effective learning rate.

## A two-sample example

Use a schedule with $T = 4$ and alpha_bars $= (0.81, 0.64, 0.36, 0.25)$. The denoiser has $D = 2$, with rows $(0.5, 0.1)$ and $(0, 0.5)$ for $W$, $w_t = (0.4, 0.4)$ and $b = (0.1, 0)$. Because $W$ multiplies from the right, the first output is $0.5 x_1$ and the second is $0.1 x_1 + 0.5 x_2$.

The first sample has $x_0 = (1, -2)$, noise $\epsilon = (0.5, -1)$ and timestep $t = 1$, so the weights are $0.8$ and $0.6$.

1. The noisy sample is $x_t = (0.8 + 0.3, -1.6 - 0.6) = (1.1, -2.2)$.
2. The linear part is $x_t W = (0.55, 0.11 - 1.1) = (0.55, -0.99)$.
3. The time feature is $1 / 4 = 0.25$, which adds $(0.1, 0.1)$, and the bias adds $(0.1, 0)$, so $\hat\epsilon = (0.75, -0.89)$.
4. The squared errors are $0.25^2 = 0.0625$ and $0.11^2 = 0.0121$, so $\ell_1 = 0.0373$.

The second sample has $x_0 = (0, 1)$, noise $\epsilon = (1, 0.5)$ and timestep $t = 2$, so the weights are $0.6$ and $0.8$.

1. The noisy sample is $x_t = (0.8, 0.6 + 0.4) = (0.8, 1.0)$.
2. The linear part is $x_t W = (0.4, 0.08 + 0.5) = (0.4, 0.58)$.
3. The time feature is $0.5$, which adds $(0.2, 0.2)$, and with the bias $\hat\epsilon = (0.7, 0.78)$.
4. The squared errors are $0.09$ and $0.0784$, so $\ell_2 = 0.0842$.

The batch loss is $(0.0373 + 0.0842) / 2 = 0.06075$. The two samples used different weights and different time features, and the only thing they share is the denoiser.

## What the ends of the schedule teach

At $t = 0$ the time feature is exactly 0 and $\bar\alpha_0$ is usually very close to 1. The noisy sample is then almost the clean sample, and the network has to find a faint noise pattern inside a strong signal.

At the other end, a schedule that reaches $\bar\alpha = 0$ makes $x_t$ equal to $\epsilon$ exactly. The target can then be read straight off the input, so an identity map scores a perfect loss and the timestep teaches the network nothing about the data. This is one reason the last timesteps contribute little learning signal under noise prediction, and it motivates the other targets and weightings later in the plan.

## Where this loss sits in training

This loss is the entire training objective of a basic DDPM. Each training step noises a batch, runs the network once, computes this loss and follows its gradient with respect to the network weights. Later problems change what the network predicts and how the per-sample losses are weighted, but they keep this structure.

## Common mistakes to avoid

- **Comparing with a fresh noise draw.** The label must be the exact noise that built $x_t$, otherwise the network is asked to predict numbers unrelated to its input.
- **Summing the squared error over features.** The loss then grows with the number of features, which changes the effective learning rate for every input size.
- **Dividing the timestep by T minus 1.** The time feature then reaches 1 at the last step and every prediction with $t > 0$ is shifted.
- **Applying the weight matrix transposed.** With a non-symmetric matrix, multiplying by its transpose gives a different linear map and a different loss.
- **Feeding the clean sample to the denoiser.** The network must see $x_t$, because the clean sample is unknown when the model generates data.
- **Using one timestep for the whole batch.** Each sample has its own timestep, so a shared signal weight gives every other sample the wrong noisy input.
- **Returning the batch loss as a tensor.** A 0-d tensor still carries its graph and device, and the batch loss is expected as a plain Python float.