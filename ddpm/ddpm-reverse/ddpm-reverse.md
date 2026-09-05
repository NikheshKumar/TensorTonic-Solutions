# <span style="font-size: 20px;">Reverse Diffusion Process</span>

## Introduction

Denoising Diffusion Probabilistic Models (DDPMs), introduced by Ho et al. (2020), generate data by learning to reverse a gradual noising process. The forward process corrupts a data sample $x_0$ over $T$ timesteps until it becomes indistinguishable from pure Gaussian noise $x_T \sim \mathcal{N}(0, I)$. The reverse process then walks backward from $x_T$ to $x_0$, denoising one step at a time.

The reverse diffusion step is where generation actually happens. Given a noisy sample $x_t$ and a neural network that predicts the noise $\epsilon_\theta(x_t, t)$, we compute the slightly-less-noisy sample $x_{t-1}$. This single reverse step, repeated from $t = T$ down to $t = 1$, transforms random noise into a coherent data sample.

---

## What It Is / What It Does

The reverse process is defined as a learned Markov chain with Gaussian transitions. Starting at $p(x_T) = \mathcal{N}(x_T; 0, I)$, the model applies a sequence of learned denoising steps $p_\theta(x_{t-1} | x_t)$, each parameterized as a Gaussian:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

At each timestep, the model takes the current noisy sample and produces a distribution over slightly cleaner samples. Sampling from this distribution gives $x_{t-1}$. The chain runs from $t = T$ down to $t = 1$, progressively removing noise until we arrive at $x_0$.

The key insight of DDPM is that instead of directly predicting the mean $\mu_\theta$, the neural network predicts the noise component $\epsilon_\theta(x_t, t)$. This noise prediction is plugged into a closed-form formula to compute the reverse step mean. The noise prediction objective turns out to be equivalent to optimizing a variational lower bound on the data log-likelihood, but with a simpler training signal.

---

## Key Equations

### The Reverse Step Formula

The complete reverse sampling step computes $x_{t-1}$ from $x_t$:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon_\theta(x_t, t) \right) + \sigma_t \, z$$

where $z \sim \mathcal{N}(0, I)$ for $t > 1$, and $z = 0$ for $t = 1$.

### Schedule Parameters

The noise schedule defines $\beta_t$ (a small positive constant at each timestep), from which all other quantities derive:

- $\alpha_t = 1 - \beta_t$ (the fraction of signal retained at step $t$)
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (the cumulative product, giving total signal retained from step 0 to step $t$)
- $\sigma_t = \sqrt{\beta_t}$ (the standard deviation of noise added in the reverse step)

### The Posterior Mean

The mean of the reverse transition is:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon_\theta(x_t, t) \right)$$

This is the deterministic part of the reverse step. The stochastic part $\sigma_t z$ adds controlled randomness to enable diverse sampling.

### The Boundary Condition

At the final step $t = 1$, we set $z = 0$. This means the last denoising step is purely deterministic: no noise is injected. The output is a clean sample $x_0$ without any additional stochastic perturbation.

---

## Deriving the Posterior Mean

### Where the Formula Comes From

The forward process gives us a closed-form expression for $x_t$ in terms of $x_0$ and noise:

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$. If we knew the true noise $\epsilon$ that was added, we could solve for $x_0$ exactly:

$$x_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon}{\sqrt{\bar{\alpha}_t}}$$

The neural network $\epsilon_\theta(x_t, t)$ approximates this true noise. Substituting the predicted noise gives an estimate $\hat{x}_0$, from which we derive the posterior mean for the reverse step.

### Role of Each Coefficient

The formula $\mu_\theta = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta \right)$ has two parts working together inside the parentheses.

**The subtraction $x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta$:** This removes the noise contribution from $x_t$. The coefficient $\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}$ scales the predicted noise to match exactly the amount added at timestep $t$, stripping away the noise component attributable to step $t$.

**The rescaling $\frac{1}{\sqrt{\alpha_t}}$:** After removing the noise, the remaining signal is slightly attenuated (because the forward step multiplied by $\sqrt{\alpha_t}$). Dividing by $\sqrt{\alpha_t}$ compensates for this attenuation, restoring the signal to its proper scale at timestep $t - 1$.

### Why Not Just Predict $x_0$ Directly?

Ho et al. found that predicting noise $\epsilon$ works much better than predicting $x_0$ directly. The noise prediction target has a more uniform distribution across timesteps and produces more stable gradients. The loss simplifies to:

$$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

This is a straightforward MSE between true and predicted noise, averaged over timesteps and training examples.

---

## Why Add Noise ($\sigma_t \cdot z$)

### Stochastic Sampling Enables Diversity

The noise term $\sigma_t z$ is not a flaw or approximation; it is essential to the generative process. Without it, the reverse process would be deterministic: every sample starting from the same $x_T$ would follow the same trajectory and produce the same $x_0$. The added noise at each step introduces stochasticity that allows the model to explore different modes of the data distribution.

Consider generating images of faces. The deterministic path from a given $x_T$ would always produce the same face. The injected noise introduces subtle variations at each step, letting different samples diverge into different faces, hairstyles, and expressions.

### What Happens Without Noise

If we set $z = 0$ at every step (not just $t = 1$), the resulting deterministic sampler tends to produce blurry, averaged outputs. The model's predicted mean $\mu_\theta$ represents the center of the posterior distribution $p_\theta(x_{t-1} | x_t)$, which averages over many possible denoised outcomes. Without sampling from the distribution, we always pick this average, and averages of sharp images are blurry. The noise injection forces the model to commit to specific, sharp details at each step rather than hedging.

### Why $z = 0$ at $t = 1$

At the very last step ($t = 1$), we want a clean final output. Adding noise at this stage would degrade the generated sample with unnecessary perturbation. The variance $\sigma_1^2 = \beta_1$ is typically very small, so the effect would be minor, but setting $z = 0$ ensures a crisp output. There is no $x_{-1}$ to transition to, so we want the sharpest possible estimate of $x_0$.

---

## Paper Context

### Ho et al.'s Formulation

In the original DDPM paper, Ho et al. (2020) define the reverse process variance as $\sigma_t^2 = \beta_t$. This is the "simplified" variance choice. The true posterior variance, conditioned on knowing $x_0$, is:

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

Ho et al. found that both choices produce comparable results. $\beta_t$ is an upper bound and $\tilde{\beta}_t$ is a lower bound on the optimal reverse process variance. Nichol and Dhariwal (2021) later showed that learning the variance (interpolating between these bounds) can improve sample quality.

### Connection to the Variational Lower Bound

The DDPM training objective derives from a variational lower bound (VLB) on the data log-likelihood $\log p_\theta(x_0)$. The VLB decomposes into KL divergences between the true posterior $q(x_{t-1} | x_t, x_0)$ and the learned reverse transition $p_\theta(x_{t-1} | x_t)$. Since both distributions are Gaussian, the KL reduces to comparing means (and variances, if learned). The simplified loss $L_{\text{simple}}$ drops the variance terms, keeping only mean-matching. Despite being a looser bound, this simplified loss produces better sample quality in practice.

### The Noise Schedule

Ho et al. use a linear schedule for $\beta_t$, interpolating from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ steps. The small $\beta_t$ values ensure each forward step adds only a small amount of noise, making the reverse step easier to learn. Later work explored cosine schedules that provide a more uniform signal-to-noise ratio across timesteps.

---

## Numerical Example

### Setup: Middle Timestep ($t = 500$)

Let us walk through the reverse step formula with concrete numbers. Suppose we have a 4-dimensional vector at timestep $t = 500$.

For a linear schedule from $\beta_1 = 0.0001$ to $\beta_T = 0.02$ with $T = 1000$:

$$\beta_{500} = 0.0001 + \frac{499}{999}(0.02 - 0.0001) = 0.0001 + 0.4995 \times 0.0199 \approx 0.01004$$

From this:

- $\alpha_{500} = 1 - \beta_{500} = 1 - 0.01004 = 0.98996$
- $\bar{\alpha}_{500} \approx 0.0495$ (the cumulative product of all $\alpha_s$ from $s = 1$ to $500$)
- $\sigma_{500} = \sqrt{\beta_{500}} = \sqrt{0.01004} \approx 0.10020$

The key derived coefficients:

- $\frac{1}{\sqrt{\alpha_{500}}} = \frac{1}{\sqrt{0.98996}} \approx 1.00507$
- $\frac{1 - \alpha_{500}}{\sqrt{1 - \bar{\alpha}_{500}}} = \frac{0.01004}{\sqrt{0.9505}} = \frac{0.01004}{0.97493} \approx 0.01030$

### The Computation

Given:

$$x_{500} = [0.82, -1.45, 0.33, 2.10]$$

$$\epsilon_\theta(x_{500}, 500) = [0.50, -0.80, 1.20, 0.35]$$

$$z = [-0.62, 0.91, 0.15, -1.30]$$

**Step 1: Scale the predicted noise.**

$$\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta = 0.01030 \times [0.50, -0.80, 1.20, 0.35]$$

$$= [0.00515, -0.00824, 0.01236, 0.00361]$$

**Step 2: Subtract the noise contribution from $x_t$.**

$$x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta = [0.82 - 0.00515, \; -1.45 + 0.00824, \; 0.33 - 0.01236, \; 2.10 - 0.00361]$$

$$= [0.81485, -1.44176, 0.31764, 2.09639]$$

**Step 3: Rescale by $\frac{1}{\sqrt{\alpha_t}}$.**

$$\mu_\theta = 1.00507 \times [0.81485, -1.44176, 0.31764, 2.09639]$$

$$= [0.81898, -1.44907, 0.31925, 2.10702]$$

**Step 4: Add the stochastic noise.**

$$\sigma_t z = 0.10020 \times [-0.62, 0.91, 0.15, -1.30]$$

$$= [-0.06212, 0.09118, 0.01503, -0.13026]$$

**Step 5: Combine to get $x_{499}$.**

$$x_{499} = \mu_\theta + \sigma_t z = [0.81898 - 0.06212, \; -1.44907 + 0.09118, \; 0.31925 + 0.01503, \; 2.10702 - 0.13026]$$

$$= [0.75686, -1.35789, 0.33428, 1.97676]$$

Each reverse step makes only a small adjustment. Over hundreds of steps, these small adjustments accumulate to transform pure noise into structured data.

### Final Timestep ($t = 1$, No Noise)

Now consider the very last reverse step. At $t = 1$ with the linear schedule:

$$\beta_1 = 0.0001, \quad \alpha_1 = 0.9999, \quad \bar{\alpha}_1 = 0.9999$$

Derived coefficients:

- $\frac{1}{\sqrt{\alpha_1}} = \frac{1}{\sqrt{0.9999}} \approx 1.00005$
- $\frac{1 - \alpha_1}{\sqrt{1 - \bar{\alpha}_1}} = \frac{0.0001}{\sqrt{0.0001}} = \frac{0.0001}{0.01} = 0.01$

Given:

$$x_1 = [0.52, -0.38, 1.15, -0.74]$$

$$\epsilon_\theta(x_1, 1) = [0.20, -0.55, 0.90, 0.10]$$

**Step 1: Scale the predicted noise.**

$$0.01 \times [0.20, -0.55, 0.90, 0.10] = [0.00200, -0.00550, 0.00900, 0.00100]$$

**Step 2: Subtract from $x_1$.**

$$[0.52 - 0.00200, \; -0.38 + 0.00550, \; 1.15 - 0.00900, \; -0.74 - 0.00100]$$

$$= [0.51800, -0.37450, 1.14100, -0.74100]$$

**Step 3: Rescale.**

$$\mu_\theta = 1.00005 \times [0.51800, -0.37450, 1.14100, -0.74100]$$

$$= [0.51803, -0.37452, 1.14106, -0.74104]$$

**Step 4: Set $z = 0$ (no noise at $t = 1$).**

$$x_0 = \mu_\theta = [0.51803, -0.37452, 1.14106, -0.74104]$$

The final output is the clean generated sample. The adjustments at $t = 1$ are tiny because the signal-to-noise ratio is already very high ($\bar{\alpha}_1 = 0.9999$).

---

## Connection to Score Matching

### Noise Prediction as Score Estimation

The predicted noise $\epsilon_\theta(x_t, t)$ is intimately connected to the score function $\nabla_{x_t} \log p_t(x_t)$, which is the gradient of the log-density of the noisy data distribution at timestep $t$. The relationship is:

$$\nabla_{x_t} \log p_t(x_t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

The score function points in the direction of increasing data density. The predicted noise points in the opposite direction (toward the noise that was added). These are the same information up to a known scaling factor.

### Score-Based Generative Models

Song and Ermon (2019, 2020) developed score-based generative models independently, using a continuous-time formulation with stochastic differential equations (SDEs). Song et al. (2021) showed that DDPMs are a discretization of this continuous framework: the DDPM reverse step is an Euler-Maruyama discretization of the reverse SDE, and the noise prediction network $\epsilon_\theta$ is equivalent to a score network $s_\theta$ up to the scaling factor above.

### Practical Implications

This equivalence means insights from score matching transfer to DDPMs and vice versa. The DDPM sampling procedure can be replaced with more sophisticated SDE solvers (like predictor-corrector methods) to improve sample quality. The connection also motivates the probability flow ODE, a deterministic counterpart to the stochastic reverse SDE that enables exact likelihood computation. DDIM (Song et al., 2020) can be viewed as a discretization of this probability flow ODE.

---

## Pitfalls and Common Mistakes

### Forgetting $z = 0$ at $t = 1$

The most common implementation bug is sampling $z \sim \mathcal{N}(0, I)$ at every timestep, including $t = 1$. This adds a small random perturbation to the final output, producing slightly noisy samples. In code, this is typically handled with a simple conditional:

$$z = \begin{cases} \mathcal{N}(0, I) & \text{if } t > 1 \\ 0 & \text{if } t = 1 \end{cases}$$

The effect may be subtle (since $\sigma_1 = \sqrt{\beta_1}$ is small), but it is incorrect and can measurably degrade FID scores.

### Confusing $\alpha_t$ with $\bar{\alpha}_t$

This is a persistent source of bugs. The two quantities are fundamentally different:

- $\alpha_t = 1 - \beta_t$ is the single-step retention factor (close to 1 for all $t$)
- $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ is the cumulative retention factor (decreases from nearly 1 at $t = 1$ to nearly 0 at $t = T$)

Using $\alpha_t$ where $\bar{\alpha}_t$ belongs (or vice versa) will produce wildly wrong results. For example, at $t = 500$, $\alpha_{500} \approx 0.99$ while $\bar{\alpha}_{500} \approx 0.05$. Swapping these in the coefficient $\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}$ changes the noise removal scale by an order of magnitude.

A helpful mnemonic: $\alpha_t$ (no bar) is the local, single-step quantity; $\bar{\alpha}_t$ (with bar) is the global, cumulative quantity. The forward formula $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ uses the bar version because it describes total corruption from step 0 to step $t$.

### Wrong Variance ($\sigma_t$ Formula)

Ho et al. use $\sigma_t^2 = \beta_t$, so $\sigma_t = \sqrt{\beta_t}$. A common mistake is using $\sigma_t = \beta_t$ (forgetting the square root), which makes the injected noise far too small. Another error is using $\sigma_t^2 = \tilde{\beta}_t$ without realizing it requires $\bar{\alpha}_{t-1}$, introducing an off-by-one issue at $t = 1$ where $\bar{\alpha}_0$ must be defined as 1.

### Reverse Step Order: Must Go $T$ to $1$

The reverse process must iterate from $t = T$ down to $t = 1$, not from $1$ to $T$. The coefficients are calibrated for the specific noise level at each timestep. Running the steps in the wrong order applies denoising coefficients at mismatched noise levels, producing garbage. In Python:

$$\text{for } t \text{ in } [T, T{-}1, \ldots, 2, 1]: \quad x_{t-1} = \text{reverse\_step}(x_t, t)$$

### Numerical Instability at Small $t$

At very small timesteps (near $t = 1$), $\bar{\alpha}_t$ is close to 1 and $1 - \bar{\alpha}_t$ is close to 0. Computing $\sqrt{1 - \bar{\alpha}_t}$ can lose numerical precision, and dividing by it amplifies errors. Clamping $1 - \bar{\alpha}_t$ to a small positive minimum (e.g., $10^{-8}$) prevents division-by-zero issues. Precomputing all schedule quantities in float64 before casting to float32 for model inference is a common best practice.

### Off-by-One Errors in Schedule Indexing

Different codebases index $\beta_t$ from $t = 0$ to $T - 1$ or from $t = 1$ to $T$. This leads to off-by-one errors when looking up $\bar{\alpha}_t$ or $\bar{\alpha}_{t-1}$. The safest approach is to precompute arrays of all needed coefficients and verify indexing with a known test case (e.g., checking that $\bar{\alpha}_T \approx 0$ and $\bar{\alpha}_1 \approx 1$).