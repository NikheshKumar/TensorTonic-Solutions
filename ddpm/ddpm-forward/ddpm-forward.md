# <span style="font-size: 20px;">Forward Diffusion Process</span>

<span style="font-size: 14px;">The forward diffusion process is the data-corrupting half of denoising diffusion probabilistic models (DDPMs). Introduced by Ho et al. (2020) in "Denoising Diffusion Probabilistic Models," it defines a fixed Markov chain that gradually adds Gaussian noise to a clean data sample over $T$ timesteps until the sample becomes indistinguishable from pure noise. The forward process has no learnable parameters. Its sole purpose is to produce noisy training pairs that teach a neural network to reverse the corruption.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">The forward process takes a clean data sample $x_0$ drawn from the real data distribution $q(x_0)$ and produces a sequence of increasingly noisy versions $x_1, x_2, \ldots, x_T$. At each timestep $t$, a small amount of Gaussian noise is added to the previous sample according to a fixed variance schedule. By the final timestep $T$, the sample has been corrupted so thoroughly that it is essentially a draw from a standard normal distribution $\mathcal{N}(0, I)$.</span>

<span style="font-size: 14px;">This is a **fixed** process. Unlike the reverse process (which uses a learned neural network to denoise), the forward process requires no training. The noise schedule $\beta_1, \beta_2, \ldots, \beta_T$ is set before training begins and never changes. Ho et al. describe this explicitly: "We define the forward process as a fixed Markov chain that gradually adds Gaussian noise to the data according to a variance schedule $\beta_1, \ldots, \beta_T$."</span>

<span style="font-size: 14px;">The Markov property means that each step depends only on the immediately preceding sample, not on any earlier timesteps. This is what makes the forward process mathematically tractable and allows the closed-form sampling trick that avoids iterating through all $t$ steps.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The forward process is defined by a single conditional Gaussian at each step:</span>

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\right)
$$

<span style="font-size: 14px;">This says: to get $x_t$ from $x_{t-1}$, scale down the previous sample by $\sqrt{1 - \beta_t}$ (shrinking the signal slightly) and add Gaussian noise with variance $\beta_t$. The parameter $\beta_t$ controls how much noise is injected at step $t$.</span>

<span style="font-size: 14px;">Using the **reparameterization trick**, we can write this as a deterministic function of $x_{t-1}$ and a noise sample:</span>

$$
x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

<span style="font-size: 14px;">To build notation for the closed-form result, define two auxiliary quantities:</span>

$$
\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

<span style="font-size: 14px;">Here $\alpha_t$ is the fraction of signal preserved at step $t$ (always slightly less than 1), and $\bar{\alpha}_t$ (alpha-bar) is the cumulative product from step 1 through $t$. Because each $\alpha_s < 1$, $\bar{\alpha}_t$ decreases monotonically toward zero as $t$ grows.</span>

<span style="font-size: 14px;">The joint distribution over the full forward chain factors as:</span>

$$
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1})
$$

<span style="font-size: 14px;">This factorization follows directly from the Markov property.</span>

---

## <span style="font-size: 16px;">The Closed-Form Trick</span>

<span style="font-size: 14px;">The most important mathematical insight in the forward process is that you do **not** need to iterate through all $t$ steps to sample $x_t$. You can jump directly from $x_0$ to $x_t$ in a single computation.</span>

<span style="font-size: 14px;">Here is the derivation. Start with one step:</span>

$$
x_1 = \sqrt{\alpha_1}\, x_0 + \sqrt{1 - \alpha_1}\, \epsilon_1
$$

<span style="font-size: 14px;">Apply the transition again to get $x_2$:</span>

$$
x_2 = \sqrt{\alpha_2}\, x_1 + \sqrt{1 - \alpha_2}\, \epsilon_2
$$

<span style="font-size: 14px;">Substitute the expression for $x_1$:</span>

$$
x_2 = \sqrt{\alpha_2}\left(\sqrt{\alpha_1}\, x_0 + \sqrt{1 - \alpha_1}\, \epsilon_1\right) + \sqrt{1 - \alpha_2}\, \epsilon_2
$$

$$
= \sqrt{\alpha_1 \alpha_2}\, x_0 + \sqrt{\alpha_2(1 - \alpha_1)}\, \epsilon_1 + \sqrt{1 - \alpha_2}\, \epsilon_2
$$

<span style="font-size: 14px;">The last two terms are independent Gaussians. The sum of independent Gaussians $\mathcal{N}(0, \sigma_1^2 I)$ and $\mathcal{N}(0, \sigma_2^2 I)$ is Gaussian with variance $\sigma_1^2 + \sigma_2^2$. The combined variance is:</span>

$$
\alpha_2(1 - \alpha_1) + (1 - \alpha_2) = \alpha_2 - \alpha_1\alpha_2 + 1 - \alpha_2 = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2
$$

<span style="font-size: 14px;">So we can replace the two noise terms with a single draw:</span>

$$
x_2 = \sqrt{\bar{\alpha}_2}\, x_0 + \sqrt{1 - \bar{\alpha}_2}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

<span style="font-size: 14px;">This pattern holds for any timestep $t$ by induction. The general closed-form result is:</span>

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) I\right)
$$

<span style="font-size: 14px;">Or equivalently, using the reparameterization trick:</span>

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

<span style="font-size: 14px;">This is the equation you will implement. It takes three inputs ($x_0$, $t$, and $\epsilon$) and produces $x_t$ in a single vectorized operation. No loops, no iterating through intermediate timesteps.</span>

---

## <span style="font-size: 16px;">What the Coefficients Mean</span>

<span style="font-size: 14px;">The closed-form equation $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$ is a weighted combination of two components: the original clean data and pure noise. The coefficients control the **signal-to-noise ratio** at each timestep.</span>

* <span style="font-size: 14px;">$\sqrt{\bar{\alpha}_t}$ scales the **signal** (the original data $x_0$). When $t$ is small, $\bar{\alpha}_t$ is close to 1, so this coefficient is close to 1, preserving most of the original data.</span>
* <span style="font-size: 14px;">$\sqrt{1 - \bar{\alpha}_t}$ scales the **noise** ($\epsilon$). When $t$ is small, $1 - \bar{\alpha}_t$ is close to 0, so very little noise is added.</span>

<span style="font-size: 14px;">As $t$ increases toward $T$, the cumulative product $\bar{\alpha}_t$ shrinks toward zero. The signal coefficient drops and the noise coefficient rises. At the extremes:</span>

* <span style="font-size: 14px;">**At $t = 0$**: $\bar{\alpha}_0 = 1$ by convention, so $x_0 = 1 \cdot x_0 + 0 \cdot \epsilon = x_0$. The sample is perfectly clean.</span>
* <span style="font-size: 14px;">**At $t = T$**: $\bar{\alpha}_T \approx 0$ (for properly chosen schedules), so $x_T \approx 0 \cdot x_0 + 1 \cdot \epsilon = \epsilon$. The sample is pure noise.</span>

<span style="font-size: 14px;">Notice that the two coefficients satisfy $(\sqrt{\bar{\alpha}_t})^2 + (\sqrt{1 - \bar{\alpha}_t})^2 = \bar{\alpha}_t + 1 - \bar{\alpha}_t = 1$. This is a **variance-preserving** property. If $x_0$ has unit variance and $\epsilon$ has unit variance, then $x_t$ also has unit variance at every timestep. The total energy in the sample stays constant; only the proportion that comes from signal versus noise changes.</span>

<span style="font-size: 14px;">This variance-preserving design is deliberate. Without it, the magnitude of $x_t$ would grow or shrink over time, making it harder for the denoising network to operate consistently across timesteps.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Ho et al. (2020) built on the theoretical framework of Sohl-Dickstein et al. (2015), who first proposed using a diffusion process for generative modeling. The key contribution of Ho et al. was showing that a simplified training objective and specific architectural choices could make diffusion models produce high-quality images competitive with GANs.</span>

<span style="font-size: 14px;">In the original paper, the forward process uses a **linear variance schedule** where $\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ timesteps. This means the noise injection starts very gently (tiny $\beta$ values in early steps) and accelerates toward the end. The linear schedule was chosen empirically; later work by Nichol and Dhariwal (2021) introduced a cosine schedule that distributes the noise more evenly across timesteps and improves sample quality.</span>

<span style="font-size: 14px;">The forward process being fixed (not learned) is a deliberate design choice. Sohl-Dickstein et al. experimented with learning the noise schedule, but Ho et al. found that fixing it simplifies training enormously. With a fixed forward process, the only learnable component is the reverse denoising network. The number of timesteps $T = 1000$ is large by design: more steps means each individual step adds only a tiny amount of noise, making the reverse transition closer to Gaussian, which is a key assumption in the derivation of the reverse process.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Suppose we have a 2D data point $x_0 = [0.5, -0.3]$ and we sample noise $\epsilon = [0.8, -1.2]$. We will compute $x_t$ at three different timesteps to see how the signal degrades progressively.</span>

<span style="font-size: 14px;">For a typical linear schedule with $T = 1000$, the approximate values of $\bar{\alpha}_t$ at selected timesteps are:</span>

* <span style="font-size: 14px;">$\bar{\alpha}_{100} \approx 0.9048$</span>
* <span style="font-size: 14px;">$\bar{\alpha}_{500} \approx 0.0821$</span>
* <span style="font-size: 14px;">$\bar{\alpha}_{1000} \approx 0.0001$</span>

### <span style="font-size: 14px;">At $t = 100$</span>

$$
x_{100} = \sqrt{0.9048} \cdot [0.5, -0.3] + \sqrt{1 - 0.9048} \cdot [0.8, -1.2]
$$

$$
= 0.9513 \cdot [0.5, -0.3] + 0.3087 \cdot [0.8, -1.2]
$$

$$
= [0.4756, -0.2854] + [0.2469, -0.3704]
$$

$$
= [0.7226, -0.6558]
$$

<span style="font-size: 14px;">The result is still close to the original. The signal coefficient (0.9513) dominates the noise coefficient (0.3087), so the original structure is largely intact.</span>

### <span style="font-size: 14px;">At $t = 500$</span>

$$
x_{500} = \sqrt{0.0821} \cdot [0.5, -0.3] + \sqrt{1 - 0.0821} \cdot [0.8, -1.2]
$$

$$
= 0.2865 \cdot [0.5, -0.3] + 0.9580 \cdot [0.8, -1.2]
$$

$$
= [0.1433, -0.0860] + [0.7664, -1.1496]
$$

$$
= [0.9097, -1.2356]
$$

<span style="font-size: 14px;">Now the noise dominates. The signal coefficient (0.2865) is dwarfed by the noise coefficient (0.9580). The values bear little resemblance to the original $[0.5, -0.3]$.</span>

### <span style="font-size: 14px;">At $t = 1000$</span>

$$
x_{1000} = \sqrt{0.0001} \cdot [0.5, -0.3] + \sqrt{1 - 0.0001} \cdot [0.8, -1.2]
$$

$$
= 0.01 \cdot [0.5, -0.3] + 0.99995 \cdot [0.8, -1.2]
$$

$$
= [0.005, -0.003] + [0.7999, -1.1999]
$$

$$
= [0.8049, -1.2029]
$$

<span style="font-size: 14px;">The original signal contributes only $[0.005, -0.003]$, which is negligible. The output is essentially the noise vector itself, confirming that at $t = T$ the forward process has destroyed all information about $x_0$. This progressive corruption is the core idea: the forward process creates a smooth bridge from structured data to unstructured noise, and the reverse process learns to walk that bridge backward.</span>

---

## <span style="font-size: 16px;">Why Return Both $x_t$ and $\epsilon$</span>

<span style="font-size: 14px;">When implementing the forward process for training, the function must return **both** the noisy sample $x_t$ and the noise vector $\epsilon$ that was used to create it. This is not optional; both are required by the training objective.</span>

<span style="font-size: 14px;">The simplified training loss from Ho et al. is:</span>

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\left\| \epsilon - \epsilon_\theta(x_t, t) \right\|^2\right]
$$

<span style="font-size: 14px;">Here $\epsilon_\theta(x_t, t)$ is the neural network's prediction of the noise, given the noisy input $x_t$ and the timestep $t$. The loss compares this prediction against the **actual noise** $\epsilon$ that was sampled during the forward process.</span>

<span style="font-size: 14px;">In a single training step, the flow is:</span>

<span style="font-size: 14px;">1. **Sample** a clean image $x_0$ from the dataset</span>

<span style="font-size: 14px;">2. **Sample** a random timestep $t \sim \text{Uniform}(\{1, 2, \ldots, T\})$</span>

<span style="font-size: 14px;">3. **Sample** noise $\epsilon \sim \mathcal{N}(0, I)$</span>

<span style="font-size: 14px;">4. **Compute** $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$ (forward process)</span>

<span style="font-size: 14px;">5. **Predict** $\hat{\epsilon} = \epsilon_\theta(x_t, t)$ (neural network forward pass)</span>

<span style="font-size: 14px;">6. **Compute loss** $\|\epsilon - \hat{\epsilon}\|^2$ (MSE between true and predicted noise)</span>

<span style="font-size: 14px;">Step 4 uses $\epsilon$ to create $x_t$. Step 6 uses the same $\epsilon$ as the regression target. If the forward function only returned $x_t$, you would have no way to compute the loss. This is conceptually different from classification, where labels come from the dataset. In DDPM training, the "label" ($\epsilon$) is generated on the fly as part of the forward process itself, so the forward function simultaneously creates the input ($x_t$) and the target ($\epsilon$).</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Confusing $\alpha_t$ with $\bar{\alpha}_t$.** $\alpha_t = 1 - \beta_t$ is the per-step signal retention factor. $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ is the cumulative product. The closed-form equation uses $\bar{\alpha}_t$, not $\alpha_t$. Using $\alpha_t$ instead means you compute only a single step of noise addition rather than the cumulative effect of $t$ steps, making the result far less noisy than it should be.</span>

* <span style="font-size: 14px;">**Swapping the signal and noise coefficients.** The correct formula is $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$. Writing $\sqrt{1 - \bar{\alpha}_t}\, x_0 + \sqrt{\bar{\alpha}_t}\, \epsilon$ reverses the roles: early timesteps would be mostly noise and late timesteps mostly signal. This produces a process that starts noisy and gets cleaner, the opposite of what the forward process should do.</span>

* <span style="font-size: 14px;">**Forgetting to return $\epsilon$.** If the forward function returns only $x_t$, the training loop cannot compute the loss $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ because it has no access to the ground-truth noise. The implementer might then sample new noise for the loss target, which destroys the correspondence between input and label.</span>

* <span style="font-size: 14px;">**Iterating through all $t$ steps instead of using the closed form.** A naive implementation might loop from $x_0$ to $x_1$ to $x_2$ and so on up to $x_t$. This is mathematically correct but absurdly slow: for $T = 1000$, you run 1000 sequential Gaussian samples per image instead of a single vectorized operation. The closed form exists precisely to avoid this.</span>

* <span style="font-size: 14px;">**Dimension mismatch between $x_0$ and $\epsilon$.** The noise $\epsilon$ must have exactly the same shape as $x_0$. For image data, this means $\epsilon \in \mathbb{R}^{B \times C \times H \times W}$. Sampling noise with the wrong shape will either crash with a broadcasting error or silently produce incorrect results through unintended broadcasting.</span>

* <span style="font-size: 14px;">**Using the wrong distribution for $\epsilon$.** The noise must be drawn from a standard normal $\mathcal{N}(0, I)$, not a uniform distribution, not a Bernoulli, and not a normal with non-unit variance. The entire derivation assumes Gaussian noise; using any other distribution breaks the closed-form result and invalidates the training objective.</span>

* <span style="font-size: 14px;">**Applying $\beta_t$ directly as the noise coefficient.** Writing $x_t = \sqrt{1 - \beta_t}\, x_0 + \sqrt{\beta_t}\, \epsilon$ uses the single-step transition formula (from $x_{t-1}$ to $x_t$), not the closed-form jump from $x_0$ to $x_t$. This drastically underestimates the total noise at any timestep beyond $t = 1$.</span>

---