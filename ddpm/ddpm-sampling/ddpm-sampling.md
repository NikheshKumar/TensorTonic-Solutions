# <span style="font-size: 20px;">DDPM Sampling</span>

<span style="font-size: 14px;">Sampling is the generative act of a diffusion model: starting from pure Gaussian noise and progressively removing it to produce a clean data sample. In the Denoising Diffusion Probabilistic Model (DDPM) framework introduced by Ho, Jain, and Abbeel (2020), sampling runs the learned reverse Markov chain from timestep $T$ all the way down to timestep $1$, inverting the forward diffusion process that gradually destroyed the data's structure.</span>

<span style="font-size: 14px;">Unlike GANs, which generate samples in a single forward pass through a generator network, DDPM sampling is an iterative procedure. Each step asks the model to predict and subtract a small amount of noise, gradually revealing the underlying data distribution one denoising step at a time.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The DDPM sampling pipeline is a complete generation procedure that transforms a random noise vector into a realistic data sample. It consists of three stages: (1) sample initial noise $\mathbf{x}_T$ from a standard Gaussian, (2) iteratively apply the learned reverse denoising transition from $t = T$ down to $t = 1$, and (3) output the final $\mathbf{x}_0$ as the generated sample.</span>

<span style="font-size: 14px;">At each step, a neural network $\epsilon_\theta(\mathbf{x}_t, t)$ predicts the noise component present in $\mathbf{x}_t$. This prediction is used to compute the mean of the reverse transition distribution, from which $\mathbf{x}_{t-1}$ is sampled. The noise network was trained to minimize the difference between its prediction and the actual noise that was added during the forward process.</span>

<span style="font-size: 14px;">The procedure is stochastic: at every step except the last, fresh Gaussian noise $\mathbf{z}$ is added to the denoised estimate. This injects randomness that allows the model to explore diverse modes of the data distribution. At $t = 1$, no noise is added because we want the final output to be a clean sample, not a noisy one.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The noise schedule defines a sequence of values $\beta_1, \beta_2, \dots, \beta_T$ (small positive constants, typically $\beta_1 = 10^{-4}$ and $\beta_T = 0.02$ with linear interpolation). From these, we derive:</span>

* <span style="font-size: 14px;">**$\alpha_t = 1 - \beta_t$:** The per-step signal retention factor.</span>
* <span style="font-size: 14px;">**$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:** The cumulative signal retention from step $0$ to step $t$.</span>

<span style="font-size: 14px;">The initial noise sample is drawn from a standard Gaussian:</span>

$$
\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

<span style="font-size: 14px;">For each timestep $t$ from $T$ down to $1$, the reverse step computes:</span>

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \, \epsilon_\theta(\mathbf{x}_t, t) \right) + \sigma_t \, \mathbf{z}
$$

<span style="font-size: 14px;">where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ for $t > 1$ and $\mathbf{z} = \mathbf{0}$ for $t = 1$.</span>

* <span style="font-size: 14px;">**$\frac{1}{\sqrt{\alpha_t}}$:** Rescales the signal back up to compensate for the contraction applied during the forward process at step $t$.</span>
* <span style="font-size: 14px;">**$\frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}$:** The coefficient that converts the noise prediction $\epsilon_\theta$ into the correct noise magnitude to subtract from $\mathbf{x}_t$.</span>
* <span style="font-size: 14px;">**$\epsilon_\theta(\mathbf{x}_t, t)$:** The neural network's estimate of the noise component in $\mathbf{x}_t$. It takes both the noisy input and the timestep as arguments.</span>
* <span style="font-size: 14px;">**$\sigma_t$:** The noise scale for the reverse step. Ho et al. set $\sigma_t^2 = \beta_t$, matching the forward process variance. An alternative is $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$, the posterior variance.</span>

<span style="font-size: 14px;">The term inside the parentheses computes the estimated mean $\mu_\theta(\mathbf{x}_t, t)$ of the reverse distribution $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$. Adding $\sigma_t \mathbf{z}$ samples from that distribution rather than taking the point estimate.</span>

---

## <span style="font-size: 16px;">The Sampling Algorithm</span>

<span style="font-size: 14px;">The full procedure, following Algorithm 2 from Ho et al. (2020):</span>

<span style="font-size: 14px;">**Step 1.** Sample $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. This is a random tensor with the same shape as the desired output (e.g., $3 \times 64 \times 64$ for a $64 \times 64$ RGB image).</span>

<span style="font-size: 14px;">**Step 2.** For $t = T, T-1, \dots, 1$:</span>

* <span style="font-size: 14px;">If $t > 1$, sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. If $t = 1$, set $\mathbf{z} = \mathbf{0}$.</span>
* <span style="font-size: 14px;">Run the neural network forward pass: $\hat{\epsilon} = \epsilon_\theta(\mathbf{x}_t, t)$.</span>
* <span style="font-size: 14px;">Compute the denoised estimate: $\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon} \right) + \sigma_t \mathbf{z}$.</span>

<span style="font-size: 14px;">**Step 3.** Return $\mathbf{x}_0$ as the generated sample.</span>

<span style="font-size: 14px;">Each iteration of the loop requires exactly one forward pass through the denoising network. The schedule values $\alpha_t$, $\bar{\alpha}_t$, and $\sigma_t$ are precomputed once and stored as lookup arrays indexed by $t$. No gradient computation is needed during sampling, so inference mode is used.</span>

---

## <span style="font-size: 16px;">Why T Steps is Slow</span>

<span style="font-size: 14px;">The default DDPM uses $T = 1000$ timesteps, meaning 1000 sequential neural network forward passes per sample. On a modern GPU, each U-Net pass for a $256 \times 256$ image takes 20-50 ms, putting total generation time at 20-50 seconds per image.</span>

<span style="font-size: 14px;">The steps are inherently sequential: $\mathbf{x}_{t-1}$ depends on $\mathbf{x}_t$, which depends on $\mathbf{x}_{t+1}$. You cannot parallelize across timesteps for a single sample. Batch parallelism across multiple samples is possible, but per-sample latency remains $T$ forward passes.</span>

<span style="font-size: 14px;">For comparison, a GAN generates a sample in a single forward pass (~5-20 ms). This 1000x slowdown is DDPM's primary practical limitation and the main focus of subsequent research.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Ho, Jain, and Abbeel (2020) presented DDPM's sampling procedure in Algorithm 2 of "Denoising Diffusion Probabilistic Models." They described it as "sampling from the model is performed by running the reverse Markov chain, starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and iteratively sampling $\mathbf{x}_{t-1} \sim p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$."</span>

<span style="font-size: 14px;">The paper achieved an FID of 3.17 and Inception Score of 9.46 on CIFAR-10, competitive with GANs at the time. On LSUN bedrooms ($256 \times 256$), DDPM produced sharp, coherent room structures.</span>

<span style="font-size: 14px;">Compared to GANs, DDPM offered several advantages despite slower sampling. Training is stable with no adversarial min-max game, no mode collapse, and a simple MSE loss. DDPM also provides better mode coverage, generating more diverse samples rather than concentrating on a few high-quality modes.</span>

<span style="font-size: 14px;">Ho et al. chose $\sigma_t^2 = \beta_t$ for the reverse process variance and a linear schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ with $T = 1000$. Subsequent work (Nichol and Dhariwal, 2021) found that a cosine schedule produces better results, especially for higher-resolution images.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a toy example with $T = 3$ and a single scalar value (1D) to trace through the full sampling loop. We use the following simplified schedule:</span>

* <span style="font-size: 14px;">$\beta_1 = 0.1, \quad \beta_2 = 0.2, \quad \beta_3 = 0.3$</span>
* <span style="font-size: 14px;">$\alpha_1 = 0.9, \quad \alpha_2 = 0.8, \quad \alpha_3 = 0.7$</span>
* <span style="font-size: 14px;">$\bar{\alpha}_1 = 0.9, \quad \bar{\alpha}_2 = 0.72, \quad \bar{\alpha}_3 = 0.504$</span>
* <span style="font-size: 14px;">$\sigma_t = \sqrt{\beta_t}$, so $\sigma_1 \approx 0.3162, \quad \sigma_2 \approx 0.4472, \quad \sigma_3 \approx 0.5477$</span>

### <span style="font-size: 14px;">Step 1: Initialize $\mathbf{x}_3$</span>

<span style="font-size: 14px;">Sample $\mathbf{x}_3 \sim \mathcal{N}(0, 1)$. Suppose we draw $\mathbf{x}_3 = 1.5$.</span>

### <span style="font-size: 14px;">Step 2: Reverse from $t = 3$ to $t = 2$</span>

<span style="font-size: 14px;">The model predicts $\epsilon_\theta(\mathbf{x}_3, 3) = 0.8$ (its estimate of the noise in $\mathbf{x}_3$). Sample $\mathbf{z} \sim \mathcal{N}(0, 1)$; suppose $\mathbf{z} = -0.3$.</span>

$$
\mathbf{x}_2 = \frac{1}{\sqrt{0.7}} \left( 1.5 - \frac{1 - 0.7}{\sqrt{1 - 0.504}} \cdot 0.8 \right) + 0.5477 \cdot (-0.3)
$$

$$
= \frac{1}{0.8367} \left( 1.5 - \frac{0.3}{0.7043} \cdot 0.8 \right) - 0.1643
$$

$$
= 1.1952 \times (1.5 - 0.3408) - 0.1643
$$

$$
= 1.1952 \times 1.1592 - 0.1643 = 1.3855 - 0.1643 = 1.2212
$$

### <span style="font-size: 14px;">Step 3: Reverse from $t = 2$ to $t = 1$</span>

<span style="font-size: 14px;">The model predicts $\epsilon_\theta(\mathbf{x}_2, 2) = 0.5$. Sample $\mathbf{z} \sim \mathcal{N}(0, 1)$; suppose $\mathbf{z} = 0.1$.</span>

$$
\mathbf{x}_1 = \frac{1}{\sqrt{0.8}} \left( 1.2212 - \frac{1 - 0.8}{\sqrt{1 - 0.72}} \cdot 0.5 \right) + 0.4472 \cdot 0.1
$$

$$
= \frac{1}{0.8944} \left( 1.2212 - \frac{0.2}{0.5292} \cdot 0.5 \right) + 0.0447
$$

$$
= 1.1180 \times (1.2212 - 0.1890) + 0.0447
$$

$$
= 1.1180 \times 1.0322 + 0.0447 = 1.1540 + 0.0447 = 1.1987
$$

### <span style="font-size: 14px;">Step 4: Reverse from $t = 1$ to $t = 0$</span>

<span style="font-size: 14px;">The model predicts $\epsilon_\theta(\mathbf{x}_1, 1) = 0.2$. At $t = 1$, we set $\mathbf{z} = \mathbf{0}$ (no noise added).</span>

$$
\mathbf{x}_0 = \frac{1}{\sqrt{0.9}} \left( 1.1987 - \frac{1 - 0.9}{\sqrt{1 - 0.9}} \cdot 0.2 \right) + 0
$$

$$
= \frac{1}{0.9487} \left( 1.1987 - \frac{0.1}{0.3162} \cdot 0.2 \right)
$$

$$
= 1.0541 \times (1.1987 - 0.0632)
$$

$$
= 1.0541 \times 1.1355 = 1.1968
$$

<span style="font-size: 14px;">The final generated sample is $\mathbf{x}_0 \approx 1.197$. Starting from pure noise ($\mathbf{x}_3 = 1.5$), the model progressively refined it through 3 denoising steps. With a well-trained model on real data, $\mathbf{x}_0$ would land in a high-density region of the training distribution.</span>

---

## <span style="font-size: 16px;">The Role of Stochasticity</span>

<span style="font-size: 14px;">The noise term $\sigma_t \mathbf{z}$ added at each reverse step is not an implementation detail but a fundamental part of the generative process. It serves as a source of diversity: different random draws of $\mathbf{z}$ at each step lead to different final samples, even from the same initial $\mathbf{x}_T$. Without this noise, the same starting point would always produce the same output.</span>

<span style="font-size: 14px;">The magnitude $\sigma_t$ controls how much exploration happens at each step. Ho et al. used $\sigma_t^2 = \beta_t$, which matches the forward process variance and corresponds to the upper bound of the reverse process entropy. The alternative $\sigma_t^2 = \tilde{\beta}_t$ (the posterior variance) gives a lower bound. Both produce valid samples, but the choice affects sample diversity and quality.</span>

<span style="font-size: 14px;">DDIM (Song, Meng, and Ermon, 2020) showed that the noise term can be removed entirely, yielding a deterministic sampling process. The mapping from $\mathbf{x}_T$ to $\mathbf{x}_0$ becomes a fixed bijection, enabling meaningful interpolation in the latent space.</span>

<span style="font-size: 14px;">Temperature scaling provides a continuous control lever. Multiplying $\sigma_t$ by a factor $\eta$ (where $\eta = 1$ recovers DDPM and $\eta = 0$ gives DDIM) interpolates between stochastic and deterministic sampling. Lower $\eta$ produces sharper but less diverse samples.</span>

---

## <span style="font-size: 16px;">Modern Speedups</span>

<span style="font-size: 14px;">The 1000-step sampling requirement of DDPM motivated extensive research into faster alternatives. These methods reduce the number of neural network evaluations while maintaining sample quality.</span>

<span style="font-size: 14px;">**DDIM (Song et al., 2020)** reinterprets the diffusion process as a non-Markovian chain, enabling sampling with a subset of timesteps. Instead of all 1000 steps, DDIM uses a subsequence like $\{1, 51, 101, \dots, 951\}$ (20 steps) with minimal quality loss. The noise prediction network generalizes across timesteps, so skipping intermediate steps works.</span>

<span style="font-size: 14px;">**DPM-Solver (Lu et al., 2022)** treats the reverse diffusion as an ODE and applies high-order numerical solvers. While DDPM uses first-order Euler-like updates, DPM-Solver uses second and third-order methods that take larger, more accurate steps, achieving strong results in 10-20 steps.</span>

<span style="font-size: 14px;">**Consistency Models (Song et al., 2023)** learn to map any point on the diffusion trajectory directly to $\mathbf{x}_0$ in a single step, trained either by distilling a pre-trained diffusion model or from scratch.</span>

<span style="font-size: 14px;">**Progressive Distillation (Salimans and Ho, 2022)** trains a student to combine two teacher steps into one, then repeats. Each round halves the step count: 1024, 512, 256, ..., down to 4 steps.</span>

<span style="font-size: 14px;">**Latent Diffusion (Rombach et al., 2022)** runs diffusion in a compressed latent space rather than pixel space. An encoder compresses images (e.g., $512 \times 512$ to $64 \times 64$ latents), diffusion operates there, and a decoder maps back to pixels. Stable Diffusion is the most prominent example.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Wrong Loop Direction</span>

<span style="font-size: 14px;">The reverse process must iterate from $t = T$ down to $t = 1$. A common implementation error is looping from $t = 1$ to $t = T$, which runs the forward (noising) direction instead of the reverse (denoising) direction. The result is progressively noisier outputs rather than cleaner ones.</span>

### <span style="font-size: 14px;">Forgetting $\mathbf{z} = \mathbf{0}$ at $t = 1$</span>

<span style="font-size: 14px;">At the final step ($t = 1$), no noise should be added. If $\mathbf{z}$ is sampled normally at $t = 1$, the final output $\mathbf{x}_0$ will have unnecessary Gaussian noise overlaid on it, producing a visibly grainy image. This is a one-line bug with significant visual impact.</span>

### <span style="font-size: 14px;">Using Wrong Schedule Values</span>

<span style="font-size: 14px;">Confusing $\alpha_t$ with $\bar{\alpha}_t$ is a frequent source of bugs. The reverse step formula uses both: $\alpha_t$ (the per-step value) appears as $\frac{1}{\sqrt{\alpha_t}}$ and in the numerator $1 - \alpha_t$, while $\bar{\alpha}_t$ (the cumulative product) appears in the denominator $\sqrt{1 - \bar{\alpha}_t}$. Swapping them produces incorrect denoising magnitudes and corrupted samples.</span>

### <span style="font-size: 14px;">Not Starting from Pure Gaussian Noise</span>

<span style="font-size: 14px;">The derivation assumes $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. If $\mathbf{x}_T$ is initialized from a different distribution (e.g., uniform noise, or Gaussian with wrong variance), the reverse process produces samples outside the learned distribution. The variance must be exactly $\mathbf{I}$, not scaled.</span>

### <span style="font-size: 14px;">Clipping Output Values</span>

<span style="font-size: 14px;">Intermediate $\mathbf{x}_t$ values can drift outside the expected range (e.g., beyond $[-1, 1]$). Aggressive clipping at every step introduces artifacts. The recommended practice is to clip only the final $\mathbf{x}_0$, or use dynamic thresholding (Saharia et al., 2022) which rescales outlier values rather than hard clipping.</span>

---