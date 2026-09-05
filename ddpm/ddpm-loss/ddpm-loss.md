# <span style="font-size: 20px;">DDPM Training Loss</span>

<span style="font-size: 14px;">Denoising Diffusion Probabilistic Models (DDPMs) learn to generate data by reversing a gradual noising process. The training objective proposed by Ho et al. (2020) is remarkably simple: predict the noise that was added to a data sample. The loss is a plain mean squared error between the true noise and the model's prediction. This simplified objective replaced a complex variational lower bound with a single, elegant formula that produces superior sample quality.</span>

---

## <span style="font-size: 16px;">What It Is</span>

<span style="font-size: 14px;">A diffusion model defines two processes. The forward process gradually adds Gaussian noise to data over $T$ timesteps until the signal is destroyed. The reverse process learns to undo each noising step, recovering clean data from pure noise. Training the reverse process requires a loss function that teaches the neural network $\epsilon_\theta$ what noise was added at each step.</span>

<span style="font-size: 14px;">The DDPM training loss, called $L_{\text{simple}}$, asks the model to predict the exact noise vector $\epsilon$ that was sampled and added to create the noisy input $x_t$. The model sees $x_t$ and the timestep $t$, and outputs its best guess $\epsilon_\theta(x_t, t)$. The loss is the mean squared error between the true noise and the prediction.</span>

<span style="font-size: 14px;">This is a departure from earlier diffusion models (Sohl-Dickstein et al., 2015) that optimized the full variational lower bound. Ho et al. showed that this simple noise-prediction MSE loss produces better images, even though it is theoretically "less correct" from a variational inference perspective.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

### <span style="font-size: 14px;">The Forward Process</span>

<span style="font-size: 14px;">Given a noise schedule $\beta_1, \beta_2, \ldots, \beta_T$ with $\beta_t \in (0, 1)$, the forward process adds noise incrementally:</span>

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I)
$$

<span style="font-size: 14px;">A key property is that we can sample $x_t$ directly from $x_0$ without iterating through all previous steps. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:</span>

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I)
$$

<span style="font-size: 14px;">This means we can write $x_t$ directly as:</span>

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

### <span style="font-size: 14px;">The Variational Lower Bound (VLB)</span>

<span style="font-size: 14px;">The full variational lower bound decomposes the negative log-likelihood into a sum of KL divergence terms:</span>

$$
L_{\text{vlb}} = L_0 + L_1 + \cdots + L_{T-1} + L_T
$$

<span style="font-size: 14px;">where each $L_{t-1}$ for $t > 1$ is:</span>

$$
L_{t-1} = D_{\text{KL}}(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))
$$

<span style="font-size: 14px;">Both $q(x_{t-1} | x_t, x_0)$ and $p_\theta(x_{t-1} | x_t)$ are Gaussian, so this KL divergence has a closed-form solution. But optimizing all $T$ terms with their individual weightings is complex and unstable.</span>

### <span style="font-size: 14px;">The Simplified Objective</span>

<span style="font-size: 14px;">Ho et al. showed that each KL term $L_{t-1}$ can be rewritten in terms of noise prediction. After reparameterizing the mean of $p_\theta$ to predict $\epsilon$ rather than $x_{t-1}$ directly, and dropping the time-dependent weighting factor, the simplified loss becomes:</span>

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

<span style="font-size: 14px;">where $t \sim \text{Uniform}(\{1, \ldots, T\})$, $x_0 \sim q(x_0)$ is a training sample, $\epsilon \sim \mathcal{N}(0, I)$, and $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$.</span>

---

## <span style="font-size: 16px;">From VLB to Simple Loss</span>

<span style="font-size: 14px;">The full VLB assigns a different weight to each timestep's loss term. These weights come from the coefficients in the KL divergence between the true posterior and the learned reverse distribution. Specifically, each term $L_{t-1}$ in the VLB carries a coefficient proportional to $\frac{\beta_t^2}{2 \sigma_t^2 \alpha_t (1 - \bar{\alpha}_t)}$, which varies with $t$.</span>

<span style="font-size: 14px;">Ho et al. found that dropping these weights and treating all timesteps equally (uniform weighting) actually produces better sample quality. The intuition is that the VLB weighting down-weights loss terms at large $t$ (high noise levels), but these are precisely the timesteps where the model needs to learn coarse global structure. By giving equal weight to all timesteps, the simple loss forces the model to perform well at every noise level.</span>

<span style="font-size: 14px;">There is a tradeoff: $L_{\text{vlb}}$ produces better log-likelihood scores (it directly optimizes the variational bound), while $L_{\text{simple}}$ produces better sample quality (measured by FID and IS). Ho et al. chose sample quality, arguing that perceptual quality matters more than log-likelihood for image generation.</span>

---

## <span style="font-size: 16px;">The Training Algorithm</span>

<span style="font-size: 14px;">The complete training procedure, as described in Algorithm 1 of the DDPM paper, repeats the following steps until convergence:</span>

<span style="font-size: 14px;">**Step 1.** Sample a training example $x_0$ from the data distribution $q(x_0)$.</span>

<span style="font-size: 14px;">**Step 2.** Sample a timestep $t$ uniformly from $\{1, 2, \ldots, T\}$.</span>

<span style="font-size: 14px;">**Step 3.** Sample a noise vector $\epsilon \sim \mathcal{N}(0, I)$ with the same shape as $x_0$.</span>

<span style="font-size: 14px;">**Step 4.** Construct the noisy sample using the forward process formula:</span>

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
$$

<span style="font-size: 14px;">**Step 5.** Pass $x_t$ and $t$ through the neural network to get the noise prediction $\epsilon_\theta(x_t, t)$.</span>

<span style="font-size: 14px;">**Step 6.** Compute the loss as the mean squared error:</span>

$$
L = \| \epsilon - \epsilon_\theta(x_t, t) \|^2
$$

<span style="font-size: 14px;">**Step 7.** Compute gradients $\nabla_\theta L$ and update the model parameters $\theta$ with the optimizer.</span>

<span style="font-size: 14px;">Each training step samples a fresh $t$ and a fresh $\epsilon$. Different elements in a batch should each get their own independently sampled $t$. The stochasticity over $t$ and $\epsilon$ provides the Monte Carlo estimate of the full expectation in $L_{\text{simple}}$.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Ho et al. (2020) tested both $L_{\text{vlb}}$ and $L_{\text{simple}}$ on CIFAR-10 and 256x256 LSUN datasets. With $T = 1000$ timesteps and a linear noise schedule from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$, they found that $L_{\text{simple}}$ achieved an FID of 3.17 on CIFAR-10, which was state-of-the-art at the time.</span>

<span style="font-size: 14px;">The paper reports that training with $L_{\text{vlb}}$ gave better negative log-likelihood (3.99 bits/dim vs 3.75 bits/dim), but the FID scores were worse. This demonstrated a fundamental tension in generative modeling: optimizing the exact variational bound does not necessarily produce the best perceptual quality.</span>

<span style="font-size: 14px;">The architecture used was a U-Net with self-attention at the 16x16 resolution. The timestep $t$ was encoded via sinusoidal position embeddings (borrowed from Transformers) and injected into each residual block. The model predicted $\epsilon$ at every timestep, and the reverse process used a fixed variance $\sigma_t^2 = \beta_t$.</span>

<span style="font-size: 14px;">A later follow-up, Improved DDPM (Nichol and Dhariwal, 2021), showed that learning the variance alongside the mean and using a hybrid loss $L_{\text{hybrid}} = L_{\text{simple}} + \lambda L_{\text{vlb}}$ could improve both sample quality and log-likelihood simultaneously.</span>

---

## <span style="font-size: 16px;">What the Model Learns</span>

<span style="font-size: 14px;">The neural network $\epsilon_\theta(x_t, t)$ learns to predict the noise component in the noisy input $x_t$. Since $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$, predicting $\epsilon$ is equivalent to predicting $x_0$ via the relationship:</span>

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

<span style="font-size: 14px;">There is also a deep connection to score matching. The score function of a distribution is $\nabla_{x} \log p(x)$. For the noisy distribution $q(x_t)$, the score is proportional to the negative noise:</span>

$$
\nabla_{x_t} \log q(x_t) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}
$$

<span style="font-size: 14px;">So training $\epsilon_\theta$ to predict $\epsilon$ is equivalent to training a score network $s_\theta(x_t, t) \approx \nabla_{x_t} \log q(x_t)$. This connects DDPMs to the score-based generative modeling framework of Song and Ermon (2019).</span>

<span style="font-size: 14px;">The three parameterizations (noise prediction, $x_0$ prediction, and score prediction) are mathematically equivalent. They differ only by scaling factors that depend on $\bar{\alpha}_t$. The choice of parameterization affects training dynamics: noise prediction works best for the original DDPM, while $x_0$ prediction and velocity prediction have advantages in other settings.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a simplified 1D example to trace through the training loss computation. Let $T = 1000$ with a linear schedule $\beta_t$ from $0.0001$ to $0.02$.</span>

<span style="font-size: 14px;">**Data point:** $x_0 = 0.8$</span>

<span style="font-size: 14px;">**Sampled timestep:** $t = 300$</span>

<span style="font-size: 14px;">**Computing $\bar{\alpha}_{300}$:** With the linear schedule, $\beta_{300} \approx 0.006$. The cumulative product $\bar{\alpha}_{300} = \prod_{s=1}^{300}(1 - \beta_s) \approx 0.448$. Therefore $\sqrt{\bar{\alpha}_{300}} \approx 0.669$ and $\sqrt{1 - \bar{\alpha}_{300}} \approx 0.743$.</span>

<span style="font-size: 14px;">**Sampled noise:** $\epsilon = -1.2$ (drawn from $\mathcal{N}(0, 1)$)</span>

<span style="font-size: 14px;">**Constructing $x_t$:**</span>

$$
x_{300} = 0.669 \times 0.8 + 0.743 \times (-1.2) = 0.535 - 0.892 = -0.357
$$

<span style="font-size: 14px;">**Model prediction:** The network sees $x_{300} = -0.357$ and $t = 300$. Suppose it outputs $\epsilon_\theta(x_{300}, 300) = -1.05$.</span>

<span style="font-size: 14px;">**Computing the loss:**</span>

$$
L = \| \epsilon - \epsilon_\theta \|^2 = (-1.2 - (-1.05))^2 = (-0.15)^2 = 0.0225
$$

<span style="font-size: 14px;">The model's prediction was close but not perfect. Over many training steps with different $x_0$, $t$, and $\epsilon$ samples, the model learns to minimize this MSE across all combinations. At convergence, the expected loss approaches the irreducible error, which depends on how well the model architecture can approximate the true noise.</span>

---

## <span style="font-size: 16px;">Modern Variants</span>

### <span style="font-size: 14px;">Velocity Prediction (v-prediction)</span>

<span style="font-size: 14px;">Salimans and Ho (2022) proposed predicting the "velocity" $v_t = \sqrt{\bar{\alpha}_t} \, \epsilon - \sqrt{1 - \bar{\alpha}_t} \, x_0$ instead of predicting $\epsilon$ directly. The loss becomes $L_v = \mathbb{E}[\| v_t - v_\theta(x_t, t) \|^2]$. This parameterization provides more stable gradients at low noise levels (small $t$) and is used in Stable Diffusion v2 and later models.</span>

### <span style="font-size: 14px;">$x_0$ Prediction</span>

<span style="font-size: 14px;">Instead of predicting noise, the model directly predicts the clean data $x_0$. The loss is $L_{x_0} = \mathbb{E}[\| x_0 - x_{0,\theta}(x_t, t) \|^2]$. This is mathematically equivalent to noise prediction (up to a $t$-dependent scaling), but changes the implicit weighting across timesteps. Some works like DALL-E 2 use $x_0$ prediction because it allows directly inspecting what the model "thinks" the clean image looks like.</span>

### <span style="font-size: 14px;">Weighted Losses</span>

<span style="font-size: 14px;">Several works have explored non-uniform weighting schemes. The P2 weighting (Choi et al., 2022) assigns higher weight to perceptually important timesteps. Min-SNR weighting (Hang et al., 2023) clips the signal-to-noise ratio to prevent any single timestep from dominating the gradient. These schemes try to recover the benefits of the VLB weighting while maintaining the stability of $L_{\text{simple}}$.</span>

### <span style="font-size: 14px;">Classifier-Free Guidance Loss</span>

<span style="font-size: 14px;">For conditional generation, classifier-free guidance (Ho and Salimans, 2022) trains the model jointly on conditional and unconditional denoising. During training, the conditioning signal (e.g., text prompt or class label) is randomly dropped with some probability (typically 10-20%), replaced by a null token. The loss remains the same MSE on noise prediction, but the model learns both $\epsilon_\theta(x_t, t, c)$ and $\epsilon_\theta(x_t, t, \varnothing)$. At inference time, the two predictions are combined: $\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))$, where $w > 1$ controls guidance strength.</span>

---

## <span style="font-size: 16px;">Common Pitfalls</span>

<span style="font-size: 14px;">**Not sampling $t$ uniformly.** Every timestep from 1 to $T$ should have equal probability of being selected. Biasing toward small or large $t$ changes the implicit loss weighting and degrades sample quality. Some practitioners accidentally use 0-indexed sampling ($t \in \{0, \ldots, T-1\}$), which shifts all the noise schedule values by one position.</span>

<span style="font-size: 14px;">**Using the same $t$ for all batch elements.** Each sample in the batch should get its own independently drawn $t$. Sharing a single $t$ across the batch reduces the diversity of the gradient signal and slows convergence. The variance reduction from seeing multiple timesteps per batch is essential for stable training.</span>

<span style="font-size: 14px;">**Forgetting to detach $x_t$ from the gradient graph.** When constructing $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$, this is a data preparation step, not part of the model's forward pass. If the computation graph tracks gradients through $x_t$ back to $x_0$, the optimizer may try to "change the data" instead of improving the model. In PyTorch, $x_0$ should be treated as a constant input, and the noise schedule values $\bar{\alpha}_t$ should not require gradients.</span>

<span style="font-size: 14px;">**Computing loss on $x_t$ instead of $\epsilon$.** A common implementation mistake is to compare the model output against the noisy input $x_t$ rather than the noise $\epsilon$. The model predicts noise, not noisy data. The target is always the sampled $\epsilon$, never $x_t$ or $x_0$ (unless using the $x_0$-prediction variant explicitly).</span>

<span style="font-size: 14px;">**Wrong MSE reduction.** PyTorch's `nn.MSELoss` defaults to `reduction='mean'`, which averages over all elements (spatial dimensions and channels). Using `reduction='sum'` instead changes the effective learning rate by a factor equal to the data dimensionality. The DDPM paper uses per-element MSE averaged over the batch, matching `reduction='mean'`. Switching reductions without adjusting the learning rate causes training to diverge or converge to poor solutions.</span>