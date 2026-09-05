# <span style="font-size: 20px;">Noise Schedule</span>

<span style="font-size: 14px;">The noise schedule is the sequence of variance values $\beta_1, \beta_2, \dots, \beta_T$ that controls how much Gaussian noise is injected at each timestep of the forward diffusion process. In Denoising Diffusion Probabilistic Models (Ho, Gulrajani, and Abbeel, 2020), this schedule is the backbone of the generative framework. Every downstream quantity -- noisy training samples, signal-to-noise ratio, and network prediction targets -- derives from this sequence of scalars.</span>

<span style="font-size: 14px;">A poorly chosen schedule either wastes steps where the data is already destroyed, or fails to fully corrupt the input into pure noise by the final timestep.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The noise schedule defines the forward process $q(x_t | x_{t-1})$ at every timestep. At each step $t$, Gaussian noise scaled by $\beta_t$ is added and the signal is scaled by $\sqrt{1 - \beta_t}$. Over $T$ steps, this transforms any data distribution into an isotropic Gaussian.</span>

* <span style="font-size: 14px;">**Forward corruption rate:** Larger $\beta_t$ means more noise per step, faster signal destruction.</span>
* <span style="font-size: 14px;">**Reverse denoising difficulty:** Larger $\beta_t$ means each reverse step must remove more noise, making prediction harder.</span>

<span style="font-size: 14px;">The schedule is a fixed hyperparameter, not learned. From $\beta_t$, two derived quantities govern all computations:</span>

* <span style="font-size: 14px;">**$\alpha_t = 1 - \beta_t$:** The signal retention factor at step $t$.</span>
* <span style="font-size: 14px;">**$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$:** The cumulative signal retention from step 0 to step $t$. This lets you jump directly from clean data $x_0$ to noisy data $x_t$ without iterating through intermediate steps.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">The forward process at each step:</span>

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t \mathbf{I})
$$

<span style="font-size: 14px;">The marginal $q(x_t | x_0)$ has a closed form via $\bar{\alpha}_t$:</span>

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t) \mathbf{I})
$$

<span style="font-size: 14px;">Any noisy sample at step $t$ can be written as:</span>

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

<span style="font-size: 14px;">The derived quantities:</span>

* <span style="font-size: 14px;">**Signal retention per step:** $\alpha_t = 1 - \beta_t$</span>
* <span style="font-size: 14px;">**Cumulative signal retention:** $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \alpha_1 \cdot \alpha_2 \cdots \alpha_t$</span>
* <span style="font-size: 14px;">**Noise coefficient:** $\sqrt{1 - \bar{\alpha}_t}$ scales the noise at step $t$.</span>
* <span style="font-size: 14px;">**Signal coefficient:** $\sqrt{\bar{\alpha}_t}$ scales the clean data at step $t$.</span>

<span style="font-size: 14px;">The linear schedule defines $\beta_t$ by interpolation:</span>

$$
\beta_t = \beta_1 + \frac{t - 1}{T - 1}(\beta_T - \beta_1)
$$

<span style="font-size: 14px;">where $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, and $T = 1000$.</span>

<span style="font-size: 14px;">The cosine schedule from Nichol and Dhariwal (2021) defines $\bar{\alpha}_t$ directly:</span>

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

<span style="font-size: 14px;">where $s = 0.008$ is a small offset preventing $\beta_t$ from being too small near $t = 0$.</span>

---

## <span style="font-size: 16px;">Linear Schedule</span>

<span style="font-size: 14px;">Ho et al. chose the simplest possible schedule: linear interpolation of $\beta_t$ from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$ over $T = 1000$ steps.</span>

* <span style="font-size: 14px;">**$\beta_1 = 10^{-4}$ (very small start):** At $t = 1$, almost no noise is added. The first forward step barely perturbs the data.</span>
* <span style="font-size: 14px;">**$\beta_T = 0.02$ (moderate end):** After 1000 cumulative steps, $\bar{\alpha}_T$ is driven close to zero -- $x_T$ is nearly pure Gaussian noise.</span>
* <span style="font-size: 14px;">**$T = 1000$ steps:** Each reverse step is a small, learnable denoising task. Fewer steps would require larger $\beta_t$ jumps.</span>

<span style="font-size: 14px;">With these values, $\bar{\alpha}_t$ starts near 1 and decays monotonically. Early on, each $\alpha_t \approx 0.9999$ so the product barely budges. In the middle, $\bar{\alpha}_t$ drops steadily. Near the end, it approaches zero rapidly.</span>

<span style="font-size: 14px;">The linear schedule has a known asymmetry: $\bar{\alpha}_t$ spends many early timesteps near 1 and late timesteps near 0. The useful transition zone is concentrated in the middle third. The model wastes capacity on near-trivial early denoising and near-impossible late denoising.</span>

---

## <span style="font-size: 16px;">Cosine Schedule</span>

<span style="font-size: 14px;">Nichol and Dhariwal (2021) identified the linear schedule's inefficiency: too many timesteps are spent where $\bar{\alpha}_t$ is near 1 or near 0. Their solution: define $\bar{\alpha}_t$ directly via a cosine function, then extract $\beta_t$.</span>

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

<span style="font-size: 14px;">Key properties:</span>

* <span style="font-size: 14px;">**At $t = 0$:** $f(0)/f(0) = 1$, so $\bar{\alpha}_0 = 1$ (pure signal).</span>
* <span style="font-size: 14px;">**At $t = T$:** The cosine argument approaches $\pi/2$, so $\bar{\alpha}_T \to 0$ (pure noise).</span>
* <span style="font-size: 14px;">**Offset $s = 0.008$:** Prevents $\beta_t$ from being too small near $t = 0$, ensuring non-trivial noise from the first step.</span>
* <span style="font-size: 14px;">**Cosine shape:** The squared cosine produces an S-shaped $\bar{\alpha}_t$ curve that transitions smoothly from 1 to 0.</span>

<span style="font-size: 14px;">Once $\bar{\alpha}_t$ is defined, $\beta_t$ is recovered as:</span>

$$
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}
$$

<span style="font-size: 14px;">Nichol and Dhariwal clip $\beta_t$ to a maximum of $0.999$ to prevent numerical instability near the end of the schedule. The practical benefit: the SNR decreases more gradually, more timesteps are spent in the informative middle range, and sample quality improves.</span>

---

## <span style="font-size: 16px;">What $\bar{\alpha}_t$ Represents</span>

<span style="font-size: 14px;">$\bar{\alpha}_t$ captures total signal retention from $x_0$ to $x_t$. It is the most important quantity in the diffusion framework.</span>

* <span style="font-size: 14px;">**$\bar{\alpha}_t$ near 1:** The sample is mostly clean data. The noise component $\sqrt{1 - \bar{\alpha}_t}\, \epsilon$ is small.</span>
* <span style="font-size: 14px;">**$\bar{\alpha}_t$ near 0.5:** Equal parts signal and noise variance. The model must extract structure from a heavily corrupted input.</span>
* <span style="font-size: 14px;">**$\bar{\alpha}_t$ near 0:** Almost pure noise. The original data is nearly unrecoverable.</span>

<span style="font-size: 14px;">The formal signal-to-noise ratio (SNR) at timestep $t$:</span>

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

<span style="font-size: 14px;">The DDPM loss can be rewritten in terms of SNR, and the timestep weighting implicitly controls focus on different noise levels.</span>

* <span style="font-size: 14px;">**High SNR (early timesteps):** The model learns fine details and high-frequency features.</span>
* <span style="font-size: 14px;">**Low SNR (late timesteps):** The model learns coarse, global structure.</span>
* <span style="font-size: 14px;">**Schedule determines the SNR curve:** Switching from linear to cosine shifts which SNR regions get more timesteps.</span>

<span style="font-size: 14px;">The coefficients satisfy $\bar{\alpha}_t + (1 - \bar{\alpha}_t) = 1$, so total variance is preserved at every timestep. This variance-preserving property makes the forward process well-behaved.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Ho et al. (2020) made deliberately simple choices to show diffusion models could compete with GANs.</span>

* <span style="font-size: 14px;">**$T = 1000$:** Ensures each reverse step is a small perturbation. The paper: "We set $T = 1000$ for all experiments."</span>
* <span style="font-size: 14px;">**Linear $\beta_t$ from $10^{-4}$ to $0.02$:** The paper: "We set the forward process variances to constants increasing linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$."</span>
* <span style="font-size: 14px;">**Fixed schedule (not learned):** They explored learning $\Sigma_\theta$ but found fixing it to $\beta_t \mathbf{I}$ or $\tilde{\beta}_t \mathbf{I}$ worked well.</span>

<span style="font-size: 14px;">The simplified training loss:</span>

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

<span style="font-size: 14px;">where $t \sim \text{Uniform}\{1, \dots, T\}$. The schedule enters through $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$. Uniform $t$ sampling with the linear schedule over-trains on easy and hard regions, under-trains on the informative middle. Nichol and Dhariwal later showed this is suboptimal.</span>

<span style="font-size: 14px;">Despite its simplicity, the linear schedule yielded an FID of 3.17 on CIFAR-10, establishing diffusion models as competitive.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Linear schedule with $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$. The step size:</span>

$$
\Delta\beta = \frac{\beta_T - \beta_1}{T - 1} = \frac{0.02 - 0.0001}{999} \approx 1.99 \times 10^{-5}
$$

<span style="font-size: 14px;">**Early timesteps (t = 1, 2, 3):**</span>

* <span style="font-size: 14px;">**$t = 1$:** $\beta_1 = 0.0001$, $\alpha_1 = 0.9999$, $\bar{\alpha}_1 = 0.9999$</span>
* <span style="font-size: 14px;">**$t = 2$:** $\beta_2 = 0.0001199$, $\alpha_2 = 0.9998801$, $\bar{\alpha}_2 = 0.9999 \times 0.9998801 = 0.9997801$</span>
* <span style="font-size: 14px;">**$t = 3$:** $\beta_3 = 0.0001399$, $\alpha_3 = 0.9998601$, $\bar{\alpha}_3 = 0.9997801 \times 0.9998601 = 0.9996403$</span>

<span style="font-size: 14px;">After 3 steps, $\bar{\alpha}_3 \approx 0.9996$. In the sampling equation, the noise coefficient is only $\sqrt{0.0004} = 0.02$. The noisy image is visually indistinguishable from the original.</span>

<span style="font-size: 14px;">**Late timesteps (t = 999, 1000):**</span>

* <span style="font-size: 14px;">**$t = 999$:** $\beta_{999} = 0.01990$, $\alpha_{999} = 0.98010$</span>
* <span style="font-size: 14px;">**$t = 1000$:** $\beta_{1000} = 0.02000$, $\alpha_{1000} = 0.98000$</span>

<span style="font-size: 14px;">The cumulative product at $t = 1000$ is computed via log-sum:</span>

$$
\log \bar{\alpha}_{1000} = \sum_{s=1}^{1000} \log(1 - \beta_s) \approx -10.0
$$

<span style="font-size: 14px;">This gives $\bar{\alpha}_{1000} \approx e^{-10.0} \approx 4.5 \times 10^{-5}$. The sampling equation becomes $x_{1000} \approx 0.0067\, x_0 + 0.9999\, \epsilon$, which is nearly pure noise.</span>

<span style="font-size: 14px;">**The decay profile across timesteps:**</span>

* <span style="font-size: 14px;">**$t = 100$:** $\bar{\alpha}_{100} \approx 0.990$ (1% noise variance)</span>
* <span style="font-size: 14px;">**$t = 250$:** $\bar{\alpha}_{250} \approx 0.940$ (6% noise variance)</span>
* <span style="font-size: 14px;">**$t = 500$:** $\bar{\alpha}_{500} \approx 0.725$ (27.5% noise variance)</span>
* <span style="font-size: 14px;">**$t = 750$:** $\bar{\alpha}_{750} \approx 0.300$ (70% noise variance)</span>
* <span style="font-size: 14px;">**$t = 900$:** $\bar{\alpha}_{900} \approx 0.045$ (95.5% noise variance)</span>
* <span style="font-size: 14px;">**$t = 1000$:** $\bar{\alpha}_{1000} \approx 0.00005$ (99.995% noise variance)</span>

<span style="font-size: 14px;">This shows the asymmetry: the first 250 steps only destroy 6% of the signal, while steps 750-1000 are spent where the signal is already 95%+ destroyed. The cosine schedule redistributes timesteps to spend more time in $\bar{\alpha}_t \in [0.1, 0.9]$.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">The linear schedule was a starting point. Subsequent work produced many alternatives.</span>

* <span style="font-size: 14px;">**Cosine schedule (Nichol and Dhariwal, 2021):** The most widely adopted improvement. Smoother SNR decay leads to better FID scores. Used in Improved DDPM and many subsequent models.</span>
* <span style="font-size: 14px;">**Sigmoid schedule:** Uses a sigmoid to define $\bar{\alpha}_t$, providing a tunable inflection point controlling where the fastest signal destruction occurs.</span>
* <span style="font-size: 14px;">**Learned schedules:** Some approaches learn $\beta_t$ jointly with model parameters, though this has not become standard.</span>
* <span style="font-size: 14px;">**Continuous-time diffusion (Song et al., 2021):** Replaces the discrete schedule with a continuous SDE, $\beta(t)$ for $t \in [0, 1]$, unifying DDPM and score-matching.</span>
* <span style="font-size: 14px;">**Flow matching (Lipman et al., 2023):** Defines straight-line paths $x_t = (1-t)\, x_0 + t\, \epsilon$ with no $\beta_t$ schedule at all.</span>
* <span style="font-size: 14px;">**Resolution-dependent schedules:** Chen (2023) showed optimal schedules depend on resolution. Higher-resolution images need slower noise addition.</span>
* <span style="font-size: 14px;">**Timestep spacing for fast sampling:** With fewer steps (e.g., DDIM with 50 steps), which timesteps to evaluate is itself a schedule design question.</span>

<span style="font-size: 14px;">The trend is toward either simpler schedules (flow matching removes them entirely) or adaptive ones (continuous-time solvers choose step sizes). Ho et al.'s linear schedule remains a baseline and pedagogical reference.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">$\bar{\alpha}_T$ Not Reaching Near Zero</span>

<span style="font-size: 14px;">If $\bar{\alpha}_T$ is not close to zero, $q(x_T | x_0)$ differs from $\mathcal{N}(0, \mathbf{I})$. The reverse process starts from this prior, so any mismatch causes systematic bias. With Ho et al.'s schedule, $\bar{\alpha}_{1000} \approx 4.5 \times 10^{-5}$, small enough to avoid this.</span>

### <span style="font-size: 14px;">Numerical Underflow in Cumulative Products</span>

<span style="font-size: 14px;">Computing $\prod_{s=1}^{t} (1 - \beta_s)$ as a running product in float32 can underflow to zero for large $t$, making the loss gradient degenerate. Fix: compute $\log \bar{\alpha}_t = \sum \log(1 - \beta_s)$ in log-space and exponentiate only when needed, or use float64.</span>

### <span style="font-size: 14px;">Off-by-One in Timestep Indexing</span>

<span style="font-size: 14px;">Some implementations index from 0 to $T-1$, others from 1 to $T$. The formula assumes 1-based indexing. Using 0-based without adjusting shifts the entire schedule. Always verify $\beta$ at the first index equals $\beta_1$ and at the last equals $\beta_T$.</span>

### <span style="font-size: 14px;">Confusing $\beta_t$ with $\alpha_t$</span>

<span style="font-size: 14px;">$\beta_t$ is noise variance; $\alpha_t = 1 - \beta_t$ is signal retention. Passing $\beta_t$ where $\alpha_t$ is expected inverts the noise level -- the model sees pure noise at $t = 1$ and clean data at $t = T$. Training loss explodes immediately.</span>

### <span style="font-size: 14px;">Wrong $T$ Value</span>

<span style="font-size: 14px;">If training uses $T = 1000$ but sampling uses $T = 500$ without recomputing the schedule, every $\bar{\alpha}_t$ is wrong. Fix: retrain with the new $T$ or use DDIM-style timestep skipping (subset of the original 1000 steps).</span>

### <span style="font-size: 14px;">$\beta_t$ Too Large</span>

<span style="font-size: 14px;">Large $\beta_t$ values (e.g., $> 0.1$) break the Gaussian approximation in the reverse process. The cosine schedule clips $\beta_t$ at 0.999 for this reason.</span>

---