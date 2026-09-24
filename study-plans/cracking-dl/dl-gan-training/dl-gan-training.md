# <span style="font-size: 20px;">GAN Training Loop</span>

<span style="font-size: 14px;">Generative Adversarial Networks (GANs), introduced by Goodfellow et al. (2014), train two networks in opposition: a generator that creates fake data and a discriminator that distinguishes real from fake. This adversarial game drives both networks to improve, with the generator eventually producing realistic samples. GANs have revolutionized image generation and remain fundamental to understanding adversarial training dynamics.</span>

---

## <span style="font-size: 16px;">Adversarial Framework</span>

<span style="font-size: 14px;">A GAN consists of two players:</span>

<span style="font-size: 14px;">**Generator** $G: \mathbb{R}^{d_z} \to \mathbb{R}^{d_x}$ maps noise vectors $z \sim p_z$ (typically standard normal) to fake data samples. The goal is to learn the data distribution $p_{\text{data}}$.</span>

<span style="font-size: 14px;">**Discriminator** $D: \mathbb{R}^{d_x} \to [0, 1]$ outputs the probability that its input is real (from the training set) rather than fake (from the generator).</span>

<span style="font-size: 14px;">The minimax objective is:</span>

$$
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

<span style="font-size: 14px;">At the Nash equilibrium, $G$ perfectly models the data distribution and $D$ outputs 0.5 everywhere (cannot distinguish real from fake). In practice, this equilibrium is rarely reached.</span>

---

## <span style="font-size: 16px;">Training Procedure</span>

<span style="font-size: 14px;">This exercise performs one alternating SGD step with supplied parameters and one supplied noise batch. The generator is affine followed by tanh; the discriminator is affine and returns logits. Derive the updates using NumPy float64. The architecture examples below describe broader GAN designs.</span>

<span style="font-size: 14px;">**Step 1 - Update Discriminator**: Use the supplied real_data and z, and generate fake data $\hat{x} = G(z)$. Sum the real and fake mean binary cross-entropy losses, without an extra factor of one half:</span>

$$
\mathcal{L}_D = \operatorname{mean}(\operatorname{softplus}(-a(X))) + \operatorname{mean}(\operatorname{softplus}(a(F)))
$$

<span style="font-size: 14px;">Here X is real_data, F is the generated batch, and a is the discriminator logit function before its update. Softplus of v is log(1 + exp(v)); each mean covers the batch. Treat F as constant while deriving discriminator gradients, and update only W_d and b_d. Report d_loss before this update. In an autodiff implementation, detaching F prevents discriminator-phase gradients from accumulating in the generator.</span>

<span style="font-size: 14px;">**Step 2 - Update Generator**: Reuse the supplied noise z and generated values F. Hold the updated discriminator fixed while deriving the generator gradient through its affine operation and tanh. Use the non-saturating generator loss:</span>

$$
\mathcal{L}_G = \operatorname{mean}(\operatorname{softplus}(-a'(F)))
$$

<span style="font-size: 14px;">The primed logit function uses the updated discriminator weights and bias. Report g_loss before updating W_g and b_g. Both updates use the supplied learning rate lr, with no momentum or weight decay. Return the updated W_g, b_g, W_d, b_d and both losses without rounding.</span>

---

## <span style="font-size: 16px;">Architecture Design</span>

<span style="font-size: 14px;">**Generator architecture**: Maps low-dimensional noise to high-dimensional data. For MLP GANs: Linear layers with ReLU activations, Tanh output to bound values in $[-1, 1]$. For image GANs (DCGAN): ConvTranspose2d layers with BatchNorm and ReLU, Tanh output.</span>

<span style="font-size: 14px;">**Discriminator architecture**: Maps data to a scalar probability. For MLP GANs: Linear layers with LeakyReLU(0.2) activations (not ReLU - avoids dead neurons for the discriminator), Sigmoid output. For image GANs: strided Conv2d with BatchNorm and LeakyReLU, Sigmoid output.</span>

<span style="font-size: 14px;">LeakyReLU with slope 0.2 for the discriminator is a widely adopted convention from DCGAN. It ensures gradients flow even for negative inputs, which is important because the discriminator receives gradients from both real and fake paths.</span>

<span style="font-size: 14px;">**Latent dimension**: typically 64-128. Too small limits generator expressiveness; too large makes the prior hard to match. For simple datasets like MNIST, 16-32 suffices.</span>

---

## <span style="font-size: 16px;">Training Challenges</span>

<span style="font-size: 14px;">**Mode collapse**: G learns to produce only a few types of outputs that fool D, ignoring the full diversity of the data. Symptom: low diversity in generated samples. Solutions include minibatch discrimination, unrolled GANs, and using Wasserstein loss (WGAN).</span>

<span style="font-size: 14px;">**Training instability**: the D/G balance is fragile. If D becomes too strong, G gradients vanish. If D becomes too weak, G gets no useful signal. Learning rate tuning and update ratios (e.g., training D more than G) help, but there is no universal recipe.</span>

<span style="font-size: 14px;">**Non-convergence**: unlike supervised learning, GAN training does not minimize a single loss - it is a two-player game that may oscillate rather than converge. The losses themselves are not reliable training indicators.</span>

<span style="font-size: 14px;">**Evaluation difficulty**: FID (Frechet Inception Distance) and IS (Inception Score) are standard metrics, but they have known limitations. There is no single number that fully captures generation quality and diversity.</span>

---

## <span style="font-size: 16px;">Key Implementation Details</span>

<span style="font-size: 14px;">**Separate parameter gradients**: During the discriminator phase, treat generated samples as constants. During the generator phase, differentiate through the fixed updated discriminator to the generator. Updating only the parameters assigned to each phase preserves the alternating procedure.</span>

<span style="font-size: 14px;">**Reuse supplied noise**: Both phases use the same input z. Sampling replacement noise would change this deterministic exercise's gradients and outputs.</span>

<span style="font-size: 14px;">**Separate SGD updates**: Apply each gradient to its own model using lr. This exercise has no optimizer state. Other GAN training setups may use separate Adam optimizers, but their update rule is outside this task.</span>

<span style="font-size: 14px;">**Loss convention**: If every discriminator logit is zero, each mean BCE term is ln(2), so this task's d_loss is 2 ln(2), approximately 1.3863. A generator loss evaluated at zero logits is ln(2), approximately 0.6931. Loss values depend on the specified reduction and measurement time.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


**Q: Why do we detach fake data when computing the discriminator loss?**

A: <span style="font-size: 14px;">When we compute discriminator_loss(D, real, G(z)), the fake data G(z) is part of G's computation graph. If we backpropagate through D(G(z)) without detaching, the discriminator's loss.backward() would compute gradients for both D and G parameters. Then optimizer_D.step() would update D, but G's gradient buffers would also be populated with gradients from the discriminator's objective - corrupting the subsequent G update. Detaching creates a new tensor with the same values but no gradient history, cleanly separating the two optimization steps.</span>

**Q: What is mode collapse and how do you detect/prevent it?**

A: <span style="font-size: 14px;">Mode collapse occurs when the generator maps diverse noise vectors to a small set of outputs, "collapsing" to a few modes of the data distribution. Detection: visually inspect generated samples for repetition, measure diversity metrics, or check if the generator output variance is low. Prevention approaches: (1) Wasserstein loss (WGAN) provides smoother gradients that don't saturate, (2) minibatch discrimination adds a diversity-encouraging term, (3) spectral normalization stabilizes the discriminator, (4) progressive growing starts with low resolution and gradually adds detail.</span>

**Q: Explain the non-saturating generator loss trick.**

A: <span style="font-size: 14px;">The original minimax objective has G minimize $\log(1 - D(G(z)))$. When D is strong (early training), $D(G(z)) \approx 0$, so $\log(1 - 0) \approx 0$ - the gradient is nearly flat, giving G almost no learning signal. The non-saturating trick instead maximizes $\log D(G(z))$, equivalent to minimizing $\text{BCE}(D(G(z)), 1)$. When $D(G(z)) \approx 0$, $\log(0)$ gives very large gradients, providing strong signal for G to improve. Both formulations have the same fixed point but different gradient dynamics.</span>

**Q: Why use LeakyReLU instead of ReLU in the discriminator?**

A: <span style="font-size: 14px;">ReLU zeros out negative activations, which can create "dead neurons" that never activate and never receive gradients. The discriminator is especially vulnerable because it receives gradients from two different distributions (real and fake). LeakyReLU with slope 0.2 ensures gradients always flow through, even for negative pre-activations. This is critical for stable training because the generator relies on gradients from the discriminator to improve - dead discriminator neurons mean lost signal for the generator.</span>

**Q: How do you evaluate GAN quality without a reconstruction loss?**

A: <span style="font-size: 14px;">Unlike VAEs, GANs have no single loss to track. Standard metrics: (1) FID (Frechet Inception Distance) measures the distance between real and generated feature distributions using a pre-trained Inception network - lower is better. (2) Inception Score (IS) measures both quality and diversity of generated images. (3) Visual inspection of samples and interpolations. (4) Precision/Recall metrics separate quality from diversity. D and G losses are unreliable - low D loss can mean either a strong discriminator or mode collapse. The most reliable approach combines FID tracking with periodic visual inspection.</span>

---