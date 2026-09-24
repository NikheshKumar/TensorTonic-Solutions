# <span style="font-size: 20px;">Autoencoder</span>

<span style="font-size: 14px;">Autoencoders are neural networks trained to reconstruct their input through a bottleneck, learning compressed representations in the process. They consist of an encoder that maps inputs to a low-dimensional latent space and a decoder that maps back to the original space. Autoencoders are foundational to representation learning and serve as building blocks for more advanced generative models like VAEs and diffusion models.</span>

---

## <span style="font-size: 16px;">Architecture</span>

<span style="font-size: 14px;">An autoencoder has three parts:</span>

<span style="font-size: 14px;">**Encoder** $f_\theta$: maps input $x \in \mathbb{R}^D$ to a latent vector $z \in \mathbb{R}^d$ where $d \ll D$. For images, this typically uses strided convolutions to reduce spatial dimensions, followed by flattening and a linear projection to the latent dimension.</span>

<span style="font-size: 14px;">**Bottleneck**: the latent vector $z$ is the compressed representation. Its dimensionality determines the information capacity - too large and the model can memorize; too small and it cannot capture enough structure.</span>

<span style="font-size: 14px;">**Decoder** $g_\phi$: maps $z$ back to $\hat{x} \in \mathbb{R}^D$. Mirrors the encoder: a linear projection to the flattened feature map size, reshape, then transposed convolutions to upsample. A Sigmoid output activation constrains pixel values to $[0, 1]$.</span>

$$
\hat{x} = g_\phi(f_\theta(x)), \quad \mathcal{L} = \|x - \hat{x}\|^2
$$

---

## <span style="font-size: 16px;">Convolutional Autoencoders</span>

<span style="font-size: 14px;">For image data, fully-connected autoencoders are impractical due to the high dimensionality. Convolutional autoencoders use:</span>

<span style="font-size: 14px;">**Encoder**: strided convolutions (stride=2) halve spatial dimensions at each layer. With kernel=3, stride=2, padding=1, an input of size $H$ becomes $\lfloor(H-1)/2\rfloor + 1$. For $H = 32$: $32 \to 16 \to 8$.</span>

<span style="font-size: 14px;">**Decoder**: transposed convolutions (stride=2, output_padding=1) double spatial dimensions. With kernel=3, stride=2, padding=1, output_padding=1: output $= 2 \times \text{input}$. For $8 \to 16 \to 32$.</span>

<span style="font-size: 14px;">Between the conv and deconv stages, linear layers connect the flattened feature maps to the latent space. This is the true bottleneck - even if spatial dimensions are large, the latent vector constrains information flow.</span>

<span style="font-size: 14px;">The encoder channels typically increase (e.g., 32, 64, 128) while spatial dims decrease. The decoder reverses this pattern.</span>

---

## <span style="font-size: 16px;">Loss Functions</span>

<span style="font-size: 14px;">**MSE (L2) loss**: $\mathcal{L} = \frac{1}{N} \sum \|x_i - \hat{x}_i\|^2$. Standard choice for continuous-valued pixels. Tends to produce blurry reconstructions because it penalizes each pixel independently.</span>

<span style="font-size: 14px;">**BCE loss**: $\mathcal{L} = -\sum [x \log \hat{x} + (1-x) \log(1-\hat{x})]$. Appropriate when pixel values are in $[0, 1]$ (after Sigmoid). Treats each pixel as a Bernoulli variable. Often produces sharper results than MSE.</span>

<span style="font-size: 14px;">**Perceptual loss**: compares feature representations from a pre-trained network (e.g., VGG) rather than raw pixels. Captures high-level structure. Used in modern autoencoders like the one in Stable Diffusion's VAE.</span>

---

## <span style="font-size: 16px;">Latent Space Properties</span>

<span style="font-size: 14px;">A well-trained autoencoder learns a latent space where similar inputs map to nearby points. However, vanilla autoencoders have no constraint on the latent space structure - it may be discontinuous, have "holes", or have irregular density. This makes it unsuitable for generation (random sampling from latent space may produce nonsensical outputs).</span>

<span style="font-size: 14px;">**Regularization approaches**:</span>
- <span style="font-size: 14px;">**VAE**: adds KL divergence to regularize the latent space to match a prior (typically $\mathcal{N}(0, I)$)</span>
- <span style="font-size: 14px;">**Sparse autoencoder**: adds L1 penalty on latent activations, encouraging most dimensions to be near zero</span>
- <span style="font-size: 14px;">**Contractive autoencoder**: adds penalty on the Jacobian of the encoder, making the representation robust to small input perturbations</span>
- <span style="font-size: 14px;">**Denoising autoencoder**: trained to reconstruct clean input from corrupted input, learning more robust features</span>

---

## <span style="font-size: 16px;">Applications</span>

<span style="font-size: 14px;">**Dimensionality reduction**: autoencoders learn non-linear compression, generalizing PCA to non-linear manifolds.</span>

<span style="font-size: 14px;">**Anomaly detection**: train on normal data, then flag inputs with high reconstruction error as anomalies.</span>

<span style="font-size: 14px;">**Pre-training**: encoder weights can initialize downstream tasks (classification, segmentation).</span>

<span style="font-size: 14px;">**Feature learning**: the latent space captures meaningful features - used in recommendation systems, drug discovery, and image retrieval.</span>

<span style="font-size: 14px;">**Generative models**: VAEs extend autoencoders with probabilistic latent spaces. The autoencoder in Stable Diffusion compresses images to a latent space where diffusion operates more efficiently.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

<span style="font-size: 14px;">Common follow-up questions in deep learning interviews:</span>


**Q: Why does the autoencoder need a bottleneck? What happens without one?**

A: <span style="font-size: 14px;">Without a bottleneck (latent_dim >= input_dim), the model can learn the identity function - simply copy the input to the output without learning any useful representation. The bottleneck forces the model to discover and retain only the most important features, discarding noise and redundancy. The optimal bottleneck size depends on the intrinsic dimensionality of the data.</span>

**Q: Why do autoencoder reconstructions tend to be blurry?**

A: <span style="font-size: 14px;">MSE loss penalizes each pixel independently and equally. When the model is uncertain about a pixel value, the optimal MSE prediction is the mean of possible values, which appears as a blur. This is especially visible for high-frequency details like edges and textures. Perceptual losses, adversarial losses (as in VAE-GAN), or combining with diffusion models can produce sharper results.</span>

**Q: Compare autoencoders with PCA for dimensionality reduction.**

A: <span style="font-size: 14px;">PCA finds the optimal linear projection that maximizes variance - it is equivalent to a single-layer linear autoencoder with MSE loss. Autoencoders with non-linear activations can capture non-linear manifolds that PCA cannot represent. However, PCA has closed-form solutions, is deterministic, and provides orthogonal components with explained variance ratios. Autoencoders are harder to train, may converge to local optima, but can learn much richer representations.</span>

**Q: How do denoising autoencoders relate to diffusion models?**

A: <span style="font-size: 14px;">Both learn to remove noise from corrupted inputs. A denoising autoencoder is trained on a single noise level, while diffusion models are trained on a continuous schedule of noise levels from clean to fully noisy. Diffusion models can be viewed as a sequence of denoising autoencoders applied iteratively. Score matching (the theoretical foundation of diffusion) is closely related to denoising autoencoder training.</span>

**Q: When would you use a convolutional autoencoder vs a fully-connected one?**

A: <span style="font-size: 14px;">Convolutional autoencoders are strongly preferred for image data because: (1) parameter sharing through convolution kernels dramatically reduces parameters, (2) translation equivariance captures spatial structure naturally, (3) strided convolutions provide hierarchical downsampling. Fully-connected autoencoders are used for tabular data, small feature vectors, or when spatial structure is not relevant (e.g., molecular fingerprints).</span>

---