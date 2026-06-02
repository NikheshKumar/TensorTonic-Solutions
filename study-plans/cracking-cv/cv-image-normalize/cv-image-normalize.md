# <span style="font-size: 20px;">Image Normalize</span>

<span style="font-size: 14px;">Image normalization rescales each colour channel so that it is roughly **zero-mean and unit-variance**: subtract a per-channel mean and divide by a per-channel standard deviation. It is the near-universal final preprocessing step before an image enters a CNN or vision transformer, and it is the operation every ImageNet pipeline applies after converting pixels to floats.</span>

---

## <span style="font-size: 16px;">The Formula</span>

<span style="font-size: 14px;">For an image of shape $(H, W, C)$, normalization is applied independently to each channel $c$ using a scalar mean $\mu_c$ and standard deviation $\sigma_c$ for that channel:</span>

$$
\text{normalized}[h, w, c] = \frac{\text{image}[h, w, c] - \mu_c}{\sigma_c}
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$\text{image}[h, w, c]$ is the input pixel value at row $h$, column $w$, channel $c$, typically already scaled to $[0, 1]$ by dividing the raw $[0, 255]$ values by $255$</span>
* <span style="font-size: 14px;">$\mu_c$ is the mean of channel $c$, a single number broadcast across every spatial location</span>
* <span style="font-size: 14px;">$\sigma_c$ is the standard deviation of channel $c$, also broadcast across all positions</span>
* <span style="font-size: 14px;">the output has the same shape $(H, W, C)$ as the input, but values now centered near $0$ and scaled to roughly unit spread</span>

<span style="font-size: 14px;">The transform is **affine and per-channel**: every pixel in channel $c$ undergoes the exact same shift-and-scale. There is no spatial mixing and no interaction between channels.</span>

---

## <span style="font-size: 16px;">Why Normalize at All</span>

<span style="font-size: 14px;">Centering and scaling the inputs makes the optimization landscape far better conditioned, which is why this step is so universal. The concrete reasons:</span>

* <span style="font-size: 14px;">**Balanced gradient magnitudes.** The gradient of a weight in the first layer is proportional to its input. If raw pixels live in $[0, 255]$, the gradients are about $255\times$ larger than they would be for unit-scale inputs, forcing tiny learning rates. Zero-mean unit-variance inputs keep first-layer gradients well scaled.</span>
* <span style="font-size: 14px;">**Faster, more stable convergence.** Centering removes a large constant bias term that the network would otherwise have to learn and subtract. Scaling equalizes the effective step size across channels, so SGD does not oscillate along high-variance directions while crawling along low-variance ones.</span>
* <span style="font-size: 14px;">**Channel parity.** If the red channel happened to have much larger variance than blue, an unnormalized network would initially weight red far more heavily simply due to scale, not signal. Per-channel scaling removes this accidental bias.</span>
* <span style="font-size: 14px;">**Match to weight initialization.** Schemes like Xavier and Kaiming initialization assume inputs with roughly unit variance. Feeding raw $[0, 255]$ pixels violates that assumption and pushes activations into saturating regions of the nonlinearity from the very first forward pass.</span>

---

## <span style="font-size: 16px;">The ImageNet Convention</span>

<span style="font-size: 14px;">The most widely used constants come from the ImageNet training set, computed once over the full corpus after scaling pixels to $[0, 1]$. In RGB order they are:</span>

* <span style="font-size: 14px;">**Mean:** $\mu = [0.485, 0.456, 0.406]$</span>
* <span style="font-size: 14px;">**Std:** $\sigma = [0.229, 0.224, 0.225]$</span>

<span style="font-size: 14px;">These exact numbers appear in nearly every PyTorch and TensorFlow vision recipe (`torchvision.transforms.Normalize(mean, std)`), because pretrained backbones such as ResNet, VGG, and ViT were trained on inputs normalized this way. To fine-tune or run inference with a pretrained model, the input must be normalized with the same constants the model saw during training; otherwise the activation statistics shift and accuracy degrades. The means hover near $0.45$ because natural images are, on average, moderately bright; the standard deviations cluster near $0.22$ because typical photo contrast is similar across channels.</span>

---

## <span style="font-size: 16px;">Dataset Statistics vs Fixed Constants</span>

<span style="font-size: 14px;">There are two regimes. When training from scratch on a custom dataset, the correct practice is to compute $\mu_c$ and $\sigma_c$ from that dataset's training split and reuse them at test time. When using a pretrained backbone, the correct practice is to reuse the backbone's original statistics (usually ImageNet's). Mixing these up - for example computing fresh statistics for a model pretrained on ImageNet - silently shifts the input distribution away from what the frozen early layers expect.</span>

<span style="font-size: 14px;">A subtlety: the statistics must be computed on the **training set only** and then applied unchanged to validation and test data. Computing test-set statistics leaks information and gives optimistically biased evaluation numbers, the same data-leakage trap that appears with any feature scaler in classical machine learning.</span>

---

## <span style="font-size: 16px;">What "Unit Variance" Buys Geometrically</span>

<span style="font-size: 14px;">Geometrically, the raw input cloud for a channel is an interval roughly centered somewhere in $[0, 1]$ with some spread. Subtracting $\mu_c$ slides that cloud so its center sits at the origin. Dividing by $\sigma_c$ then stretches or shrinks it so its spread is $1$. After both steps, every channel occupies the same standardized range regardless of its original brightness or contrast.</span>

<span style="font-size: 14px;">This matters for the dot products that a linear or convolutional layer computes. A neuron's pre-activation is $\sum_c w_c x_c + b$. If the $x_c$ have wildly different scales, the weights $w_c$ must compensate with equally mismatched magnitudes, which makes the loss surface an elongated, ill-conditioned valley. Standardizing the inputs makes the valley closer to spherical, so gradient descent takes a more direct path to the minimum and tolerates a larger, single learning rate across all channels.</span>

---

## <span style="font-size: 16px;">Worked Example (one pixel, 3 channels)</span>

<span style="font-size: 14px;">Take a single RGB pixel already scaled to $[0, 1]$: $[0.6, 0.5, 0.4]$, and use the ImageNet constants $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.</span>

<span style="font-size: 14px;">1. **Red channel**: $(0.6 - 0.485)/0.229 = 0.115 / 0.229 = 0.5022$</span>

<span style="font-size: 14px;">2. **Green channel**: $(0.5 - 0.456)/0.224 = 0.044 / 0.224 = 0.1964$</span>

<span style="font-size: 14px;">3. **Blue channel**: $(0.4 - 0.406)/0.225 = -0.006 / 0.225 = -0.0267$</span>

<span style="font-size: 14px;">The normalized pixel is $[0.5022, 0.1964, -0.0267]$. Note the blue value went slightly negative: its raw value $0.4$ sits just below the blue mean $0.406$, so after centering it lands below zero. Negative outputs are normal and expected, which is why the output cannot be stored as an unsigned 8-bit image. The magnitudes also reveal that red sits about half a standard deviation above its typical value while green is only a fifth of a sigma above and blue is essentially average, a compact numeric summary of the pixel's colour relative to the dataset.</span>

---

## <span style="font-size: 16px;">A Small Patch Through the Transform</span>

<span style="font-size: 14px;">Consider a single-channel $2 \times 2$ patch (so $C = 1$) with values $\begin{pmatrix} 0.485 & 0.714 \\ 0.256 & 0.943 \end{pmatrix}$, mean $\mu = 0.485$, std $\sigma = 0.229$.</span>

* <span style="font-size: 14px;">**Top-left**: $(0.485 - 0.485)/0.229 = 0$ exactly, since the value equals the mean</span>
* <span style="font-size: 14px;">**Top-right**: $(0.714 - 0.485)/0.229 = 0.229/0.229 = 1.0$, exactly one standard deviation above the mean</span>
* <span style="font-size: 14px;">**Bottom-left**: $(0.256 - 0.485)/0.229 = -0.229/0.229 = -1.0$, one std below</span>
* <span style="font-size: 14px;">**Bottom-right**: $(0.943 - 0.485)/0.229 = 0.458/0.229 = 2.0$, two std above</span>

<span style="font-size: 14px;">The output $\begin{pmatrix} 0 & 1 \\ -1 & 2 \end{pmatrix}$ is in units of standard deviations. This is the precise meaning of normalization: every value is re-expressed as "how many sigmas from the channel mean".</span>

<span style="font-size: 14px;">It is worth confirming that the constants used to normalize a patch are not derived from that patch. The mean $0.485$ and std $0.229$ above are the channel-wide (dataset-wide) statistics; the four patch values happen to give a clean answer here only because they were chosen to land at exact multiples of $\sigma$. In a real image the four pixels of a patch would not be centered at the dataset mean, and that is fine: normalization standardizes against the global distribution, not the local one.</span>

---

## <span style="font-size: 16px;">Relationship to Batch and Layer Norm</span>

<span style="font-size: 14px;">Input normalization is the static, dataset-level cousin of the learned normalizers inside the network:</span>

* <span style="font-size: 14px;">**Input normalization** uses fixed constants computed once, applied only at the input. It has no learnable parameters and never changes during training.</span>
* <span style="font-size: 14px;">**Batch normalization** (Ioffe and Szegedy, 2015) recomputes mean and variance from the current mini-batch at every layer, then applies learnable scale and shift. It addresses internal covariate shift between layers, which static input scaling cannot.</span>
* <span style="font-size: 14px;">**Layer normalization** normalizes across the feature dimension of each sample and dominates in transformers.</span>

<span style="font-size: 14px;">Input normalization does not replace these; it complements them by giving the first layer a well-scaled starting point. The formula is identical in spirit ($(x - \mu)/\sigma$); only the source of $\mu$ and $\sigma$ and whether learnable parameters follow differ.</span>

---

## <span style="font-size: 16px;">Per-Channel Scaling vs Full Whitening</span>

<span style="font-size: 14px;">The transform here divides each channel by its own standard deviation but does nothing about correlations between channels. Natural-image channels are heavily correlated (a bright pixel tends to be bright in all three channels), so the standardized cloud is still a tilted ellipsoid, not a sphere.</span>

<span style="font-size: 14px;">Full **whitening** (also called ZCA or PCA whitening) goes further: it multiplies by the inverse square root of the full covariance matrix $\Sigma^{-1/2}$, decorrelating the channels and giving an isotropic, spherical cloud. Whitening was popular in early feature-learning work but is rarely used as input preprocessing today because it is expensive, sensitive to the covariance estimate, and offers little benefit once batch normalization handles intra-network conditioning. Per-channel standardization is the cheap, robust 90% solution, which is why it became the standard. The key conceptual point is that this problem implements only the diagonal part of whitening: scale each channel independently, ignore cross-channel structure.</span>

---

## <span style="font-size: 16px;">Invertibility and Denormalization</span>

<span style="font-size: 14px;">Unlike grayscale conversion, per-channel normalization is exactly invertible because $\sigma_c$ is nonzero. Given a normalized value $z$, the original is recovered by $x = z \sigma_c + \mu_c$. This denormalization step is essential whenever a model's output must be displayed: a colourization or super-resolution network operating on normalized inputs produces normalized outputs, which have to be multiplied by $\sigma$, shifted by $\mu$, clamped to $[0, 1]$, and scaled by $255$ before they become a viewable image. Forgetting this inverse is a frequent cause of washed-out or garishly tinted visualizations.</span>

---

## <span style="font-size: 16px;">Implementation Notes</span>

<span style="font-size: 14px;">Vectorized, the operation broadcasts a length-$C$ mean and std across the $(H, W)$ spatial grid. In NumPy with a channels-last array `img` of shape $(H, W, C)$: `(img - mean) / std`, where `mean` and `std` are shape $(C,)$ and broadcast over the trailing axis. In PyTorch the convention is channels-first $(C, H, W)$, so the constants are reshaped to $(C, 1, 1)$ before broadcasting. The cost is $O(H \cdot W \cdot C)$ subtract-divide operations, and the result is dense floating point that must not be cast back to `uint8`.</span>

<span style="font-size: 14px;">Because the transform is affine and per-channel, it composes cleanly with the rest of the pipeline. The typical preprocessing chain is: load to `uint8`, convert to float and divide by $255$, optionally resize, then normalize. The order matters: resizing interpolates colour values and should happen before normalization so the interpolation works on intuitive $[0, 1]$ ranges rather than on signed standardized values, though mathematically the linearity means the two orders differ only by rounding.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Forgetting to scale to $[0, 1]$ first.** The ImageNet constants assume inputs already divided by $255$. Applying $\mu = 0.485$ to raw $[0, 255]$ pixels barely shifts them, leaving the data effectively unnormalized and the model badly miscalibrated.</span>
* <span style="font-size: 14px;">**Channel order mismatch (RGB vs BGR).** The ImageNet mean and std are listed in RGB order. If the image is loaded as BGR (OpenCV default), the red and blue statistics are applied to the wrong channels, producing a subtle but real distribution shift.</span>
* <span style="font-size: 14px;">**Storing the result as an unsigned image.** Normalized values are signed and routinely negative or above $1$. Casting to `uint8` clips and wraps them, destroying the very property normalization was meant to create. Keep the output in float.</span>
* <span style="font-size: 14px;">**Recomputing statistics on the test set.** Statistics must come from the training split only. Computing them on validation or test data leaks information and inflates reported accuracy.</span>
* <span style="font-size: 14px;">**Dividing by a zero or near-zero std.** A constant channel (for example an all-black alpha or a padding channel) has $\sigma_c = 0$, and dividing by it yields infinities or NaNs that propagate through the whole network. Guard with a small epsilon or skip constant channels entirely, since a degenerate channel carries no information to standardize.</span>

---