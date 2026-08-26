## <span style="font-size: 20px;">Image Representation: The Raw Material</span>

- A digital image is a 3D array of numbers: height, width, and color channel
- The most common color model is RGB, where each pixel has three values: red, green, and blue intensity
- A 224x224 RGB image contains 224 x 224 x 3 = 150,528 individual values

## HWC vs CHW Layout

- Most image libraries (PIL, OpenCV, matplotlib) store images in HWC order: $(H, W, C)$. Each row of pixels is contiguous, and the three channel values for a single pixel sit next to each other
- PyTorch convolutional layers expect CHW order: $(C, H, W)$. All spatial values for the red channel are contiguous, followed by green, followed by blue
- CHW is computationally advantageous for convolutions: a 2D kernel operates on a single channel at a time, and having each channel's spatial data contiguous improves cache locality and enables efficient vectorized operations
- When you load an image from disk, it arrives in HWC format. You must transpose to CHW before feeding it to a PyTorch model. This is not merely convention but a requirement baked into the convolution implementation

## Data Types: uint8 vs float32

- Image files encode pixel intensities as unsigned 8-bit integers (uint8), range $[0, 255]$. Value 0 = no intensity (black), 255 = full intensity. Three bytes per pixel, so a 224x224 image uses about 147 KB uncompressed
- Neural networks operate on 32-bit floats. Integer arithmetic is not differentiable: working with integers directly would prevent gradients from flowing
- Beyond differentiability, the scale matters. Pixel values in $[0, 255]$ are large relative to typical neural network weight magnitudes (usually near zero), producing enormous activations and gradients and making training numerically unstable

## Why Neural Networks Need Normalized Inputs

## The Optimization Landscape Argument

- When input features have very different magnitudes, the loss surface becomes elongated in some directions and compressed in others
- Consider a linear model $y = w_1 x_1 + w_2 x_2$ where $x_1 \in [0, 255]$ and $x_2 \in [0, 1]$. A small change in $w_1$ produces a large output change (multiplied by large input), while the same change in $w_2$ produces a tiny change
- The loss surface becomes an elongated valley, and gradient descent zig-zags inefficiently
- Normalizing both inputs to the same scale makes the surface more isotropic (equally curved in all directions), enabling faster convergence

## The Gradient Flow Argument

- With small initial weights and inputs in $[0, 255]$, pre-activation values $z = Wx + b$ are large
- For sigmoid or tanh, large inputs push into saturation where gradients are nearly zero
- Even for ReLU, unnormalized inputs produce activations with large variance, causing exploding gradients in deeper layers
- Normalizing inputs to zero mean and unit variance keeps pre-activations in the linear region of activation functions, where gradients flow freely

## The Batch Normalization Connection

- Batch normalization normalizes activations at each layer during training
- Input normalization serves the same purpose for the very first layer
- You can think of input normalization as batch normalization applied to layer zero

## The Normalization Pipeline

## Convert to Float and Scale to [0, 1]

$$
I_{\text{float}} = \frac{I_{\text{uint8}}}{255.0}
$$

- Maps every pixel to $[0, 1]$. The divisor 255 is simply $2^8 - 1$
- In torchvision, the ToTensor transform handles float conversion, 255 scaling, and HWC-to-CHW permutation simultaneously. This bundling is convenient but can cause confusion if you are not aware it does three things at once

## Channel Reordering

- The image must be transposed from $(H, W, C)$ to $(C, H, W)$
- The permute operation returns a view of the same data with different stride information; it does not copy the data. This is efficient but the resulting tensor may not be contiguous in memory

## Per-Channel Standardization

$$
I_c^{\text{norm}} = \frac{I_c - \mu_c}{\sigma_c}
$$

- $\mu_c$ and $\sigma_c$ are the mean and standard deviation for channel $c$, computed over a reference dataset
- Result: each channel has approximately zero mean and unit variance (if the image comes from the same distribution as the reference)
- For broadcasting to work correctly, the mean and std vectors (length $C$) must be reshaped to $(C, 1, 1)$ to align with the channel dimension and broadcast across spatial dimensions
- If you forget to reshape and leave mean as shape $(3,)$, PyTorch will broadcast along the wrong dimension, subtracting different means from different spatial positions rather than from different channels

## ImageNet Statistics and Why They Are Used

- $\mu = [0.485, 0.456, 0.406]$ (R, G, B means)
- $\sigma = [0.229, 0.224, 0.225]$ (R, G, B standard deviations)
- Computed over the ImageNet ILSVRC 2012 training set (~1.2 million images across 1,000 categories)
- Every major pretrained vision model (ResNet, VGG, Inception, EfficientNet) was trained with these statistics
- The means are not exactly $[0.5, 0.5, 0.5]$ because natural images are not uniformly distributed in color space. Red has the highest mean (warm tones: skin, earth, sunlight), green slightly lower, blue lowest
- The standard deviations are all close to 0.22-0.23, indicating similar channel spread once the mean is accounted for

## When to Use ImageNet Statistics vs Your Own

- **Pretrained model (transfer learning):** always normalize with ImageNet statistics, because the model's weights were learned under that assumption. Changing normalization effectively feeds the model out-of-distribution data
- **Training from scratch on domain-specific data** (medical X-rays, satellite imagery): ImageNet statistics are unlikely to match your distribution. Compute your own by iterating over all training images and accumulating per-channel mean and standard deviation

## Computing Your Own Statistics

$$
\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n=1}^{N} \sum_{i=1}^{H} \sum_{j=1}^{W} x^{(n)}_{c,i,j}
$$

$$
\begin{aligned}
\sigma_c = \bigg(\frac{1}{N H W}
&\sum_{n=1}^{N} \sum_{i=1}^{H} \sum_{j=1}^{W} \\
&\left(x^{(n)}_{c,i,j} - \mu_c\right)^2 \bigg)^{1/2}
\end{aligned}
$$

- Requires two passes (one for mean, one for variance), or a single pass using Welford's online algorithm
- A practical approach uses a DataLoader with only the float/tensor conversion applied (no normalization) and accumulates statistics batch by batch

## Effect of Normalization on Gradient Flow

For a convolutional filter $w$ applied to an input channel, the output at one spatial position:

$$
z = \sum_{i,j} w_{ij} \cdot x_{ij} + b
$$

The gradient with respect to weight $w_{ij}$:

$$
\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial z} \cdot x_{ij}
$$

- The gradient is proportional to the input $x_{ij}$. If inputs are in $[0, 255]$, gradients are 255 times larger than for inputs in $[0, 1]$. You would need a learning rate 255 times smaller to get the same effective step size
- Different channels with different scales would require different learning rates, but standard optimizers use a single rate. Normalizing ensures similar gradient magnitudes across channels
- When all input values are positive (as with $[0, 1]$ or $[0, 255]$), all weight gradients share the same sign as the upstream gradient. Weight updates can only move all weights in the same direction simultaneously, leading to inefficient zig-zagging
- Centering inputs around zero allows gradients to have both positive and negative values, enabling more direct optimization paths

## Data Augmentation Transforms

Data augmentation applies random transformations to training images so the model sees slightly different versions at each epoch. This acts as a powerful regularizer, implicitly encoding that labels should be invariant to certain visual changes.

- **RandomResizedCrop:** randomly selects a rectangular region (random aspect ratio and area), resizes to target size. Forces recognition at different scales and positions. Standard recipe: area from $[8\%, 100\%]$ of original, aspect ratio from $[3/4, 4/3]$
- **RandomHorizontalFlip:** mirrors left-to-right with probability 0.5. One of the cheapest and most effective augmentations for natural images. Vertical flipping is less common (world has consistent up direction) but appropriate for aerial or microscopy images
- **RandomRotation:** rotates by angle from a specified range (e.g., $[-15, 15]$ degrees). Introduces empty corners that must be filled. Too much rotation can harm tasks where orientation matters (a 6 rotated 180 degrees becomes a 9)
- **ColorJitter:** randomly perturbs brightness, contrast, saturation, and hue. Makes the model robust to lighting conditions. Large hue shifts should be used carefully because they can change apparent object identity (red car shifted to blue)
- **GaussianBlur:** applies blur with random kernel size, simulating defocus
- **RandomAffine:** combines rotation, translation, scaling, and shearing into a single affine transformation
- **RandomErasing (Cutout):** randomly masks out a rectangular region with random values, forcing predictions from partial information
- **RandomGrayscale:** converts to grayscale with some probability, encouraging shape-based rather than color-based recognition

## The Compose Pattern

- Torchvision's Compose takes a list of transforms and chains them sequentially: the output of one becomes the input to the next
- It is itself a callable: calling it on an image applies each transform in sequence
- Order matters critically:
  - Spatial augmentations (crop, flip, rotation) must come before the tensor conversion because they operate on PIL images
  - Normalization must come after tensor conversion because it operates on float tensors
  - Placing normalization before tensor conversion would fail because the image is still a PIL object

## Callable Classes as Transforms

- Each transform is implemented as a class with a call method
- This design is essential because many transforms have parameters to store (mean, std, crop size)
- A class stores configuration as instance attributes and applies them when called
- This makes transforms composable, serializable, and inspectable

## Custom Transforms

- Create a callable class that takes an image (PIL or tensor) and returns the transformed image
- Can be inserted into a Compose pipeline alongside built-in transforms
- Useful for domain-specific preprocessing: specific color space conversions, frequency-domain filtering, physics-based augmentations for scientific imaging

## Torchvision Transforms v1 vs v2

## The v1 API

- Accessible via the standard transforms module, operates primarily on PIL images
- Some transforms also accept tensors, but the primary design assumes PIL input
- Critical limitation: cannot jointly transform images and associated labels. For object detection or segmentation, bounding boxes or masks must be transformed in sync with the image (flip the image, flip the boxes too), and v1 requires manual coordination

## The v2 API

- Introduced in torchvision 0.15, solves the joint transformation problem
- Accepts arbitrary input structures: images, bounding boxes, masks, and videos, all passed together
- Transforms know how to apply the correct transformation to each type (e.g., horizontal flip mirrors the image and simultaneously mirrors bounding box coordinates)
- Operates natively on tensors (not just PIL), supports batches, and enables GPU acceleration for some operations
- For new projects, v2 is recommended. For simple classification (image-only transforms), v1 and v2 are functionally equivalent

## Tensor vs PIL Transforms

- In v1, spatial transforms (crop, rotation, color jitter) are implemented for PIL images. Tensor operations (normalization) operate on tensors. The tensor conversion serves as the bridge. Place all PIL transforms before conversion, all tensor transforms after
- In v2, this distinction is largely eliminated. Most transforms accept both PIL and tensor inputs. The preferred workflow converts from PIL to tensor early in the pipeline

## Preprocessing for Different Architectures

Different model architectures expect different input sizes and preprocessing. Using the wrong preprocessing degrades performance, sometimes catastrophically.

- **ResNet and classic CNNs:** 224x224 inputs normalized with ImageNet statistics. Training: RandomResizedCrop(224) + RandomHorizontalFlip. Evaluation: Resize(256) then CenterCrop(224). The evaluation pipeline first resizes the shorter edge to 256, then takes a center crop to ensure consistent input size without distortion
- **Vision Transformers (ViT):** split input into fixed-size patches (typically 16x16 or 14x14). Original paper used 224x224 or 384x384. Positional embeddings are learned for a specific grid size; different resolution requires interpolation. Uses ImageNet normalization when pretrained on ImageNet
- **Inception v3:** expects 299x299 inputs with normalization to $[-1, 1]$ (mean $= [0.5, 0.5, 0.5]$, std $= [0.5, 0.5, 0.5]$). Using ImageNet statistics with Inception produces subtly wrong results
- **Pretrained weights API** (torchvision 0.13+): attaches the correct preprocessing transforms directly to pretrained weight enums. Returns the exact transform pipeline used during training, including correct resize, crop, normalization, and interpolation mode. Eliminates mismatch risk

## Advanced Augmentation Strategies

Manually selecting augmentation types and magnitudes is tedious and suboptimal. Automated strategies:

- **AutoAugment:** uses reinforcement learning to search for the best augmentation policy. A policy consists of 25 sub-policies, each containing two operations with associated probability and magnitude. The resulting policy is dataset-specific. Available in torchvision
- **RandAugment:** dramatically simplifies the search space to two hyperparameters: $N$ (number of operations to apply) and $M$ (global magnitude). At each step, $N$ operations are randomly selected from a fixed pool, each applied with magnitude $M$. Matches or exceeds AutoAugment with far less computational cost
- **TrivialAugment:** applies exactly one randomly selected operation with randomly selected magnitude per step. Zero hyperparameters to tune. Despite extreme simplicity, achieves competitive results. Key insight: randomness provides sufficient diversity over many epochs
- **Mixup:** creates synthetic examples by linearly interpolating pairs of images and labels:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j, \quad \tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$. Encourages linear behavior between training examples and provides soft labels

- **CutMix:** pastes a rectangular region from one image onto another, mixing labels proportional to area. Both Mixup and CutMix improve generalization and reduce overconfidence

## Training vs Evaluation Transforms

- Training transforms include random augmentations to improve generalization
- Evaluation transforms are deterministic: resize and crop without any randomness, ensuring reproducible results
- Normalization is the same in both
- The difference is entirely in spatial preprocessing: random for training, deterministic for evaluation
- Forgetting to switch is a common bug:
  - Training transforms during evaluation introduce noise into metrics
  - Evaluation transforms during training miss the regularization benefit of augmentation

## Unnormalization: Reversing the Transform

When visualizing model inputs or predictions, reverse the normalization:

$$
I_c = I_c^{\text{norm}} \cdot \sigma_c + \mu_c
$$

- Clip values to $[0, 1]$ afterward (augmentations can push values slightly outside this range)
- Convert to uint8 by multiplying by 255
- Forgetting to unnormalize produces images with washed-out or inverted colors

## Performance Considerations

- Transform pipelines run on CPU by default, as part of data loading
- For large-scale training, CPU-bound transforms can become a bottleneck
- Strategies:
  - Torchvision v2 transforms support tensor inputs natively; some operations can be moved to GPU
  - NVIDIA DALI provides GPU-accelerated data loading and augmentation pipelines
  - Increase DataLoader num_workers to prepare batches in parallel, hiding transform latency behind GPU computation
- Interpolation modes affect both quality and speed:
  - Bilinear: default, good balance of quality and performance
  - Bicubic: higher quality but slower
  - Nearest-neighbor: fastest but produces blocky artifacts; required for segmentation masks to avoid creating invalid label values
