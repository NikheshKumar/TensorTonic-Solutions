## <span style="font-size: 18px;">Residual Blocks and Skip Connections</span>

<span style="font-size: 14px;">Residual blocks are the core building blocks of ResNets (Residual Networks), introduced by He et al. in 2015. They address the degradation problem: as neural networks get deeper, training accuracy can saturate and then degrade, not because of overfitting, but because deeper networks become harder to optimize.</span>

### <span style="font-size: 16px;">The Core Idea</span>

<span style="font-size: 14px;">Instead of learning a direct mapping $H(x)$, a residual block learns the residual function:</span>

$$
F(x) = H(x) - x
$$

<span style="font-size: 14px;">The output is then:</span>

$$
y = F(x) + x
$$

<span style="font-size: 14px;">This is implemented by adding a skip connection (also called a shortcut connection) that bypasses the convolutional layers and adds the input directly to the output.</span>

### <span style="font-size: 16px;">Why Residuals Help</span>

- <span style="font-size: 14px;">If the optimal transformation is close to identity, the network only needs to learn small residual adjustments $F(x) \approx 0$, which is easier than learning the full mapping</span>
- <span style="font-size: 14px;">Gradients flow directly through the skip connection during backpropagation, mitigating the vanishing gradient problem</span>
- <span style="font-size: 14px;">The identity path ensures that adding more layers never hurts performance: in the worst case, extra layers learn $F(x) = 0$</span>

### <span style="font-size: 16px;">Basic Block Architecture</span>

<span style="font-size: 14px;">A standard residual block with two convolutional layers follows this structure:</span>

- <span style="font-size: 14px;">$\text{out} = \text{Conv}_1(x)$: first 3x3 convolution</span>
- <span style="font-size: 14px;">$\text{out} = \text{ReLU}(\text{BN}_1(\text{out}))$: batch normalization then activation</span>
- <span style="font-size: 14px;">$\text{out} = \text{Conv}_2(\text{out})$: second 3x3 convolution</span>
- <span style="font-size: 14px;">$\text{out} = \text{BN}_2(\text{out})$: batch normalization (no activation yet)</span>
- <span style="font-size: 14px;">$y = \text{ReLU}(\text{out} + x)$: add skip connection, then activate</span>

### <span style="font-size: 16px;">Batch Normalization</span>

<span style="font-size: 14px;">Each convolution is followed by batch normalization, which normalizes activations across the batch dimension:</span>

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

<span style="font-size: 14px;">where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, and $\gamma$, $\beta$ are learnable parameters. This stabilizes training by reducing internal covariate shift.</span>

### <span style="font-size: 16px;">When Dimensions Match</span>

<span style="font-size: 14px;">The skip connection $x + F(x)$ requires that $x$ and $F(x)$ have the same shape. This holds when:</span>

- <span style="font-size: 14px;">Input and output channels are the same</span>
- <span style="font-size: 14px;">Spatial dimensions are preserved (using padding=1 with 3x3 kernels)</span>

<span style="font-size: 14px;">When dimensions do not match (e.g., downsampling or changing channels), a 1x1 convolution projection is used on the skip path to align shapes. This problem focuses on the simpler case where no projection is needed.</span>