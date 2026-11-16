# Spatial Self-Attention Block

A convolution looks at each pixel together with its close neighbours, so a pixel in one corner of an image learns about the opposite corner only after many layers. Self-attention gives every pixel a direct line to every other pixel, much like a group chat where everyone reads every message, compared with passing a note along a row of people. Diffusion models need this to keep a whole picture consistent, so that two eyes match and a horizon stays straight. This page builds the block from one query looking at a few keys, up to several heads working across every pixel of a feature map, with the result added back to the input as you implement here.

## Why a denoiser needs global context

Denoising a heavily corrupted image requires consistent decisions across the whole picture. Two eyes of a face should match, a horizon should stay straight and a texture should agree from one side to the other. Convolutions at low resolution see more of the image, but their information still travels through a chain of local steps.

Self-attention gives every position a direct path to every other position in a single layer. Ho et al. (2020) placed such blocks at the 16x16 resolution of their U-Net, and later models such as the ablated diffusion model of Dhariwal and Nichol (2021) added them at several resolutions with several heads. The cost grows with the square of the number of pixels, so attention is used only where the feature maps are small.

## Pixels become tokens

Attention works on a list of vectors, so the feature map has to become one. A feature map of shape (B, C, H, W) holds a C-dimensional feature vector at each of its H times W pixels. Flattening the two spatial axes in row-major order and moving the channels last gives a tensor of shape (B, N, C) with $N = HW$, where pixel $(i, j)$ becomes token $iW + j$.

On a map with 2 rows and 3 columns, pixel $(0, 2)$ becomes token 2, pixel $(1, 0)$ becomes token 3 and pixel $(1, 2)$ becomes token 5. Each token is the column of C channel values at that pixel. After attention, the reverse step puts token $iW + j$ back at pixel $(i, j)$ and moves the channels back in front.

## One query attends to a few keys

Start with the smallest version. One pixel holds a query, a single number $q$ that says what it is looking for. Every pixel, including itself, holds a key $k$ that says what it offers and a value $v$ that is the content it passes on. The match between the query and each key is their product, and a softmax turns the matches into weights that are non-negative and sum to 1.

Take $q = 2$, three keys $1$, $-1$ and $0.5$, and values $4$, $0$ and $2$.

1. The scores are $2 \cdot 1 = 2$, $2 \cdot (-1) = -2$ and $2 \cdot 0.5 = 1$.
2. The softmax weights are about $0.7214$, $0.0132$ and $0.2654$.
3. The output is $0.7214 \cdot 4 + 0.0132 \cdot 0 + 0.2654 \cdot 2 \approx 3.4164$.

The query pulls mostly from the first pixel, whose key matches it best, and barely at all from the second. The output is a weighted average of the values, so it always stays inside their range.

## Vectors, matrices and the scale factor

With vectors of size $d$ the match becomes a dot product. Stack all $N$ queries into a matrix $Q$ of shape (N, d), and the keys and values into $K$ and $V$ of the same shape. One matrix product $Q K^{\top}$ then gives all $N \times N$ scores at once, row $m$ holding the scores of query $m$ against every key, and

$$
O = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d}}\right) V .
$$

The softmax runs along each row, over the key axis, so each query gets its own weights over all keys. Row $m$ of $O$ is the weighted average of the value rows for query $m$.

The division by $\sqrt{d}$ comes from Vaswani et al. (2017). If the entries of a query and a key are independent with unit variance, their dot product has variance $d$. With $d = 64$, random scores have a standard deviation of 8, and the largest of 16 softmax weights averages about 0.85, so attention is almost one-hot and its gradients nearly vanish. After dividing by 8, the largest weight averages about 0.25, and the distribution stays soft enough to learn.

## A two-token example

Take one head with $d = 2$ and two tokens whose queries and keys are both $(1, 0)$ and $(0, 2)$, with values $(1, -1)$ and $(3, 1)$.

1. The raw scores $Q K^{\top}$ have rows $(1, 0)$ and $(0, 4)$.
2. Dividing by $\sqrt 2$ gives rows $(0.7071, 0)$ and $(0, 2.8284)$.
3. The softmax rows are about $(0.6698, 0.3302)$ and $(0.0558, 0.9442)$.
4. The first output is $0.6698 \cdot (1, -1) + 0.3302 \cdot (3, 1) \approx (1.6605, -0.3395)$.
5. The second output is $0.0558 \cdot (1, -1) + 0.9442 \cdot (3, 1) \approx (2.8884, 0.8884)$.

The second token is dominated by its own value because its self-score is large, while the first token mixes the two.

## Queries, keys and values come from learned projections

In the block the queries, keys and values are learned from the pixel features. First the feature map passes through a group norm with num_groups groups, so the projections see well scaled inputs, and the result is flattened into tokens $Z$ of shape (N, C) per sample. Three linear maps then give

$$
Q = Z W_q^{\top} + b_q, \qquad K = Z W_k^{\top} + b_k, \qquad V = Z W_v^{\top} + b_v ,
$$

where $W_q$, $W_k$ and $W_v$ are q_weight, k_weight and v_weight of shape (C, C), and $b_q$, $b_k$ and $b_v$ are q_bias, k_bias and v_bias of shape (C,). The weights follow the linear-layer layout, so each token row is multiplied by the transpose of the weight. Because the same weights are used at every pixel, the block works for any image size.

## Several heads look for different things

One attention pattern per pixel is limiting, since a pixel may want to match neighbours of similar colour and, separately, distant pixels of similar shape. Multi-head attention splits the C channels of $Q$, $K$ and $V$ into $n$ contiguous blocks of $d = C / n$ channels, where $n$ is num_heads. Head $j$ uses channels $jd$ to $(j + 1)d - 1$ and runs its own attention,

$$
O_j = \mathrm{softmax}\!\left(\frac{Q_j K_j^{\top}}{\sqrt{d}}\right) V_j .
$$

The head outputs are concatenated back along the channel axis in head order, giving shape (N, C), and the output projection with out_weight and out_bias mixes information across heads. The scale uses the per-head size, so with 8 channels and 4 heads the divisor is $\sqrt 2$, while $\sqrt 8$ would make every head too flat.

## Back to image layout and the residual

After the output projection the tokens are reshaped back to (B, C, H, W) with the inverse of the flattening. A plain reshape of the (B, N, C) tensor straight to (B, C, H, W) skips the channel move and scrambles channels and pixels together, even though the shape looks right.

The block output is then added to the original input $x$, the tensor from before the group norm. As in residual blocks, this lets the attention branch learn a correction and gives gradients a direct path. Real implementations often initialise the output projection to zero, so a freshly built block starts as the identity.

## Edge cases and numerical care

A 1x1 feature map has a single token, so every softmax weight is exactly 1 and the block reduces to a chain of projections added to $x$. A softmax should subtract the row maximum before exponentiating, which torch.softmax does internally, so large scores do not overflow. When num_heads is 1 the head split changes nothing, and when $d = 1$ each head works with scalar queries and keys, exactly like the first example on this page.

## Where this block sits in the U-Net

In a diffusion U-Net, attention blocks follow residual blocks at the lowest resolutions, in the encoder, at the bottom and in the decoder. They do not read the timestep directly, because the residual block before them has already mixed the time embedding into the features.

## Common mistakes to avoid

- **Scaling by the square root of C instead of the head size.** The scores become too small by a factor of the square root of the number of heads, which flattens every attention distribution.
- **Interleaving channels across heads.** Viewing the projection as (N, d, heads) gives each head every n-th channel, which pairs the wrong query and key entries.
- **Running the softmax over the query axis.** Each key then spreads weight over queries, and the outputs are no longer weighted averages of the values.
- **Using the weights without transposing.** A linear layer computes $z W^{\top}$, and multiplying by $W$ applies the transposed map.
- **Reshaping tokens to an image without moving channels back.** The result has the right shape but mixes pixels and channels in a meaningless way.
- **Adding the output to the normalized features.** The residual must carry the original x, otherwise the block replaces its input with a normalized copy.