## What Max Pooling Does

Max pooling is a **downsampling** operation that reduces the spatial dimensions of feature maps while retaining the most important information. It slides a window over the input and outputs the **maximum value** within each window.

**Input:** Feature map of shape $(C, H_{in}, W_{in})$

**Output:** Smaller feature map of shape $(C, H_{out}, W_{out})$

The number of channels stays the same. Only height and width are reduced.

---

## Why Use Max Pooling?

**1. Dimensionality reduction**

Reduces spatial size, which reduces:
- Computation in subsequent layers
- Memory requirements
- Number of parameters (if followed by FC layer)

**2. Translation invariance**

Small shifts in the input do not change the output much. If a feature moves slightly, the max in the window is often the same.

**3. Feature selection**

Keeps the strongest activation (maximum) in each region. Weak or negative activations are discarded.

**4. Increased receptive field**

Each output neuron effectively "sees" a larger portion of the original input after pooling.

---

## The Max Pooling Operation

For each spatial position in the output, look at a window (pool) of the input and take the maximum:

$$
\text{output}[c, i, j] = \max_{m, n} \text{input}[c, i \cdot s + m, j \cdot s + n]
$$

where:
- $c$ is the channel (pooling is done independently per channel)
- $s$ is the stride
- $m \in [0, k_h)$ and $n \in [0, k_w)$ iterate over the pooling window
- $k_h, k_w$ are the pool height and width

---

## Output Size Calculation

For input size $H_{in}$, pool size $k$, padding $p$, and stride $s$:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

Same formula for width.

**Most common configuration:** $k = 2$, $s = 2$, $p = 0$

This halves both height and width: $H_{out} = H_{in} / 2$

---

## Step-by-Step Example: 2x2 Max Pool

**Input:** 4x4 feature map (single channel)

$$
\begin{bmatrix}
1 & 3 & 2 & 4 \\
5 & 6 & 7 & 8 \\
3 & 2 & 1 & 0 \\
1 & 2 & 3 & 4
\end{bmatrix}
$$

**Pool size:** 2x2, **Stride:** 2, **Padding:** 0

**Output size:** $(4 - 2)/2 + 1 = 2$, so 2x2 output

**Step 1: Top-left window (rows 0-1, cols 0-1)**

Window:
$$
\begin{bmatrix}
1 & 3 \\
5 & 6
\end{bmatrix}
$$

$\max(1, 3, 5, 6) = 6$

Output[0,0] = 6

**Step 2: Top-right window (rows 0-1, cols 2-3)**

Window:
$$
\begin{bmatrix}
2 & 4 \\
7 & 8
\end{bmatrix}
$$

$\max(2, 4, 7, 8) = 8$

Output[0,1] = 8

**Step 3: Bottom-left window (rows 2-3, cols 0-1)**

Window:
$$
\begin{bmatrix}
3 & 2 \\
1 & 2
\end{bmatrix}
$$

$\max(3, 2, 1, 2) = 3$

Output[1,0] = 3

**Step 4: Bottom-right window (rows 2-3, cols 2-3)**

Window:
$$
\begin{bmatrix}
1 & 0 \\
3 & 4
\end{bmatrix}
$$

$\max(1, 0, 3, 4) = 4$

Output[1,1] = 4

**Final output:**

$$
\begin{bmatrix}
6 & 8 \\
3 & 4
\end{bmatrix}
$$

---

## Detailed Example: 3x3 Max Pool with Stride 1

**Input:** 5x5 feature map

$$
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20 \\
21 & 22 & 23 & 24 & 25
\end{bmatrix}
$$

**Pool size:** 3x3, **Stride:** 1, **Padding:** 0

**Output size:** $(5 - 3)/1 + 1 = 3$, so 3x3 output

**Position (0,0):** Window covers rows 0-2, cols 0-2

$$
\begin{bmatrix}
1 & 2 & 3 \\
6 & 7 & 8 \\
11 & 12 & 13
\end{bmatrix}
$$

$\max = 13$

**Position (0,1):** Window covers rows 0-2, cols 1-3

$$
\begin{bmatrix}
2 & 3 & 4 \\
7 & 8 & 9 \\
12 & 13 & 14
\end{bmatrix}
$$

$\max = 14$

**Position (0,2):** Window covers rows 0-2, cols 2-4

$\max = 15$

**Continuing this pattern, the full output:**

$$
\begin{bmatrix}
13 & 14 & 15 \\
18 & 19 & 20 \\
23 & 24 & 25
\end{bmatrix}
$$

Notice how the max values "propagate" from the bottom-right region where values are highest.

---

## Multi-Channel Example

**Input:** 4x4 with 2 channels

Channel 0:
$$
\begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$

Channel 1:
$$
\begin{bmatrix}
16 & 15 & 14 & 13 \\
12 & 11 & 10 & 9 \\
8 & 7 & 6 & 5 \\
4 & 3 & 2 & 1
\end{bmatrix}
$$

**2x2 max pool, stride 2:**

Channel 0 output:
$$
\begin{bmatrix}
6 & 8 \\
14 & 16
\end{bmatrix}
$$

Channel 1 output:
$$
\begin{bmatrix}
16 & 14 \\
8 & 6
\end{bmatrix}
$$

Each channel is pooled independently. Channels do not interact during pooling.

---

## Pooling with Padding

**Input:** 5x5, **Pool:** 2x2, **Stride:** 2, **Padding:** 0

Without padding: $(5-2)/2 + 1 = 2.5 \rightarrow 2$ (floor)

The rightmost column and bottom row are not fully covered.

**With padding = 1:** Input effectively becomes 7x7 (with zeros on edges)

$(7-2)/2 + 1 = 3$

Padding ensures all input elements are covered, but edge values include padded zeros.

---

## Overlapping vs. Non-Overlapping Pooling

**Non-overlapping (stride = pool size):**
- Most common: 2x2 pool with stride 2
- Each input element contributes to exactly one output
- Clean 2x or 4x reduction

**Overlapping (stride < pool size):**
- 3x3 pool with stride 2
- Input elements can contribute to multiple outputs
- Slight improvement in accuracy (used in AlexNet)
- Less common today

---

## Max Pooling vs. Average Pooling

**Max pooling:**
- Takes maximum value in window
- Preserves strongest activations
- More common in classification networks
- Introduces sparsity (only max matters)

**Average pooling:**
- Takes mean of values in window
- Smooths activations
- Common for global pooling before classifier
- All values contribute equally

**When to use which:**
- Max pooling: general feature extraction, most CNN architectures
- Average pooling: global pooling, some specific architectures

---

## The Gradient During Backpropagation

During the forward pass, track which position had the maximum (the "argmax").

During backpropagation:
- The gradient flows only to the position that had the maximum
- All other positions receive zero gradient

$$
\frac{\partial L}{\partial \text{input}[c, i, j]} = \begin{cases}
\frac{\partial L}{\partial \text{output}} & \text{if } (i,j) \text{ was the argmax} \\
0 & \text{otherwise}
\end{cases}
$$

This is called a "max routing" gradient. Information about which position was max must be stored during forward pass for use in backward pass.

---

## Position in CNN Architecture

Typical CNN structure:

Conv -> ReLU -> **MaxPool** -> Conv -> ReLU -> **MaxPool** -> ... -> FC

Max pooling usually follows conv + activation blocks to progressively reduce spatial dimensions.

**Modern trends:**
- Some architectures replace pooling with strided convolutions
- ResNet uses pooling sparingly (one at start, one at end)
- Still widely used in VGG, AlexNet-style networks

---

## Common Configurations

**Standard 2x2 pooling:**
- Pool size: 2x2
- Stride: 2
- Padding: 0
- Effect: Halves both dimensions

**Overlapping pooling (AlexNet):**
- Pool size: 3x3
- Stride: 2
- Slight accuracy improvement

**Global max pooling:**
- Pool size: entire spatial dimension
- Output: one value per channel
- Used before classifier in some networks

---

## Implementation Algorithm

**For each output position (c, i, j):**

1. Compute the input region: rows from $i \cdot s$ to $i \cdot s + k_h$, cols from $j \cdot s$ to $j \cdot s + k_w$

2. Extract the window from input channel $c$

3. Find the maximum value in the window

4. Store the maximum as output[c, i, j]

5. (For backprop) Store the position of the maximum

**Complexity:** $O(C \times H_{out} \times W_{out} \times k_h \times k_w)$

---

## Edge Cases and Considerations

**Input not divisible by stride:**
- Either use padding to make it divisible
- Or accept that edge pixels may be excluded

**Very small inputs:**
- If input is smaller than pool size, pooling is not possible (or use global pooling)

**Negative values:**
- Max pooling works fine with negatives
- The max of [-5, -3, -7, -2] is -2

**All same values:**
- Max is that value
- For gradient, typically the first occurrence is used as argmax