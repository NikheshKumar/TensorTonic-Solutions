# <span style="font-size: 20px;">Zero Pad and Center Crop</span>

<span style="font-size: 14px;">Zero padding and center cropping are two of the most common spatial preprocessing operations in computer vision. **Padding** adds a border of zeros around an image to enlarge it; **center cropping** extracts a fixed-size window from the middle. Combined, they reshape an image to a target resolution without resampling, which is the standard way training pipelines force every image to a uniform size while keeping the subject centered.</span>

---

## <span style="font-size: 16px;">What Each Operation Does</span>

<span style="font-size: 14px;">**Zero padding** surrounds the image with rows and columns of zeros, leaving the original pixels untouched. Padding by `pad` on every side of an $(H, W)$ image grows it to $(H + 2\cdot\texttt{pad},\ W + 2\cdot\texttt{pad})$: `pad` new rows on top, `pad` on the bottom, `pad` new columns on the left, and `pad` on the right. The original content sits unchanged in the interior, offset from the top-left corner by exactly `pad` in each axis.</span>

<span style="font-size: 14px;">**Center cropping** does the opposite: it discards the outer border and keeps a centered rectangle of size $(\texttt{crop\_h}, \texttt{crop\_w})$. If the crop is smaller than the input, the operation trims symmetrically from all sides as evenly as the parity allows. If the crop is larger than the input, the crop window extends into the padded zero region (or, with negative start indices, would index outside the array - a case the index math must handle carefully).</span>

<span style="font-size: 14px;">The two are frequently chained: pad first to guarantee enough margin, then crop to the exact target shape. This is exactly the pattern in `torchvision.transforms.Pad` followed by `CenterCrop`, and it lets a single fixed output size accommodate inputs both smaller and larger than the target.</span>

---

## <span style="font-size: 16px;">The Crop Index Math</span>

<span style="font-size: 14px;">Let the padded image have dimensions $H_p = H + 2\cdot\texttt{pad}$ and $W_p = W + 2\cdot\texttt{pad}$. To center a crop of size $(\texttt{crop\_h}, \texttt{crop\_w})$, the leftover margin must be split as evenly as possible between the two sides. The crop's top-left corner is:</span>

$$
r_{\text{start}} = \left\lfloor \frac{H_p - \texttt{crop\_h}}{2} \right\rfloor, \qquad c_{\text{start}} = \left\lfloor \frac{W_p - \texttt{crop\_w}}{2} \right\rfloor
$$

<span style="font-size: 14px;">Substituting $H_p = H + 2\cdot\texttt{pad}$ gives the form in the problem statement:</span>

$$
r_{\text{start}} = \left\lfloor \frac{H + 2\cdot\texttt{pad} - \texttt{crop\_h}}{2} \right\rfloor, \qquad c_{\text{start}} = \left\lfloor \frac{W + 2\cdot\texttt{pad} - \texttt{crop\_w}}{2} \right\rfloor
$$

<span style="font-size: 14px;">The output then takes rows $r_{\text{start}}$ through $r_{\text{start}} + \texttt{crop\_h} - 1$ and columns $c_{\text{start}}$ through $c_{\text{start}} + \texttt{crop\_w} - 1$ of the padded image. In NumPy this is the slice `padded[r_start : r_start + crop_h, c_start : c_start + crop_w]`.</span>

---

## <span style="font-size: 16px;">Why Floor Division and Where the Extra Pixel Goes</span>

<span style="font-size: 14px;">The total leftover margin along the height is $H_p - \texttt{crop\_h}$. When this is **even**, it splits perfectly: equal margins above and below. When it is **odd**, there is one extra pixel that cannot be shared, and a convention must decide which side gets it.</span>

<span style="font-size: 14px;">Using Python's floor-division operator `//` (equivalently $\lfloor \cdot \rfloor$) puts the smaller margin on top and left, which pushes the extra pixel to the **bottom and right**. For example, if the leftover is $3$, then $\lfloor 3/2 \rfloor = 1$ pixel of margin goes on top and the remaining $2$ go on the bottom. This is the convention PyTorch's `CenterCrop` uses, and matching it exactly is what the problem requires. Rounding the other way (ceiling, or rounding to nearest) would shift the crop by one pixel and produce a different, wrong output.</span>

---

## <span style="font-size: 16px;">Padding Modes and Why Zeros</span>

<span style="font-size: 14px;">Zero padding is one of several ways to fabricate border pixels, and the choice has real consequences:</span>

* <span style="font-size: 14px;">**Zero (constant) padding** fills the border with $0$. It is the simplest and the default for convolution. The downside is that it introduces an artificial dark edge that can bias gradients and create visible halos near borders, since real images rarely fade to pure black at their boundaries.</span>
* <span style="font-size: 14px;">**Reflection padding** mirrors the image across its edge, so the border continues the existing texture without a hard discontinuity. Common for style transfer and super-resolution where edge artifacts are objectionable.</span>
* <span style="font-size: 14px;">**Replication (edge) padding** repeats the outermost pixel value outward, a cheaper smoothing of the boundary than reflection.</span>

<span style="font-size: 14px;">This problem specifies zero padding, which matches the convolution convention and keeps the border content semantically "empty". Because the padded values are exactly $0$, they are easy to recognize and to mask out later if needed, unlike reflected or replicated values that blend with real content.</span>

---

## <span style="font-size: 16px;">Worked Example (pad then crop)</span>

<span style="font-size: 14px;">Start with a $2 \times 2$ image $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, with $\texttt{pad} = 1$, $\texttt{crop\_h} = 3$, $\texttt{crop\_w} = 3$.</span>

<span style="font-size: 14px;">1. **Pad by 1 on every side.** The padded image is $4 \times 4$ ($H_p = 2 + 2 = 4$):</span>

$$
\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 2 & 0 \\ 0 & 3 & 4 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix}
$$

<span style="font-size: 14px;">2. **Compute crop start.** $r_{\text{start}} = \lfloor (4 - 3)/2 \rfloor = \lfloor 0.5 \rfloor = 0$, and likewise $c_{\text{start}} = 0$. The single leftover pixel of margin lands on the bottom and right.</span>

<span style="font-size: 14px;">3. **Slice rows 0..2, columns 0..2.** The output $3 \times 3$ crop is:</span>

$$
\begin{pmatrix} 0 & 0 & 0 \\ 0 & 1 & 2 \\ 0 & 3 & 4 \end{pmatrix}
$$

<span style="font-size: 14px;">Notice the crop is not perfectly symmetric: the top and left each have one zero border, but the bottom and right rows/columns of the padded image (all zeros) are dropped. The asymmetry is the direct consequence of the floor-division convention with an odd leftover.</span>

---

## <span style="font-size: 16px;">A Second Example (pure crop, no padding)</span>

<span style="font-size: 14px;">Take a $4 \times 4$ image with $\texttt{pad} = 0$ and a $2 \times 2$ center crop. The leftover along each axis is $4 - 2 = 2$, which is even.</span>

<span style="font-size: 14px;">1. **Crop start**: $r_{\text{start}} = \lfloor 2/2 \rfloor = 1$, $c_{\text{start}} = 1$.</span>

<span style="font-size: 14px;">2. **Slice rows 1..2, columns 1..2**: this extracts the central $2 \times 2$ block, dropping one row and column from every side.</span>

<span style="font-size: 14px;">With an even leftover the crop is perfectly centered, so the choice of rounding never matters. The rounding convention only becomes visible when the leftover is odd, as in the first example. A useful mental shortcut: the leftover is even exactly when $H_p$ and $\texttt{crop\_h}$ have the same parity, meaning both even or both odd, and the leftover is odd otherwise. Knowing the parity ahead of time tells you immediately whether the floor convention will shift the crop off perfect center.</span>

---

## <span style="font-size: 16px;">A Third Example (odd leftover, larger crop)</span>

<span style="font-size: 14px;">To see the floor convention with a clearly odd leftover, take a $3 \times 3$ image, $\texttt{pad} = 0$, and a $2 \times 2$ crop. The leftover is $3 - 2 = 1$, odd.</span>

<span style="font-size: 14px;">1. **Crop start**: $r_{\text{start}} = \lfloor 1/2 \rfloor = 0$, $c_{\text{start}} = 0$.</span>

<span style="font-size: 14px;">2. **Slice rows 0..1, columns 0..1**: this keeps the top-left $2 \times 2$ corner. The single dropped row (the last) and dropped column (the last) come off the bottom and right, exactly as the convention dictates.</span>

<span style="font-size: 14px;">Had the rounding gone the other way, the crop would have kept the bottom-right corner instead. For a centered subject, the one-pixel difference is usually invisible, but for pixel-exact tasks such as aligning a crop to a known annotation it changes the answer.</span>

---

## <span style="font-size: 16px;">Why This Pattern Is Everywhere</span>

<span style="font-size: 14px;">Networks require fixed-size inputs, but real images arrive at arbitrary resolutions. There are two ways to standardize size: **resize** (interpolate to the target, changing pixel values and aspect ratio) or **pad and crop** (rearrange existing pixels and zero borders, never resampling). Pad-and-crop is preferred when preserving the original pixels and aspect ratio matters, for example in detection and segmentation where resampling would distort object boundaries and shift annotation coordinates by sub-pixel amounts that accumulate into measurable localization error.</span>

<span style="font-size: 14px;">The standard ImageNet evaluation pipeline resizes the short side to 256 and then takes a $224 \times 224$ **center crop**, ensuring the network sees the most informative central region at a fixed resolution. Center crop is also the deterministic test-time counterpart of the random crop used during training: training samples random offsets for augmentation, while evaluation uses the fixed center for reproducibility.</span>

<span style="font-size: 14px;">Zero padding specifically (rather than reflection or edge padding) is used when the border content should be treated as "absent". It is the same padding that convolution layers apply to preserve spatial dimensions, which is why understanding the index math here transfers directly to reasoning about convolution output shapes.</span>

---

## <span style="font-size: 16px;">Output Shape and Boundary Cases</span>

<span style="font-size: 14px;">The output shape is always exactly $(\texttt{crop\_h}, \texttt{crop\_w})$ by construction, independent of the input size, which is the whole point of the operation. Several boundary cases deserve attention:</span>

* <span style="font-size: 14px;">**Crop larger than padded image.** If $\texttt{crop\_h} > H_p$, the formula gives a negative $r_{\text{start}}$. A naive slice with a negative start would index from the end of the array and silently return wrong pixels; a correct implementation must either forbid this or pad further so the crop fits.</span>
* <span style="font-size: 14px;">**Crop equal to padded image.** Then $r_{\text{start}} = 0$ and the crop is the whole padded image unchanged.</span>
* <span style="font-size: 14px;">**Zero padding amount.** With $\texttt{pad} = 0$ the operation reduces to a pure center crop of the original image.</span>

---

## <span style="font-size: 16px;">Relationship to Convolution Padding</span>

<span style="font-size: 14px;">The same zero padding studied here is what convolution layers apply to control output size. A convolution with kernel size $k$, stride $s$, and padding $p$ on an input of size $H$ produces output of size $\lfloor (H + 2p - k)/s \rfloor + 1$. The $H + 2p$ term is identical to the padded dimension $H_p$ computed above, and the same floor operation appears. Padding $p = (k-1)/2$ with stride $1$ yields "same" convolution, where output size equals input size, precisely because the zeros restore the rows and columns that the kernel would otherwise consume at the borders.</span>

<span style="font-size: 14px;">This is why mastering the pad-and-crop index arithmetic pays off broadly: the floor-division reasoning, the role of $2p$, and the off-by-one risks are the same exact skills needed to predict convolution and pooling output shapes throughout a network.</span>

---

## <span style="font-size: 16px;">Complexity</span>

<span style="font-size: 14px;">Both operations are pure memory rearrangement. Padding allocates an $(H_p, W_p)$ array of zeros and copies the original into the interior, costing $O(H_p \cdot W_p)$. Cropping copies a $(\texttt{crop\_h}, \texttt{crop\_w})$ slice, costing $O(\texttt{crop\_h} \cdot \texttt{crop\_w})$. There is no arithmetic on pixel values beyond copying, so the operation is fast and exact; the only "computation" is the index arithmetic that locates the crop window.</span>

<span style="font-size: 14px;">In a fused implementation the explicit padded array can be skipped entirely. Since cropping only reads a window, the padded coordinates can be mapped directly back to original coordinates, returning $0$ wherever the mapped index falls outside the original image and the original pixel otherwise. This avoids materializing the full padded buffer and is how efficient libraries implement pad-plus-crop as a single pass, but the index logic must still match the floor-division convention exactly.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Off-by-one from the wrong rounding direction.** Using ceiling division or rounding-to-nearest instead of floor shifts the crop by one pixel when the leftover margin is odd. The result looks almost right but is offset, which silently corrupts alignment-sensitive tasks. Always use `//` so the extra pixel falls bottom-right.</span>
* <span style="font-size: 14px;">**Forgetting to add the padding into the size formula.** The crop start must be computed against the **padded** dimensions $H + 2\cdot\texttt{pad}$, not the original $H$. Omitting the $2\cdot\texttt{pad}$ term centers the crop on the wrong image and drops or includes the wrong rows.</span>
* <span style="font-size: 14px;">**Negative start indices from an oversized crop.** When the crop exceeds the padded image, $r_{\text{start}}$ goes negative and Python slicing wraps to the array's end, returning garbage rather than erroring. Validate that the crop fits, or pad enough that it does.</span>
* <span style="font-size: 14px;">**Padding only one side or padding asymmetrically.** The spec pads `pad` on every side. Padding only top-left, or confusing total padding with per-side padding, changes $H_p$ and therefore every downstream index. Each side gets exactly `pad` rows or columns.</span>
* <span style="font-size: 14px;">**Mismatched dtype between zeros and image.** Allocating an integer zero array and copying float pixels (or vice versa) can truncate values or upcast unexpectedly, and the rounded output then disagrees. Allocate the padded buffer in the same dtype as the image, and round only at the end.</span>

---