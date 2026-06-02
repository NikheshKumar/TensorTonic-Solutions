# <span style="font-size: 20px;">RGB to Grayscale</span>

<span style="font-size: 14px;">Grayscale conversion collapses the three colour channels of an RGB image into a single **luminance** (intensity) channel. It is the first preprocessing step for a large fraction of classical computer vision: edge detectors, corner detectors, thresholding, template matching, and hand-crafted descriptors like SIFT and HOG all operate on a single-channel intensity map rather than on colour.</span>

---

## <span style="font-size: 16px;">Why Convert to Grayscale</span>

<span style="font-size: 14px;">Colour carries information, but for many tasks it is redundant or even harmful. Reducing three channels to one has concrete benefits:</span>

* <span style="font-size: 14px;">**Less data, faster compute.** A single channel uses one third of the memory and one third of the arithmetic of a 3-channel image. A 1080p frame drops from about 6.2M values to 2.1M, which matters for real-time pipelines and embedded vision.</span>
* <span style="font-size: 14px;">**Most structure lives in intensity.** Edges, gradients, and texture are largely defined by changes in brightness, not hue. The classical operators (Sobel, Canny, Harris) were designed to act on a scalar field, and their gradient definitions assume a single intensity value per pixel.</span>
* <span style="font-size: 14px;">**Robustness to colour shifts.** White balance, illumination colour, and sensor differences perturb the channel ratios. Working in intensity removes one axis of nuisance variation, so a detector trained on grayscale generalizes across cameras more easily.</span>
* <span style="font-size: 14px;">**Algorithmic simplicity.** Many algorithms (Otsu thresholding, connected components, distance transforms) are defined only for scalar fields. Grayscale is the natural input domain.</span>

---

## <span style="font-size: 16px;">Luminance Is Not a Plain Average</span>

<span style="font-size: 14px;">The naive conversion is to average the channels: $Y = (R + G + B)/3$, giving each colour a weight of $0.333$. This is wrong perceptually. The human visual system is not equally sensitive to red, green, and blue. The retina has far more cones tuned to medium (green) wavelengths than to red or blue, so a green patch of a given physical intensity looks much brighter than a blue patch of the same intensity.</span>

<span style="font-size: 14px;">To produce a grayscale image whose brightness matches what a human sees, the channels must be weighted by perceptual sensitivity. This weighted intensity is called **luma**, denoted $Y$ (or $Y'$ when computed from gamma-corrected signals). A pure green pixel $[0, 255, 0]$ should map to a far brighter gray than a pure blue pixel $[0, 0, 255]$, even though both have a single channel at full strength.</span>

---

## <span style="font-size: 16px;">The Rec. 601 Formula</span>

<span style="font-size: 14px;">The ITU-R BT.601 standard, developed for standard-definition television, defines luma as a fixed weighted sum of the red, green, and blue channels:</span>

$$
Y(i, j) = 0.299 \cdot R(i, j) + 0.587 \cdot G(i, j) + 0.114 \cdot B(i, j)
$$

<span style="font-size: 14px;">where:</span>

* <span style="font-size: 14px;">$R, G, B$ are the per-pixel channel values at row $i$, column $j$</span>
* <span style="font-size: 14px;">the three weights sum to exactly $1$: $0.299 + 0.587 + 0.114 = 1.000$, so a neutral gray pixel with $R = G = B = v$ maps to $Y = v$ (no brightening or darkening)</span>
* <span style="font-size: 14px;">$Y$ has the same numeric range as the inputs: if $R, G, B \in [0, 255]$ then $Y \in [0, 255]$; if normalized to $[0, 1]$, $Y$ stays in $[0, 1]$</span>

<span style="font-size: 14px;">Green carries roughly **59%** of the luma, red about **30%**, and blue only about **11%**. This is why a pure-green pixel looks bright while a pure-blue pixel of equal intensity looks dark. The weights are a perceptual fingerprint of the human eye baked into a single linear equation.</span>

---

## <span style="font-size: 16px;">Where the Weights Come From</span>

<span style="font-size: 14px;">The coefficients are not arbitrary. They derive from the NTSC colour primaries and the standard CIE luminous-efficiency function $V(\lambda)$, which peaks in the green region near 555 nm. BT.601 fixes the green weight highest because the eye's photopic sensitivity is maximal there. Blue is weighted lowest because blue cones (S-cones) are sparse, making up only about 2% of all cones, and contribute little to perceived brightness.</span>

<span style="font-size: 14px;">A common variant is **Rec. 709** (used for HDTV and the sRGB colour space), which uses $Y = 0.2126 R + 0.7152 G + 0.0722 B$. The weights differ because the display primaries differ, but the principle is identical: green dominates, blue is nearly negligible. There is also the simpler "luminosity" coefficient set $(0.21, 0.72, 0.07)$ used in some image editors. This problem uses the BT.601 weights $(0.299, 0.587, 0.114)$, the same convention as OpenCV's default `cvtColor(img, COLOR_RGB2GRAY)` and PIL's `convert("L")`.</span>

---

## <span style="font-size: 16px;">Luma vs True Luminance</span>

<span style="font-size: 14px;">A subtle but important distinction: the BT.601 formula computes **luma** $Y'$ from the gamma-encoded (nonlinear) channel values that are stored in typical 8-bit images, not the physically-linear **luminance** $Y$. True luminance is a weighted sum of linear-light values, which would require first undoing gamma (linearizing), summing, and re-encoding.</span>

<span style="font-size: 14px;">In practice most pipelines apply the weights directly to the stored gamma-corrected values, exactly as this problem specifies. This is technically an approximation, but it is the universal convention in OpenCV, PIL, and television hardware, and it is what virtually every classical CV tutorial means by "grayscale". The approximation is good enough that the perceptual ordering (green bright, blue dark) is preserved.</span>

---

## <span style="font-size: 16px;">Worked Example (one 3-channel pixel)</span>

<span style="font-size: 14px;">Take a single orange-ish pixel $[R, G, B] = [200, 120, 40]$ on the $0$ to $255$ scale.</span>

<span style="font-size: 14px;">1. **Red term**: $0.299 \times 200 = 59.80$</span>

<span style="font-size: 14px;">2. **Green term**: $0.587 \times 120 = 70.44$</span>

<span style="font-size: 14px;">3. **Blue term**: $0.114 \times 40 = 4.56$</span>

<span style="font-size: 14px;">4. **Sum**: $Y = 59.80 + 70.44 + 4.56 = 134.80$</span>

<span style="font-size: 14px;">Notice that green contributes $70.44$ of the total even though its raw value $(120)$ is smaller than red's $(200)$: the $0.587$ weight amplifies its influence. Blue, despite being present at $40$, adds only $4.56$.</span>

<span style="font-size: 14px;">Contrast with the naive average: $(200 + 120 + 40)/3 = 120$. The perceptual luma $134.80$ is meaningfully brighter, because the average under-weights the bright green channel and over-weights blue.</span>

<span style="font-size: 14px;">As a second check, take pure colours. Pure red $[255, 0, 0]$ gives $Y = 0.299 \times 255 = 76.2$. Pure green $[0, 255, 0]$ gives $Y = 0.587 \times 255 = 149.7$. Pure blue $[0, 0, 255]$ gives $Y = 0.114 \times 255 = 29.1$. Green is nearly twice as bright as red and over five times as bright as blue, matching how these colours appear to the eye.</span>

---

## <span style="font-size: 16px;">A Small Image Patch</span>

<span style="font-size: 14px;">Consider a $2 \times 2$ image where each pixel is a colour: top-left red $[255, 0, 0]$, top-right green $[0, 255, 0]$, bottom-left blue $[0, 0, 255]$, bottom-right white $[255, 255, 255]$. Applying the formula pixel by pixel:</span>

* <span style="font-size: 14px;">**Top-left (red):** $Y = 76.2$</span>
* <span style="font-size: 14px;">**Top-right (green):** $Y = 149.7$</span>
* <span style="font-size: 14px;">**Bottom-left (blue):** $Y = 29.1$</span>
* <span style="font-size: 14px;">**Bottom-right (white):** $Y = 0.299(255) + 0.587(255) + 0.114(255) = 255$ exactly, since the weights sum to one</span>

<span style="font-size: 14px;">The resulting grayscale patch is $\begin{pmatrix} 76.2 & 149.7 \\ 29.1 & 255 \end{pmatrix}$. White stays at full intensity, confirming the weights are normalized; the colours order themselves by perceptual brightness.</span>

---

## <span style="font-size: 16px;">Computing It Over the Whole Image</span>

<span style="font-size: 14px;">In vectorized form, with an image tensor $X$ of shape $(H, W, 3)$, the operation is a dot product along the channel axis with the weight vector $w = [0.299, 0.587, 0.114]$:</span>

$$
Y = X \cdot w, \qquad Y \in \mathbb{R}^{H \times W}
$$

<span style="font-size: 14px;">In NumPy this is `Y = X @ w` or `np.tensordot(X, w, axes=([2],[0]))`, producing an $(H, W)$ array in a single fused operation. There is no spatial mixing: each output pixel depends only on the three channel values at the same location, so the cost is $O(H \cdot W)$ multiply-adds and the operation is trivially parallel. Because it is a fixed linear projection, it can also be folded into the first layer of a network if desired, though most pipelines keep it as an explicit preprocessing step.</span>

<span style="font-size: 14px;">The linearity has a useful consequence: grayscale conversion commutes with any other per-pixel linear operation. Brightening every channel by a factor $c$ scales $Y$ by the same $c$, and adding two images before converting gives the same result as converting then adding. This is why luma can be precomputed and cached without worrying about ordering relative to other affine adjustments.</span>

---

## <span style="font-size: 16px;">Relationship to Other Colour Reductions</span>

<span style="font-size: 14px;">Luma is one of several ways to reduce colour to a scalar, and it helps to know how it differs from the alternatives:</span>

* <span style="font-size: 14px;">**Mean of channels** $(R+G+B)/3$: ignores perception entirely, treats every channel as equally bright. Simple but perceptually inaccurate.</span>
* <span style="font-size: 14px;">**Max of channels** $\max(R, G, B)$: this is the **Value** channel of HSV. It captures the brightest primary, not perceived brightness, so pure blue and pure white both map near the top of the range.</span>
* <span style="font-size: 14px;">**Lightness** $(\max + \min)/2$: the **L** channel of HSL. A different compromise that desaturates by averaging the extreme channels.</span>
* <span style="font-size: 14px;">**Luma (BT.601)** $0.299R + 0.587G + 0.114B$: the perception-weighted choice, and the default meaning of grayscale in OpenCV and PIL.</span>

<span style="font-size: 14px;">For the test patch pixel $[200, 120, 40]$: mean $= 120$, max $= 200$, lightness $= (200+40)/2 = 120$, luma $= 134.8$. The four reductions disagree by tens of levels, which is exactly why the choice of formula matters for any threshold or detector applied afterward.</span>

---

## <span style="font-size: 16px;">Use in Modern Deep Learning</span>

<span style="font-size: 14px;">Even though convolutional networks can ingest colour directly, grayscale conversion still appears throughout deep vision. It is a standard data-augmentation transform (`torchvision.transforms.Grayscale` and `RandomGrayscale`), used to make classifiers invariant to colour and to regularize training. It is the input domain for many medical and document-imaging models where colour is uninformative. It also defines the target space for colourization networks, which learn the inverse map from a single luma channel back to plausible RGB. Understanding that the forward map is a fixed, non-invertible linear projection clarifies why colourization is an ill-posed problem: infinitely many RGB triples share the same $Y$.</span>

<span style="font-size: 14px;">The non-invertibility is worth dwelling on. The weight vector $w = [0.299, 0.587, 0.114]$ maps a 3-dimensional colour space onto a 1-dimensional line. Every plane perpendicular to $w$ in RGB space collapses to a single luma value, so an entire 2-parameter family of colours becomes indistinguishable after conversion. This is the information-theoretic reason grayscale cannot be undone without a learned prior, and it is also why grayscale is such an effective compression and anonymization step when colour is not needed.</span>

---

## <span style="font-size: 16px;">Numerical Precision and Rounding</span>

<span style="font-size: 14px;">Because the weights are decimal fractions, the luma of integer-valued pixels is almost never an integer. The pixel $[200, 120, 40]$ produced exactly $134.80$, but most pixels land on long decimals: $[173, 91, 60]$ gives $0.299(173) + 0.587(91) + 0.114(60) = 51.727 + 53.417 + 6.84 = 111.984$. The problem asks for the value rounded to 4 decimals, so the output is $111.9840$, preserving fractional precision rather than collapsing to $112$.</span>

<span style="font-size: 14px;">Keeping the fractional luma matters when the grayscale output feeds further floating-point operations such as gradient computation or interpolation, where premature rounding to 8-bit integers would inject quantization noise. The rule of thumb is to defer rounding to the last possible step, ideally only when writing back to an 8-bit display buffer.</span>

<span style="font-size: 14px;">The same caution applies in reverse for very small intensities: a dark blue pixel like $[0, 0, 12]$ gives $Y = 0.114 \times 12 = 1.368$, which truncates to $1$ but rounds to $1.3680$. Across a large image these sub-integer differences accumulate into visible banding if dropped, so the 4-decimal output requested here intentionally preserves them.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Channel order confusion (RGB vs BGR).** OpenCV loads images as BGR by default, so applying RGB-ordered weights swaps the red and blue coefficients. The result is subtly wrong: blue gets $0.299$ and red gets $0.114$. Always confirm whether `image[i][j]` is $[R, G, B]$ or $[B, G, R]$ before indexing.</span>
* <span style="font-size: 14px;">**Integer truncation and dtype overflow.** Computing on `uint8` arrays can overflow during intermediate steps or truncate fractional results to integers. Cast to `float` before the weighted sum, then round at the end. This problem expects the float luma rounded to 4 decimals, not a truncated integer.</span>
* <span style="font-size: 14px;">**Using a plain mean instead of luma weights.** Averaging the channels is a common shortcut that produces a perceptually wrong intensity: it darkens greens and brightens blues relative to human perception. Downstream edge detectors and thresholds then respond differently than expected.</span>
* <span style="font-size: 14px;">**Mismatched standard (601 vs 709).** Mixing BT.601 weights with content authored for Rec. 709 (sRGB) gives a slightly off luminance. The two are close but not identical; pick the standard the rest of the pipeline assumes and stay consistent.</span>

---