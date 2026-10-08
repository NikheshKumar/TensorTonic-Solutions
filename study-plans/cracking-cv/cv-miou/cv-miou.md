# <span style="font-size: 20px;">Mean IoU for Segmentation</span>

<span style="font-size: 14px;">Mean Intersection-over-Union (mIoU) is the standard evaluation metric for semantic segmentation. It measures how well a predicted pixel-wise class map overlaps the ground truth, computing an IoU per class and averaging across classes. It is the headline metric on benchmarks like Pascal VOC, Cityscapes, and ADE20K, and is what papers report when claiming segmentation state of the art.</span>

---

## <span style="font-size: 16px;">What It Measures</span>

<span style="font-size: 14px;">Semantic segmentation assigns every pixel a class label. A natural way to score this is to ask, for each class, how much the predicted region for that class overlaps the true region for that class. Intersection-over-Union, also called the Jaccard index, captures exactly this: the area both agree on, divided by the area either one claims.</span>

<span style="font-size: 14px;">IoU originates from set theory as the Jaccard index, a similarity measure for sets defined as the size of the intersection over the size of the union. Semantic segmentation reuses it almost verbatim by treating each class's pixels as a set. The same index also underlies object-detection metrics, where IoU between predicted and ground-truth bounding boxes decides whether a detection counts as a match at a given threshold, so the concept recurs throughout computer vision.</span>

<span style="font-size: 14px;">For a single class $c$, treat the prediction and target as binary masks (pixel is class $c$ or not). Then:</span>

$$
\text{IoU}_c = \frac{|P_c \cap G_c|}{|P_c \cup G_c|}
$$

<span style="font-size: 14px;">where $P_c$ is the set of pixels predicted as class $c$ and $G_c$ is the set of pixels whose ground truth is class $c$. The value lies in $[0, 1]$: 1 is a perfect match, 0 means no overlap at all.</span>

<span style="font-size: 14px;">A few properties follow directly from the definition. IoU is symmetric in the two masks, swapping prediction and target leaves it unchanged. It is also strict: to reach 1, the masks must be exactly equal, since any extra or missing pixel enlarges the union beyond the intersection. And it is harder to satisfy than recall or precision alone, because both over-prediction (false positives) and under-prediction (false negatives) inflate the denominator. This strictness is why IoU correlates well with perceived segmentation quality.</span>

<span style="font-size: 14px;">Relating it to precision and recall makes the strictness concrete. Precision for class $c$ is $TP/(TP+FP)$ and recall is $TP/(TP+FN)$. IoU combines both denominators into one, $TP/(TP+FP+FN)$, so it can be high only when both precision and recall are high. A model that floods the image with class $c$ gets high recall but low precision, and IoU catches the resulting false positives that recall alone would miss.</span>

---

## <span style="font-size: 16px;">From Confusion Counts</span>

<span style="font-size: 14px;">In practice IoU is computed from per-class counts of true positives, false positives, and false negatives over the $H \times W$ pixel grid:</span>

$$
\text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}
$$

<span style="font-size: 14px;">where for class $c$:</span>

* <span style="font-size: 14px;">**$TP_c$** is the count of pixels where both prediction and target equal $c$ (the intersection)</span>
* <span style="font-size: 14px;">**$FP_c$** is pixels predicted $c$ but truly some other class</span>
* <span style="font-size: 14px;">**$FN_c$** is pixels truly $c$ but predicted as some other class</span>

<span style="font-size: 14px;">The denominator $TP_c + FP_c + FN_c$ equals the union size: every pixel in $P_c \cup G_c$ is counted once, with $TP_c$ counted in both masks but not double-counted because true negatives are excluded entirely. This is why IoU differs from accuracy: it never rewards correctly labeling the vast background as not-$c$.</span>

<span style="font-size: 14px;">It is worth seeing how the union formula arises. By inclusion-exclusion, $|P_c \cup G_c| = |P_c| + |G_c| - |P_c \cap G_c|$. Now $|P_c| = TP_c + FP_c$ (all pixels predicted $c$, correct or not), $|G_c| = TP_c + FN_c$ (all pixels truly $c$), and $|P_c \cap G_c| = TP_c$. Substituting gives $(TP_c + FP_c) + (TP_c + FN_c) - TP_c = TP_c + FP_c + FN_c$. The notable absence is $TN_c$, the pixels correctly excluded from class $c$; they never enter the formula, which is precisely what immunizes IoU against a large easy background.</span>

---

## <span style="font-size: 16px;">The Confusion Matrix View</span>

<span style="font-size: 14px;">The cleanest way to compute mIoU for $C$ classes is to build a single $C \times C$ confusion matrix $N$, where $N_{ab}$ counts pixels whose true class is $a$ and predicted class is $b$. From this one matrix every per-class quantity falls out by simple sums:</span>

* <span style="font-size: 14px;">$TP_c = N_{cc}$, the diagonal entry for class $c$</span>
* <span style="font-size: 14px;">$FN_c = \sum_b N_{cb} - N_{cc}$, the row sum minus the diagonal (truly $c$, predicted otherwise)</span>
* <span style="font-size: 14px;">$FP_c = \sum_a N_{ac} - N_{cc}$, the column sum minus the diagonal (predicted $c$, truly otherwise)</span>

<span style="font-size: 14px;">So $\text{IoU}_c = N_{cc} / (\text{row}_c + \text{col}_c - N_{cc})$. Building the confusion matrix in one pass over the pixels is the standard implementation, and it generalizes effortlessly from one image to a whole dataset by accumulating into the same matrix. It also makes the union formula self-evident: the row plus column double-counts the diagonal once, so subtracting $N_{cc}$ recovers the union.</span>

<span style="font-size: 14px;">The single-pass build is also why mIoU is cheap at scale. Flatten the prediction and target to 1D, form the linear index $a \cdot C + b$ for each pixel pair, and bin-count it into a length-$C^2$ histogram reshaped to $C \times C$. Whether the test set is one image or thousands, the only state carried forward is this $C \times C$ matrix, and the per-class IoUs are read off at the end. A class with a zero row and zero column has an empty union and is dropped from the mean automatically.</span>

---

## <span style="font-size: 16px;">Averaging Over Classes</span>

<span style="font-size: 14px;">The mean is taken over classes, not over pixels:</span>

$$
\text{mIoU} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{IoU}_c
$$

<span style="font-size: 14px;">A crucial detail is which classes belong in $\mathcal{C}$. A class is included only if it appears in either the prediction or the target, that is $TP_c + FP_c + FN_c > 0$. Classes absent from both masks have an undefined IoU ($0/0$) and are excluded from the average rather than counted as zero. If no class appears at all (an empty union for every class), the mean over an empty set is defined to be $0.0$.</span>

<span style="font-size: 14px;">This per-class averaging is deliberate and is what makes mIoU a fair metric for imbalanced scenes. Because each class contributes equally regardless of how many pixels it occupies, a model cannot inflate its score by nailing the dominant background while ignoring small but important classes like pedestrians or traffic signs.</span>

<span style="font-size: 14px;">The exclusion rule also matters for fair comparison across images. A validation set may contain images where only a few of the dataset's classes appear. Averaging over only the present classes per image, or better, accumulating counts dataset-wide and then averaging, prevents absent classes from injecting spurious zeros that would otherwise depend on label-set size rather than model quality. The empty-set fallback of $0.0$ is the boundary case of this rule: when nothing is present, there is no meaningful overlap to score.</span>

---

## <span style="font-size: 16px;">Why Not Pixel Accuracy</span>

<span style="font-size: 14px;">The obvious alternative, pixel accuracy (fraction of pixels labeled correctly), is badly misleading under class imbalance. In a street scene where road and sky cover most pixels, a model that labels everything road or sky can reach high pixel accuracy while completely missing every small object. Pixel accuracy rewards the majority class.</span>

<span style="font-size: 14px;">IoU resists this in two ways. First, it penalizes false positives: claiming extra pixels for a class inflates the union and lowers IoU. Pixel accuracy only counts correct pixels and ignores how the wrong ones are distributed. Second, the per-class mean gives a rare class the same weight as a common one. A model must do well on every class to score a high mIoU, which is exactly the behavior segmentation tasks care about.</span>

<span style="font-size: 14px;">Concretely, imagine a scene that is 95 percent road and 5 percent pedestrian, and a model that predicts road everywhere. Pixel accuracy is 95 percent, which sounds excellent. But $\text{IoU}_\text{road}$ is high while $\text{IoU}_\text{pedestrian} = 0$ (zero intersection), so mIoU is roughly $(0.95 + 0)/2 \approx 0.48$. The mIoU exposes the failure that pixel accuracy hides, which is exactly why segmentation benchmarks report it as the primary number.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider a tiny $2 \times 2$ image with 2 classes (0 and 1). Ground truth and prediction laid out row-major:</span>

* <span style="font-size: 14px;">**Target:** $[[0, 0], [1, 1]]$</span>
* <span style="font-size: 14px;">**Prediction:** $[[0, 1], [1, 1]]$</span>

<span style="font-size: 14px;">Flatten both to pixel lists: target $[0, 0, 1, 1]$, prediction $[0, 1, 1, 1]$.</span>

<span style="font-size: 14px;">**Class 0:** $TP_0 = 1$ (pixel 0 both 0), $FP_0 = 0$ (no pixel predicted 0 that is not 0), $FN_0 = 1$ (pixel 1 is truly 0 but predicted 1). $\text{IoU}_0 = 1/(1+0+1) = 0.5$.</span>

<span style="font-size: 14px;">**Class 1:** $TP_1 = 2$ (pixels 2,3 both 1), $FP_1 = 1$ (pixel 1 predicted 1 but truly 0), $FN_1 = 0$. $\text{IoU}_1 = 2/(2+1+0) = 0.6667$.</span>

<span style="font-size: 14px;">Both classes appear, so $\text{mIoU} = (0.5 + 0.6667)/2 = 0.5833$, rounded to 4 decimals. Note the single misclassified pixel hurts class 0 (as a false negative) and class 1 (as a false positive) simultaneously, which is why one wrong pixel can move mIoU more than pixel accuracy would suggest.</span>

<span style="font-size: 14px;">Compare with pixel accuracy on the same example: 3 of 4 pixels are correct, so accuracy is 0.75, noticeably higher than the 0.5833 mIoU. The gap comes from the same misclassified pixel being penalized twice under IoU, once per affected class, whereas accuracy charges it only once. As a sanity check, suppose a third class 2 is defined in the dataset but appears in neither mask: $TP_2 = FP_2 = FN_2 = 0$, its union is empty, so it is excluded and mIoU stays $0.5833$ rather than collapsing to $(0.5 + 0.6667 + 0)/3 = 0.3889$. This is exactly the absent-class rule in action.</span>

---

## <span style="font-size: 16px;">Where It Is Used</span>

<span style="font-size: 14px;">mIoU is the benchmark metric across the segmentation literature. Pascal VOC 2012 reports mean IoU over 20 object classes plus background; Cityscapes reports it over 19 urban classes; ADE20K over 150 classes. Architectures from FCN (Long et al., 2015) and U-Net through DeepLab, PSPNet, and the recent SegFormer and Mask2Former are all ranked by mIoU, which is why advances are quoted as point gains in mIoU.</span>

<span style="font-size: 14px;">Because IoU itself is non-differentiable (it counts discrete pixels), it is used for evaluation rather than as a training loss. Models typically train with per-pixel cross-entropy, sometimes augmented with a soft, differentiable surrogate such as the Lovasz-Softmax loss or a soft Dice loss that approximate IoU and align the training objective with the evaluation metric.</span>

<span style="font-size: 14px;">A soft IoU surrogate replaces the hard 0/1 masks with the predicted class probabilities: the soft intersection is the sum over pixels of $p_c \cdot g_c$ and the soft union the sum of $p_c + g_c - p_c g_c$, both differentiable in the probabilities $p_c$. Optimizing $1 - \text{soft-IoU}$ pushes the model directly toward the metric it will be judged on, which helps especially for small classes that cross-entropy under-weights.</span>

---

## <span style="font-size: 16px;">Variants</span>

* <span style="font-size: 14px;">**Frequency-weighted IoU.** Weights each class IoU by its pixel frequency before averaging, trading the equal-weight property for a measure closer to overall pixel coverage. Less common as a headline metric precisely because it reintroduces majority-class bias.</span>
* <span style="font-size: 14px;">**Dataset-level vs image-level mIoU.** Cityscapes and most benchmarks accumulate $TP$, $FP$, $FN$ across the whole dataset before computing per-class IoU, rather than averaging per-image IoUs. Per-image averaging over-weights small images and is noisier; the per-image variant is what this problem computes for a single image.</span>
* <span style="font-size: 14px;">**Ignore label.** Datasets mark unlabeled or void pixels with an ignore index (often 255). Those pixels are excluded from all $TP/FP/FN$ counts so they neither help nor hurt the score.</span>
* <span style="font-size: 14px;">**Boundary IoU.** A refinement that scores only pixels near object boundaries, to better distinguish models that differ mainly in edge quality, which standard mIoU under-weights.</span>
* <span style="font-size: 14px;">**Dice coefficient.** The closely related $\text{Dice} = 2TP/(2TP+FP+FN) = 2\,\text{IoU}/(1+\text{IoU})$. It ranks models identically to IoU but reads higher; medical imaging conventionally reports Dice while scene segmentation reports mIoU.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Counting absent classes as zero.** A class present in neither prediction nor target has IoU $0/0$, undefined. Counting it as 0 and including it in the mean drags mIoU down artificially and makes the score depend on how many classes the dataset defines rather than on prediction quality. Exclude such classes from the average.</span>
* <span style="font-size: 14px;">**Confusing IoU with the Dice coefficient.** Dice is $2TP/(2TP + FP + FN)$, which weights $TP$ twice and is always at least as large as IoU. They are monotonically related but not equal; reporting Dice and calling it IoU overstates the score.</span>
* <span style="font-size: 14px;">**Forgetting the empty-set case.** If every class has an empty union (for example both masks are entirely an ignore label), the mean is over zero classes. Dividing by a class count of 0 is a crash or NaN; the convention is to return $0.0$.</span>
* <span style="font-size: 14px;">**Double-counting the intersection in the union.** A common bug computes the union as $|P_c| + |G_c|$ instead of $|P_c| + |G_c| - |P_c \cap G_c|$, equivalently $TP + FP + FN$ rather than $TP + (TP+FP) + (TP+FN)$. The correct denominator counts each shared pixel once, giving the $TP + FP + FN$ form.</span>

---