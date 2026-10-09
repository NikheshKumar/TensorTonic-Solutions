# <span style="font-size: 20px;">Top-K Classification Accuracy</span>

<span style="font-size: 14px;">Top-K accuracy is a classification metric that counts a prediction as correct if the true label appears among the model's $k$ highest-scored classes. It relaxes ordinary accuracy, which demands the single top guess be right, and is the standard way image classification results are reported on ImageNet, where both top-1 and top-5 accuracy are quoted side by side.</span>

---

## <span style="font-size: 16px;">What It Measures</span>

<span style="font-size: 14px;">A classifier outputs a score (logit) for each of $K$ classes. Top-1 accuracy asks only whether the single highest-scoring class is the correct one. Top-K accuracy is more forgiving: it asks whether the correct class is anywhere in the top $k$ ranked classes. A sample is counted correct if its true label is among those $k$.</span>

<span style="font-size: 14px;">The motivation is that a single forced guess is a harsh test when many classes are visually or semantically close. A model that ranks the correct class second out of a thousand has clearly understood the image, yet top-1 scores it identically to a model that ranked it dead last. Top-K credits the model for surfacing the right answer near the top, which is often what downstream use cares about: a retrieval system or a human-in-the-loop reviewer typically inspects several candidates, not just one.</span>

<span style="font-size: 14px;">Given logits of shape $(N, K)$ and integer targets of shape $(N,)$ with each target in $[0, K)$, the metric is:</span>

$$
\text{acc} = \frac{1}{N} \sum_{i=0}^{N-1} \mathbb{1}\!\left[\, \text{targets}[i] \in \text{TopK}(\text{logits}[i], k) \,\right]
$$

<span style="font-size: 14px;">where $\mathbb{1}[\cdot]$ is the indicator function, 1 when the condition holds and 0 otherwise, and $\text{TopK}(\text{logits}[i], k)$ is the set of class indices with the $k$ largest logits for sample $i$.</span>

---

## <span style="font-size: 16px;">Defining TopK</span>

<span style="font-size: 14px;">$\text{TopK}(\text{logits}[i], k)$ is defined as the first $k$ entries of the argsort of the logits in descending order, using a stable ordering so that ties are broken by the lower class index. Concretely:</span>

<span style="font-size: 14px;">1. **Argsort descending**: order the $K$ class indices by their logit value, largest first.</span>

<span style="font-size: 14px;">2. **Stable tie-break**: when two classes have equal logits, the one with the smaller index comes first.</span>

<span style="font-size: 14px;">3. **Take $k$**: keep the first $k$ class indices from that ordering.</span>

<span style="font-size: 14px;">4. **Membership test**: the sample is correct if its target index is in that set of $k$.</span>

<span style="font-size: 14px;">The stable-descending requirement can be read as a single total order on classes: sort by logit descending, and within equal logits by index ascending. Equivalently, one can sort ascending by the pair $(-\text{logit}, \text{index})$, which a stable lexicographic sort handles directly. Both phrasings produce the same top-$k$ set; the key invariant is that the comparison is total and deterministic, so the metric is reproducible bit-for-bit across runs and machines.</span>

<span style="font-size: 14px;">The accuracy is the fraction of samples that pass the test, returned as a single float rounded to 4 decimals. Note the metric only depends on the **ranking** of the logits, not their absolute values, so applying a softmax first does not change the result: softmax is monotonic, so it preserves the order.</span>

<span style="font-size: 14px;">This rank-invariance has a practical upshot: top-K accuracy can be computed directly from logits without normalization, which is why deep learning libraries expose it as a metric over raw model outputs. It also means the metric ignores calibration entirely. A model that is correct but wildly overconfident, and a model that is correct but barely confident, score identically. Top-K measures only whether the right answer is ranked high enough, never how sure the model is, which is a strength when comparing architectures but a blind spot if calibration is what matters.</span>

---

## <span style="font-size: 16px;">Top-1 vs Top-5: the ImageNet Convention</span>

<span style="font-size: 14px;">ImageNet (Deng et al., 2009; the ILSVRC challenge) has 1000 classes, many of them fine-grained and genuinely ambiguous: dozens of dog breeds, many species of birds, multiple kinds of similar objects. An image labeled "Siberian husky" may look almost identical to "Eskimo dog". Demanding the exact top-1 match penalizes a model for a near-miss between two visually indistinguishable classes.</span>

<span style="font-size: 14px;">Top-5 accuracy was adopted as the headline ILSVRC metric for this reason: a prediction is correct if the true label is among the model's five best guesses, which tolerates these fine-grained confusions while still requiring the model to be in the right neighborhood. Historically top-5 is the number that defined milestones, AlexNet's 2012 result and the later sub-human-error claims were top-5. As models improved, top-1 became the more discriminating headline number, but both are still reported together.</span>

<span style="font-size: 14px;">Top-1 is always less than or equal to top-5: any sample correct at top-1 is also correct at top-5, since the top guess is in the top five. The gap between them measures how often the model is close but not exactly right.</span>

<span style="font-size: 14px;">The choice of $k = 5$ specifically was tied to the ILSVRC annotation protocol. Each ImageNet image has a single label, but an image can plausibly contain several of the 1000 categories, so allowing five guesses acknowledged that a single forced choice is sometimes unfair to the model. As classification matured, the field shifted emphasis back toward top-1 because top-5 saturated near 99 percent and stopped distinguishing strong models. Modern papers (ViT, ConvNeXt, EVA) lead with top-1 on ImageNet-1k and may omit top-5 entirely, but the dual reporting convention persists for historical comparability.</span>

---

## <span style="font-size: 16px;">Efficient Computation</span>

<span style="font-size: 14px;">A full argsort of each row costs $O(K \log K)$, but only the top $k$ are needed. A partial selection ($\text{argpartition}$ to find the $k$ largest, or a max-heap of size $k$) runs in $O(K)$ or $O(K \log k)$, which matters when $K$ is large (1000 for ImageNet, tens of thousands for some retrieval-style classifiers) and $k$ is small. The membership test per sample is then $O(k)$.</span>

<span style="font-size: 14px;">A clean vectorized formulation avoids per-sample sorting entirely: take the top-$k$ indices of each row in one batched operation, then test for each sample whether its target equals any of its $k$ indices, and average the resulting boolean vector. The subtlety is that batched top-$k$ routines must still honor the tie-break convention; a partial-selection primitive that does not guarantee stable ordering at the boundary can disagree with a stable full sort exactly on tied logits, so the convention has to be respected even in the fast path.</span>

---

## <span style="font-size: 16px;">Why Tie-Breaking Matters</span>

<span style="font-size: 14px;">Real-valued logits from a trained network rarely tie exactly, but ties are common with synthetic inputs, integer logits, or after aggressive quantization. The result must be deterministic, so the spec fixes the rule: ties go to the lower class index via a stable sort. This matters most exactly at the boundary, the $k$-th position.</span>

<span style="font-size: 14px;">Consider $k = 1$ with logits $[5, 5, 3]$ for classes $0, 1, 2$. Classes 0 and 1 tie for the top. The stable rule picks class 0 as the single top-1, so a sample with target 1 is counted **wrong** even though its logit equals the winner's. A different tie rule (picking the higher index, or breaking randomly) would flip this outcome, which is why the convention must be pinned down.</span>

<span style="font-size: 14px;">The danger is concentrated at the $k$-th rank because that is the boundary of the kept set. Ties strictly inside the top $k$ are harmless, since reordering members of the set does not change membership, and ties strictly below rank $k$ are equally harmless. Only a tie that straddles the cutoff, with the target on the losing side of the cutoff, can change the verdict. A robust implementation therefore needs a deterministic ordering only to resolve which tied class sits at exactly position $k$, and the lower-index rule supplies that determinism cheaply.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">The metric also generalizes beyond ImageNet's 1000-way setup. In large-vocabulary language modeling, top-K (often top-1 next-token accuracy) tracks how often the true next token is the argmax. In recommendation and retrieval, hit-rate@k is the same computation over a catalog of items. The defining structure is always: rank scores, keep the top $k$, test whether the ground truth made the cut.</span>

<span style="font-size: 14px;">Take $N = 3$ samples, $K = 4$ classes, and evaluate top-2 accuracy ($k = 2$).</span>

* <span style="font-size: 14px;">**Sample 0:** logits $[0.1, 0.9, 0.3, 0.2]$, target 1. Descending order of indices: $1, 2, 3, 0$. Top-2 = $\{1, 2\}$. Target 1 is in it: correct.</span>
* <span style="font-size: 14px;">**Sample 1:** logits $[2.0, 0.5, 0.1, 0.4]$, target 3. Descending: $0, 1, 3, 2$. Top-2 = $\{0, 1\}$. Target 3 not in it: wrong.</span>
* <span style="font-size: 14px;">**Sample 2:** logits $[0.2, 0.2, 0.9, 0.1]$, target 0. Descending with tie between 0 and 1 broken toward lower index: $2, 0, 1, 3$. Top-2 = $\{2, 0\}$. Target 0 is in it: correct.</span>

<span style="font-size: 14px;">Two of three samples are correct, so top-2 accuracy $= 2/3 = 0.6667$. For comparison, top-1 on the same data: sample 0 picks class 1 (correct), sample 1 picks class 0 (wrong, target 3), sample 2 picks class 2 (wrong, target 0), giving $1/3 = 0.3333$, lower than top-2 as expected.</span>

<span style="font-size: 14px;">Sample 2 is the instructive case. Classes 0 and 1 both have logit 0.2, but the tie-break favors the lower index, so when filling positions after the clear winner (class 2), class 0 lands at rank 2 and class 1 at rank 3. The top-2 set is therefore $\{2, 0\}$, which includes the target 0, making the sample correct. Had the tie gone to the higher index, the top-2 would be $\{2, 1\}$ and the sample would be wrong, flipping the overall accuracy to $1/3$. This shows how a single boundary tie can change the reported number, which is precisely why the convention is fixed.</span>

---

## <span style="font-size: 16px;">Relationship to Other Metrics</span>

* <span style="font-size: 14px;">**Top-1 accuracy.** The special case $k = 1$. The strictest and most commonly cited single number for modern classifiers.</span>
* <span style="font-size: 14px;">**Top-K is monotonic in $k$.** Increasing $k$ can only keep or raise the score, since enlarging the candidate set never removes a previously included correct label. At $k = K$ accuracy is trivially 1, and the curve of accuracy versus $k$ is a useful diagnostic of how often the model is close.</span>
* <span style="font-size: 14px;">**Recall@k in retrieval.** The same idea appears in information retrieval as recall at $k$: whether the relevant item is in the top $k$ retrieved results. Top-K classification is recall@k with a single relevant class per query.</span>
* <span style="font-size: 14px;">**Mean reciprocal rank.** A related ranking metric that rewards the correct class for being ranked higher, not just present in the top $k$. Top-K throws away rank position within the top $k$; MRR keeps it.</span>
* <span style="font-size: 14px;">**Cross-entropy loss.** The training objective is closely related but continuous: it penalizes the model by the negative log-probability of the true class, so it cares about how much probability mass the true class gets, not just its rank. Top-K is the discrete, non-differentiable evaluation counterpart, which is why it is reported but not optimized directly.</span>

---

## <span style="font-size: 16px;">Interpreting the Numbers</span>

<span style="font-size: 14px;">A random classifier over $K$ balanced classes has expected top-K accuracy $k/K$, so on ImageNet random top-1 is $0.001$ and random top-5 is $0.005$. This sets the floor: any reported accuracy should be read relative to that chance baseline, and a top-5 number looking high is partly because the chance baseline is five times the top-1 baseline.</span>

<span style="font-size: 14px;">The gap between top-1 and top-5 is itself informative. A large gap means the model frequently ranks the truth second through fifth, suggesting confusion among similar classes rather than gross errors. A small gap on a strong model means that when it is wrong at top-1, the truth is usually not even in the top five, indicating harder, more fundamental failures. Reporting both, as ImageNet does, exposes this structure that a single number would hide.</span>

<span style="font-size: 14px;">Top-K accuracy is also an unbiased estimate only to the extent the evaluation set is representative. Because it averages a 0/1 indicator over $N$ samples, its standard error scales like $\sqrt{p(1-p)/N}$, so small validation sets give noisy top-K numbers and apparent differences of a fraction of a percent between models can be within noise. This is why ImageNet's 50,000-image validation set is large enough to resolve the sub-percent gaps that separate state-of-the-art models.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Sorting ascending instead of descending.** TopK needs the $k$ largest logits, so the argsort must be descending (or take the last $k$ of an ascending sort). Taking the first $k$ of an ascending sort returns the $k$ smallest, silently inverting the metric and producing near-zero accuracy.</span>
* <span style="font-size: 14px;">**Unstable or inconsistent tie-breaking.** When logits tie at the $k$-th boundary, an unstable sort makes membership depend on implementation details, producing nondeterministic accuracy that fails exact tests. The convention is a stable sort favoring the lower class index.</span>
* <span style="font-size: 14px;">**Rounding or thresholding logits before ranking.** Rounding logits to fewer decimals before the argsort can create artificial ties or reorder classes that differ slightly, changing which classes fall inside the top $k$. Rank on the raw logits and round only the final accuracy.</span>
* <span style="font-size: 14px;">**Off-by-one on $k$ or target range.** Taking $k+1$ or $k-1$ entries, or assuming labels are 1-indexed when targets are in $[0, K)$, shifts every membership test. These off-by-one errors do not crash but quietly bias the reported accuracy.</span>

---