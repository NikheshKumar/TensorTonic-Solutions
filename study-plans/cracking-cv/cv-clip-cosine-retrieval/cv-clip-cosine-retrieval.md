# <span style="font-size: 20px;">CLIP Cosine Retrieval</span>

<span style="font-size: 14px;">CLIP cosine retrieval ranks corpus items by their cosine similarity to a query embedding, returning the top-$k$ matches. It is the inference-time operation that makes CLIP (Radford et al., 2021, "Learning Transferable Visual Models From Natural Language Supervision") useful for zero-shot classification, image-text search, and cross-modal retrieval: because CLIP trains image and text encoders into a shared space, a simple cosine similarity is all that is needed to match across modalities.</span>

---

## <span style="font-size: 16px;">What It Computes</span>

<span style="font-size: 14px;">Given $M$ query vectors and $N$ corpus vectors in $\mathbb{R}^D$, the task returns, for each query, the indices and scores of the $k$ corpus items with the highest cosine similarity. The similarity between a query $q_i$ and a corpus item $c_j$ is:</span>

$$
\text{sim}(q_i, c_j) = \frac{q_i \cdot c_j}{(\lVert q_i \rVert_2 + \varepsilon)\,(\lVert c_j \rVert_2 + \varepsilon)}
$$

<span style="font-size: 14px;">A small $\varepsilon = 10^{-12}$ is added to each norm so that an all-zero embedding yields a finite similarity of zero rather than a division by zero. Results are sorted in descending similarity, with ties broken by the lower corpus index, and scores are rounded to 4 decimals.</span>

<span style="font-size: 14px;">The $\varepsilon$ sits on the denominator, added to each norm before multiplying, rather than inside the square root. For a non-zero vector its effect is negligible ($10^{-12}$ against a norm of order 1), so it does not perturb real similarities. For a zero vector the numerator is exactly 0 and the denominator is $\varepsilon \cdot \lVert c_j \rVert$, giving a clean 0 instead of an undefined $0/0$. This is the standard guard used in retrieval code that must tolerate degenerate or padded rows.</span>

---

## <span style="font-size: 16px;">Cosine Similarity</span>

<span style="font-size: 14px;">Cosine similarity measures the angle between two vectors, ignoring their magnitudes:</span>

$$
\cos\theta = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}
$$

<span style="font-size: 14px;">It ranges in $[-1, 1]$: a value of 1 means the vectors point in the same direction, 0 means orthogonal, and $-1$ means opposite. By dividing out both norms, cosine similarity captures **direction only**, which is exactly what is wanted when comparing embeddings: two captions describing the same image should match regardless of how long their embedding vectors happen to be.</span>

<span style="font-size: 14px;">Geometrically, the dot product $a \cdot b$ equals $\lVert a \rVert \lVert b \rVert \cos\theta$, so cosine is just the dot product with the two magnitudes factored out, leaving the cosine of the angle $\theta$ between the vectors. In high-dimensional embedding spaces this angular view is robust: random vectors in high dimensions are nearly orthogonal (cosine near 0), so a meaningfully positive cosine is a strong signal of semantic relatedness rather than chance overlap.</span>

<span style="font-size: 14px;">When vectors are already L2-normalized to unit length, the denominator becomes 1 and cosine similarity reduces to a plain dot product. CLIP normalizes its image and text embeddings during both training and inference, so in practice the retrieval is a normalized dot product, often implemented as a single matrix multiply $Q C^\top$ after normalization.</span>

<span style="font-size: 14px;">This matrix form is why CLIP retrieval scales well. Normalizing $M$ queries and $N$ corpus vectors is $O((M+N)D)$, and the full similarity matrix is one $(M, D) \times (D, N)$ matmul costing $O(MND)$, which modern hardware executes extremely fast. For very large corpora the brute-force matmul is replaced by approximate nearest-neighbor indexes (FAISS, HNSW) that store the unit vectors and answer top-$k$ cosine queries in sublinear time, but the underlying metric is unchanged.</span>

---

## <span style="font-size: 16px;">The Contrastive Pretraining Context</span>

<span style="font-size: 14px;">CLIP is trained on 400 million image-text pairs scraped from the web. Each batch of $n$ pairs is encoded into $n$ image vectors and $n$ text vectors, all L2-normalized. CLIP then computes the full $n \times n$ matrix of cosine similarities scaled by a learned temperature, and applies a symmetric cross-entropy loss that pushes the $n$ correct image-text pairs (the diagonal) to high similarity and the $n^2 - n$ mismatched pairs (off-diagonal) to low similarity.</span>

<span style="font-size: 14px;">This contrastive objective is the reason cosine retrieval works at inference. The training never asks the model to predict exact pixel values or words; it only asks it to make matching pairs more similar than non-matching ones under cosine similarity. The geometry of the learned space is therefore organized precisely so that nearest-neighbor search by cosine similarity recovers semantically matching items. Retrieval is not an afterthought bolted on later, it is the literal training signal.</span>

<span style="font-size: 14px;">The CLIP paper makes a deliberate design point here: earlier attempts predicted the exact caption of an image (a generative objective) and learned slowly because predicting precise words is far harder than predicting whether two things match. Switching to a contrastive matching objective gave a roughly four-times efficiency gain in their experiments. The in-batch negatives are central: with batch size $n$, each image is contrasted against $n-1$ wrong captions for free, so a large batch (CLIP used $32{,}768$) supplies an enormous number of negative pairs per step, sharpening the embedding space cheaply.</span>

<span style="font-size: 14px;">A subtle consequence is that CLIP embeddings live, by construction, on the unit sphere. The L2 normalization in training forces every embedding to unit length, so the only thing that can vary is direction. This is why magnitude carries no semantic information at inference and why cosine, the angle metric, is the principled similarity to use.</span>

---

## <span style="font-size: 16px;">Temperature</span>

<span style="font-size: 14px;">CLIP scales the cosine similarities by a temperature $\tau$ before the softmax in its loss:</span>

$$
\text{logit}_{ij} = \frac{\cos(q_i, c_j)}{\tau}
$$

<span style="font-size: 14px;">Because cosine similarity is bounded in $[-1, 1]$, the raw range is too narrow for a sharp softmax. Dividing by a small $\tau$ (CLIP parameterizes it as a learned log value, initialized so the effective scale is around $1/0.07$) stretches the logits so the softmax can become confident. Temperature affects only the relative scale of the scores and the sharpness of the distribution; for pure top-$k$ retrieval by ranking it does not change the argmax ordering, since dividing every score by the same positive constant preserves order. It matters when the scores feed a softmax, as in zero-shot classification.</span>

<span style="font-size: 14px;">Making $\tau$ learnable rather than a fixed hyperparameter lets the model decide how peaked its similarity distribution should be over training. CLIP clamps the learned value to prevent it from collapsing to an extreme that would destabilize the loss. Conceptually, a low temperature makes the contrastive loss penalize even small margins between the correct pair and the hardest negative, driving the matching pair to be unambiguously the most similar, which is exactly the property top-$k$ retrieval depends on.</span>

---

## <span style="font-size: 16px;">From Retrieval to Zero-Shot Classification</span>

<span style="font-size: 14px;">Zero-shot classification is retrieval in disguise. To classify an image among a set of labels, CLIP turns each label into a text prompt such as "a photo of a {label}", encodes all prompts into text embeddings, and treats them as the corpus. The image embedding is the query. The label whose text embedding has the highest cosine similarity to the image is the prediction, exactly the top-1 retrieval result.</span>

<span style="font-size: 14px;">This is what made CLIP striking: with no task-specific training, it matched a fully supervised ResNet-50 on ImageNet purely by comparing the image to the text embeddings of the class names. The same retrieval primitive also drives image-to-image search (query is an image, corpus is images) and text-to-image search (query is text, corpus is images), all using one shared embedding space and one cosine similarity.</span>

---

## <span style="font-size: 16px;">The Retrieval Algorithm</span>

<span style="font-size: 14px;">1. **Compute norms**: for each query and each corpus vector, compute its L2 norm and add $\varepsilon = 10^{-12}$.</span>

<span style="font-size: 14px;">2. **Build the similarity matrix**: for every query-corpus pair, divide the dot product by the product of the two padded norms, giving an $(M, N)$ matrix.</span>

<span style="font-size: 14px;">3. **Sort each row descending**: rank the $N$ corpus items for each query by similarity, breaking ties toward the lower corpus index.</span>

<span style="font-size: 14px;">4. **Take the top $k$**: keep the first $k$ indices and their scores per query, rounding scores to 4 decimals.</span>

<span style="font-size: 14px;">The output is a dict with `indices` of shape $(M, k)$ (integer corpus indices) and `scores` of shape $(M, k)$ (the corresponding cosine values).</span>

<span style="font-size: 14px;">An important ordering detail: the ranking must be done on full-precision similarities, and rounding to 4 decimals applied only to the reported scores. The tie-break rule, lower corpus index wins, is implemented with a stable sort: rank by descending similarity, and where similarities are equal preserve the original ascending index order. This guarantees deterministic output even when the embeddings produce exactly equal scores, which is common with synthetic or quantized vectors.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Let the query be $q = [1, 0]$ and the corpus be $c_0 = [1, 1]$, $c_1 = [2, 0]$, $c_2 = [0, 3]$, with $k = 2$.</span>

* <span style="font-size: 14px;">$\text{sim}(q, c_0) = (1\cdot1 + 0\cdot1) / (1 \cdot \sqrt{2}) = 1/1.4142 \approx 0.7071$</span>
* <span style="font-size: 14px;">$\text{sim}(q, c_1) = (1\cdot2 + 0\cdot0) / (1 \cdot 2) = 2/2 = 1.0000$</span>
* <span style="font-size: 14px;">$\text{sim}(q, c_2) = (1\cdot0 + 0\cdot3) / (1 \cdot 3) = 0/3 = 0.0000$</span>

<span style="font-size: 14px;">Sorting descending gives $c_1$ (1.0), $c_0$ (0.7071), $c_2$ (0.0). The top-2 indices are $[1, 0]$ with scores $[1.0, 0.7071]$. Note $c_1 = [2,0]$ scores higher than $c_0 = [1,1]$ because cosine rewards alignment of direction, not magnitude: $c_1$ points exactly along $q$ so its angle is $0$, while $c_0$ sits at $45$ degrees. A plain dot product would also rank $c_1$ first here (2 vs 1), but in general dot product and cosine can disagree whenever magnitudes differ.</span>

---

## <span style="font-size: 16px;">Comparison With Other Metrics</span>

* <span style="font-size: 14px;">**Dot product.** Faster (no normalization) but conflates magnitude with relevance: a long vector can dominate retrieval regardless of direction. Equivalent to cosine only when vectors are unit-normalized, which is why CLIP normalizes first.</span>
* <span style="font-size: 14px;">**Euclidean distance.** Measures absolute separation. For unit-normalized vectors it is monotonically related to cosine ($\lVert a - b \rVert^2 = 2 - 2\cos\theta$), so ranking by smallest Euclidean distance gives the same order as largest cosine. Off the unit sphere they differ.</span>
* <span style="font-size: 14px;">**Cosine similarity.** Magnitude-invariant, bounded in $[-1, 1]$, the natural choice for embedding spaces where direction carries meaning. The standard metric for CLIP and most dense retrieval systems.</span>

---

## <span style="font-size: 16px;">Variants and Practical Notes</span>

* <span style="font-size: 14px;">**Prompt engineering and ensembling.** CLIP's zero-shot accuracy depends on the text prompts. The paper found that ensembling many prompt templates ("a photo of a {label}", "a blurry photo of a {label}", etc.) and averaging their text embeddings improves accuracy by a few points, because it stabilizes the corpus side of the retrieval.</span>
* <span style="font-size: 14px;">**The modality gap.** Image and text embeddings, though in a shared space, tend to occupy separate cones rather than fully intermixing. Cosine retrieval still works because within each query's row the relative ordering is what matters, but absolute cross-modal scores are often lower than intra-modal scores.</span>
* <span style="font-size: 14px;">**SigLIP.** Zhai et al., 2023 replace CLIP's softmax contrastive loss with a pairwise sigmoid loss, removing the need for a global normalization across the batch and improving training at smaller batch sizes. The retrieval interface, cosine over normalized embeddings, is identical.</span>
* <span style="font-size: 14px;">**Downstream use.** The same embeddings feed diffusion models (text conditioning), open-vocabulary detection, and multimodal LLMs, all of which rely on the cosine-aligned space that retrieval exposes.</span>
* <span style="font-size: 14px;">**Bidirectional retrieval.** Swapping the roles of queries and corpus gives text-to-image and image-to-text retrieval from the same embeddings. Benchmarks like Flickr30k and MS-COCO report Recall@1/5/10 in both directions, all computed as cosine top-$k$.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Dividing by zero on empty rows.** An all-zero embedding has norm 0. Without the $\varepsilon$ on the denominator, its similarity is $0/0 = $ NaN, which then poisons sorting (NaN comparisons are undefined). Adding $\varepsilon = 10^{-12}$ makes the similarity cleanly 0 instead.</span>
* <span style="font-size: 14px;">**Unstable tie-breaking.** When two corpus items have equal similarity, the spec requires breaking ties toward the lower index. A non-stable sort or one that breaks ties arbitrarily produces nondeterministic indices that fail exact-match tests, even though the scores are correct.</span>
* <span style="font-size: 14px;">**Forgetting to normalize, or normalizing twice.** Using raw dot products instead of cosine lets high-magnitude vectors win regardless of direction. Conversely, normalizing inputs and then dividing by norms again double-counts the normalization and shrinks all scores. The norm must be applied exactly once, in the denominator.</span>
* <span style="font-size: 14px;">**Rounding before sorting.** Rounding scores to 4 decimals before ranking can create artificial ties or flip the order of items that differ only in the fifth decimal. Sort on the full-precision similarity, then round only the reported scores.</span>

---