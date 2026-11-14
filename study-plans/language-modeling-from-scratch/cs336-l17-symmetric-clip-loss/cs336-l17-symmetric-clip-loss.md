# Implement Symmetric CLIP Contrastive Loss

CLIP (Radford et al., 2021) learns to connect images and text by training an image encoder and a text encoder together on hundreds of millions of image-caption pairs. It learns to pick the right caption for each image out of a batch, and the right image for each caption, which is far cheaper than generating captions word by word. The heart of this problem is computing that symmetric contrastive loss from a batch of paired embeddings, along with the accuracy in each direction.

## Learning by matching pairs

A batch holds $B$ images and their $B$ captions, with image $i$ paired with caption $i$. Every other caption in the batch serves as a wrong answer for image $i$. The model is trained to make each image's embedding close to its own caption's embedding and far from the others.

This contrastive setup scales well because it needs no labels beyond the pairing itself, and every batch provides $B^2 - B$ negative pairs for free. Larger batches give more and harder negatives, which is why CLIP trained with batches of 32,768 pairs. The result is a shared space where an image and a sentence that describe the same thing land near each other, which makes zero-shot classification possible, since an image can be classified by comparing it with the embeddings of text such as "a photo of a dog".

## Cosine similarities with a temperature

Each embedding is first divided by its length so that only its direction matters. With normalized image vectors $\widehat{v}_i$, normalized text vectors $\widehat{t}_j$ and a temperature $\tau$, the similarity matrix is

$$
S_{ij} = \frac{\widehat{v}_i^{\mathsf T} \widehat{t}_j}{\tau}
$$

Each entry is a cosine similarity between $-1$ and $1$, divided by $\tau$. A small temperature stretches the differences between similarities, which sharpens the softmax that follows. CLIP learns its temperature during training, starting from 0.07.

## The symmetric loss

Each row of $S$ is a classification problem over captions, with the correct answer on the diagonal, and each column is a classification problem over images. Cross-entropy is applied in both directions:

$$
L = -\frac{1}{2B} \sum_{i=1}^{B} \left[ \log \frac{e^{S_{ii}}}{\sum_{j} e^{S_{ij}}} + \log \frac{e^{S_{ii}}}{\sum_{j} e^{S_{ji}}} \right]
$$

The first term asks each image to pick its caption, and the second asks each caption to pick its image. Averaging both makes the loss symmetric, so neither encoder is favored. Subtracting each row's or column's largest similarity before exponentiating, the log-sum-exp trick, leaves the softmax unchanged and keeps the exponentials from overflowing when $\tau$ is small.

## Accuracy in each direction

Image-to-text accuracy is the fraction of rows whose largest entry sits on the diagonal, and text-to-image accuracy is the same for columns. The two can differ, since an image can pick the right caption while that caption prefers a different image.

## A batch where both directions get one wrong

Take two pairs with temperature 0.5. Image 1 is $[1, 0]$ and image 2 is $[0.6, 0.8]$, while caption 1 is $[0.8, 0.6]$ and caption 2 is $[0, 1]$. All four already have length 1. The cosine similarities divided by 0.5 give

$$
S = \begin{bmatrix} 1.6 & 0 \\ 1.92 & 1.6 \end{bmatrix}
$$

Image 1 prefers caption 1, its own, but image 2 also prefers caption 1, with 1.92 against 1.6, so image-to-text accuracy is 0.5. In the columns, caption 1 prefers image 2, with 1.92 against 1.6, while caption 2 correctly prefers image 2, so text-to-image accuracy is also 0.5.

For the loss, row 1 costs $-\log \frac{e^{1.6}}{e^{1.6} + e^{0}} \approx 0.184$ and row 2 costs $-\log \frac{e^{1.6}}{e^{1.92} + e^{1.6}} \approx 0.866$. Column 1 costs $-\log \frac{e^{1.6}}{e^{1.6} + e^{1.92}} \approx 0.866$ and column 2 costs $-\log \frac{e^{1.6}}{e^{0} + e^{1.6}} \approx 0.184$. Averaging the four gives a loss of 0.524897. Image 2 and caption 1 sit too close together, and the gradient pushes them apart.

## Common mistakes to avoid

- **Skipping normalization.** Without dividing by the lengths, similarities depend on vector size and the temperature loses its meaning.
- **Computing only one direction.** The loss averages the image-to-text and text-to-image cross-entropies.
- **Using the wrong targets.** The correct class for row $i$ and for column $i$ is index $i$, the diagonal.
- **Multiplying by the temperature.** Similarities are divided by $\tau$, so a smaller temperature sharpens them.
- **Exponentiating similarities directly.** The log-sum-exp trick keeps large logits from overflowing.
