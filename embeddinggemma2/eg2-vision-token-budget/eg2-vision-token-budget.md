# Vision Token Budget

The vision token budget sits between the EmbeddingGemma 2 vision encoder and the shared transformer backbone. It caps how many soft tokens one image or one video frame may contribute, so the cost of an image no longer grows with its resolution. This problem covers the pooling step that shrinks a grid of patch features until it fits that cap.

## Why patch grids overflow the sequence

A vision transformer with patch size 16 turns an image into a grid of patch features, one per 16 by 16 pixel square. An image of 896 by 896 pixels already yields a 56 by 56 grid, which is 3136 features. The backbone in EmbeddingGemma 2 trains on sequences of at most 8192 positions, and a video may carry up to 32 frames, so passing every patch through would leave no room for text and would make attention cost explode.

Long sequences also hurt the embedding, because the final representation is a mean over all output tokens and thousands of image tokens would swamp any text in the same input. Pooling neighboring patches keeps the coarse layout of the image while bounding its weight in that mean.

## A fixed budget during training and a dial at inference

The model trains with one budget per modality, 280 soft tokens for an image and 140 for each video frame. At inference the same weights accept 70, 140, 280, 560 or 1120 tokens, which turns the budget into a dial that trades speed for quality without retraining. Schechter Vera et al., 2026 measure it on MMEB-v2.

The modalities respond very differently. Visual documents depend on small print and fine layout, and their score keeps rising past the training budget, from 63.60 at 140 tokens and 63.38 at 280 to 67.62 at 560. Images saturate near 280. Video gets worse with more tokens, falling from 48.51 at 280 to 46.52 at 560, which the paper attributes to sequences so long that attention and the mean-pooled representation are diluted across 32 frames.

Training budgets matter as well. A model trained with 70 tokens per video frame does better on video when evaluated at 70, but it loses on images and documents, and at an evaluation budget of 140 it trails the 140-token model everywhere, 47.16 against 47.82 on video. That is why 140 tokens per frame is the training default. This flexibility belongs to the same design goal as truncatable embeddings and quantized weights, which is a single model that can shrink to fit the hardware it runs on.

## The pooling rule this exercise uses

The paper says images are pooled below the budget and stops there. This exercise chooses square windows of side $k$, picks the smallest $k$ whose window grid fits the budget $N$, lets windows on the bottom and right edges be partial and average only their real patches, and lists tokens in row-major window order. Taking the smallest valid $k$ keeps as many tokens as the budget allows, which preserves the most detail.

The window count is a product of two ceilings, and that is why no closed form gives $k$ directly. Guessing $k = \lceil \sqrt{G_h G_w / N} \rceil$ ignores the rounding on each axis. For a single row of 7 patches with $N = 3$, that guess gives $k = 2$ and four windows, while the true answer is $k = 3$ with three windows.

## Pooling a 3 by 5 grid into six tokens

Take $G_h = 3$, $G_w = 5$, $D = 1$, patch values 1 through 15 in row-major order, and $N = 6$. With $k = 1$ there are 15 windows, which is too many. With $k = 2$ there are $\lceil 3/2 \rceil \lceil 5/2 \rceil = 2 \times 3 = 6$ windows, which fits.

The top-left window holds 1, 2, 6 and 7 with mean 4, and the next holds 3, 4, 8 and 9 with mean 6. The top-right window is only one column wide and holds 5 and 10, so its mean is 7.5. The bottom row of windows is only one row tall and gives means 11.5 and 13.5, and the corner window holds 15 alone. The six tokens in order are 4, 6, 7.5, 11.5, 13.5 and 15. Dividing the top-right window by the full $k^2 = 4$ would wrongly report 3.75.

## Common mistakes to avoid

- **Using a square-root shortcut for the window.** The ceilings on each axis can push the window count above the budget, so the smallest valid window has to be found by checking candidates.
- **Dropping partial edge windows.** Floor division throws away the last rows or columns of the image and returns fewer tokens than the budget allows.
- **Dividing every window by $k^2$.** Zero padding is fine for summing, but a partial window must be divided by the number of patches it really contains.
- **Listing windows column by column.** The token order follows the row-major patch order, so the window index runs across a row before moving down.
- **Returning the window size as a tensor.** The second item is a plain Python integer that describes the pooling, and nothing downstream needs it as a tensor.
- **Averaging over the feature axis.** Pooling mixes neighboring patches, and every token keeps all $D$ feature channels.