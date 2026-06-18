# <span style="font-size: 20px;">Padding and Truncation</span>

<span style="font-size: 14px;">Padding and truncation prepare variable-length text sequences for batch processing. Neural networks require fixed-size inputs within a batch, so all sequences must be adjusted to a uniform length.</span>

---

## <span style="font-size: 16px;">Why Padding and Truncation?</span>

* <span style="font-size: 14px;">Sentences have different lengths, but GPU operations require rectangular tensors</span>
* <span style="font-size: 14px;">Padding adds special tokens to short sequences, truncation cuts long ones</span>
* <span style="font-size: 14px;">Together they ensure every sequence in a batch has exactly the same length</span>
* <span style="font-size: 14px;">This is the last preprocessing step before feeding text into a model</span>

---

## <span style="font-size: 16px;">Truncation</span>

<span style="font-size: 14px;">If a sequence exceeds the maximum length, it must be shortened:</span>

* <span style="font-size: 14px;">**Right truncation** (most common): Keep the first</span> `max_length` <span style="font-size: 14px;">tokens, discard the rest. Works well for classification where the beginning is most informative</span>
* <span style="font-size: 14px;">**Left truncation**: Keep the last tokens. Useful for language models where recent context matters most</span>
* <span style="font-size: 14px;">BERT has a maximum context of 512 tokens. GPT-2 supports 1024 tokens. Exceeding these limits requires truncation</span>

---

## <span style="font-size: 16px;">Padding</span>

<span style="font-size: 14px;">If a sequence is shorter than the target length, pad it with a special token:</span>

* <span style="font-size: 14px;">**Right padding** (most common): Add padding tokens after the sequence. Standard for encoder models like BERT</span>
* <span style="font-size: 14px;">**Left padding**: Add padding tokens before the sequence. Preferred for decoder/generation models so the actual tokens are at the end, adjacent to the generation position</span>
* <span style="font-size: 14px;">The padding token is typically 0, corresponding to the</span> `<pad>` <span style="font-size: 14px;">token in the vocabulary</span>

---

## <span style="font-size: 16px;">Attention Masks</span>

<span style="font-size: 14px;">Padding tokens should not influence model computation. Attention masks solve this:</span>

* <span style="font-size: 14px;">A binary mask with 1 for real tokens and 0 for padding tokens</span>
* <span style="font-size: 14px;">The attention mechanism multiplies by this mask, zeroing out attention to padding positions</span>
* <span style="font-size: 14px;">Without attention masks, the model would attend to padding tokens, degrading performance</span>

---

## <span style="font-size: 16px;">Dynamic vs Static Padding</span>

* <span style="font-size: 14px;">**Static padding**: Pad all sequences to a fixed maximum length (e.g., 512). Simple but wastes computation on short sequences</span>
* <span style="font-size: 14px;">**Dynamic padding**: Pad to the length of the longest sequence in the batch. More efficient, used by HuggingFace's</span> `DataCollatorWithPadding`
* <span style="font-size: 14px;">**Bucketing**: Group sequences of similar length into the same batch, minimizing padding overhead</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Why not just use variable-length inputs?** GPU parallelism requires tensors with uniform dimensions. Variable-length sequences would force sequential processing, losing the speed advantage of batching</span>
* <span style="font-size: 14px;">**Does padding affect loss computation?** Yes - you must mask padding tokens when computing the loss, otherwise the model optimizes for predicting padding, which is meaningless</span>
* <span style="font-size: 14px;">**What padding value should you use?** Typically 0. PyTorch's</span> `nn.Embedding` <span style="font-size: 14px;">has a</span> `padding_idx` <span style="font-size: 14px;">parameter that zeros out gradients for the padding index</span>

---