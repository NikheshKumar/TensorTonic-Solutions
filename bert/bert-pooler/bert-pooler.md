# <span style="font-size: 20px;">BERT Pooler</span>

<span style="font-size: 14px;">The BERT Pooler converts a variable-length sequence of hidden states into a single fixed-size vector for sequence-level classification. It selects the hidden state at position 0 (the [CLS] token) and projects it through a dense layer with tanh activation. This small module bridges BERT's encoder output and any downstream task that requires one vector to represent the entire input.</span>

<span style="font-size: 14px;">In the BERT paper (Devlin et al., 2019): "The final hidden state corresponding to this token ([CLS]) is used as the aggregate sequence representation for classification tasks. We denote this vector as C in R^H." The pooler produces this C vector.</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">The pooler takes the encoder output tensor of shape $(B, T, H)$ and extracts only the hidden state at position 0 (the [CLS] token). It then performs:</span>

* <span style="font-size: 14px;">**Extraction:** Select the hidden state at index 0 from the sequence dimension, yielding a vector of shape $(B, H)$.</span>
* <span style="font-size: 14px;">**Linear projection:** Multiply by $W \in \mathbb{R}^{H \times H}$ and add bias $b \in \mathbb{R}^H$. Square projection preserving dimensionality.</span>
* <span style="font-size: 14px;">**Tanh activation:** Apply element-wise tanh to bound the output to $[-1, 1]$ and introduce nonlinearity.</span>

<span style="font-size: 14px;">The result is a fixed-size representation regardless of input length. Variable-length inputs become uniform-length vectors for classification.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $h_0, h_1, \ldots, h_{T-1}$ denote the hidden states from the final transformer layer. The [CLS] token is always at position 0, so $h_{\text{CLS}} = h_0 \in \mathbb{R}^H$.</span>

<span style="font-size: 14px;">The pooler computes:</span>

$$
h_{\text{pooled}} = \tanh(h_{\text{CLS}} \cdot W + b)
$$

* <span style="font-size: 14px;">**$h_{\text{CLS}} \in \mathbb{R}^H$:** Hidden state at position 0 from the final encoder layer.</span>
* <span style="font-size: 14px;">**$W \in \mathbb{R}^{H \times H}$:** Learnable weight matrix. Square because input and output share dimensionality.</span>
* <span style="font-size: 14px;">**$b \in \mathbb{R}^H$:** Learnable bias vector.</span>
* <span style="font-size: 14px;">**$\tanh$:** Applied element-wise, squashing each component to $[-1, 1]$.</span>

<span style="font-size: 14px;">For sequence classification, a classifier head is applied on top:</span>

$$
\text{logits} = h_{\text{pooled}} \cdot W_c + b_c
$$

* <span style="font-size: 14px;">**$W_c \in \mathbb{R}^{H \times C}$:** Classifier weight matrix, where $C$ is the number of classes.</span>
* <span style="font-size: 14px;">**$b_c \in \mathbb{R}^C$:** Classifier bias vector.</span>
* <span style="font-size: 14px;">**logits $\in \mathbb{R}^C$:** Unnormalized scores passed to softmax for probabilities.</span>

<span style="font-size: 14px;">The tanh function is defined as:</span>

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

<span style="font-size: 14px;">This maps any real value to $(-1, 1)$. Unlike sigmoid which maps to $(0, 1)$, tanh is zero-centered, meaning outputs have a mean closer to zero. This matters for downstream layers consuming the pooled representation.</span>

---

## <span style="font-size: 16px;">Why [CLS] Token</span>

<span style="font-size: 14px;">BERT prepends a special [CLS] token to every input. This token has no linguistic meaning. It serves as a "blank slate" that collects information from all other tokens through self-attention across all encoder layers.</span>

<span style="font-size: 14px;">In each self-attention layer, [CLS] computes attention over every other token. After 12 or 24 layers (BERT-Base or BERT-Large), its hidden state has aggregated global sequence information. Why position 0 specifically:</span>

* <span style="font-size: 14px;">**Consistent location:** Regardless of sequence length, [CLS] is always at index 0. The pooler extracts it with a simple indexing operation $h_0$.</span>
* <span style="font-size: 14px;">**No content bias:** [CLS] starts content-free, so its final hidden state is purely a function of what it attended to. A word like "good" carries sentiment bias from its embedding; [CLS] does not.</span>
* <span style="font-size: 14px;">**Neutral vantage point:** In sentence-pair tasks (like NLI), [CLS] sits before both segments, attending to both sentences equally.</span>
* <span style="font-size: 14px;">**Pre-training alignment:** During pre-training, the [CLS] position is used for Next Sentence Prediction (NSP), training it to encode inter-sentence relationships from the start.</span>

<span style="font-size: 14px;">Position 0 is a convention, not a mathematical necessity. GPT-style models use the last token instead (causal attention means only the final position has seen all preceding tokens). In BERT's bidirectional attention, every position sees every other, so the convention provides consistency.</span>

---

## <span style="font-size: 16px;">Why Tanh Activation</span>

<span style="font-size: 14px;">The pooler uses tanh rather than ReLU, GELU, or no activation.</span>

### <span style="font-size: 14px;">Bounded and Zero-Centered</span>

<span style="font-size: 14px;">Tanh bounds every component of $h_{\text{pooled}}$ to $[-1, 1]$, preventing any dimension from dominating the classifier. Outputs are centered around zero, so gradients for the downstream weight matrix are better conditioned and training converges faster. ReLU outputs are non-negative (mean > 0), which causes zig-zagging gradients.</span>

### <span style="font-size: 14px;">Why Not ReLU</span>

<span style="font-size: 14px;">ReLU ($\max(0, x)$) zeros out negative components, destroying useful information. After layer normalization, the [CLS] hidden state has both positive and negative values. ReLU discards roughly half the information. Tanh preserves the sign while compressing magnitude.</span>

### <span style="font-size: 14px;">Comparison with Other Pooling Strategies</span>

* <span style="font-size: 14px;">**Mean pooling:** Average all token hidden states. Can dilute important signals but captures distributed information.</span>
* <span style="font-size: 14px;">**Max pooling:** Element-wise max across positions. Captures strongest activation per dimension but loses positional information.</span>
* <span style="font-size: 14px;">**[CLS] without projection:** Simpler but lacks the nonlinear transformation that reshapes the representation.</span>
* <span style="font-size: 14px;">**Attention pooling:** Learned weights over positions. More flexible but adds parameters.</span>

<span style="font-size: 14px;">BERT uses [CLS] + dense + tanh: simple, minimal parameters ($H^2 + H$), and pre-trained through NSP.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">In "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2019), the pooler serves both pre-training and fine-tuning.</span>

### <span style="font-size: 14px;">The C Vector</span>

<span style="font-size: 14px;">The paper defines token-level representations $T_i$ (hidden state at each position) and a sequence-level representation $C$ (the pooled [CLS] hidden state). $C \in \mathbb{R}^H$ refers to the pooler output, not the raw [CLS] hidden state before projection.</span>

### <span style="font-size: 14px;">Pre-training with NSP</span>

<span style="font-size: 14px;">BERT pre-trains with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). NSP feeds $C$ to a binary classifier predicting whether sentence B follows sentence A. This is what pre-trains the pooler's $W$ and $b$, teaching it to encode inter-sentence coherence so it arrives at fine-tuning already producing useful representations.</span>

### <span style="font-size: 14px;">Fine-tuning for Classification</span>

<span style="font-size: 14px;">During fine-tuning, $C$ feeds a newly initialized layer $W_c \in \mathbb{R}^{K \times H}$ ($K$ = number of labels). The entire model is fine-tuned end-to-end. The paper reports results on SST-2, MRPC, QNLI, and other GLUE tasks using the pooler output.</span>

### <span style="font-size: 14px;">NSP's Influence on Pooler Quality</span>

<span style="font-size: 14px;">RoBERTa (Liu et al., 2019) showed removing NSP does not hurt and can help, raising questions about whether the pooler's NSP pre-training provides useful initialization or whether weights are re-learned during fine-tuning. Models dropping NSP (RoBERTa, ALBERT) include a pooler but it is not meaningfully pre-trained.</span>

---

## <span style="font-size: 16px;">Sequence vs Token Classification</span>

<span style="font-size: 14px;">BERT handles two fundamentally different classification paradigms, and the pooler is relevant to only one.</span>

### <span style="font-size: 14px;">Sequence Classification</span>

<span style="font-size: 14px;">The entire input receives a single label. The pooler extracts the [CLS] representation and the classifier produces one set of logits per sequence.</span>

* <span style="font-size: 14px;">**Sentiment analysis:** "This movie was fantastic" maps to Positive.</span>
* <span style="font-size: 14px;">**Natural Language Inference:** Classify premise-hypothesis pair as entailment, contradiction, or neutral.</span>
* <span style="font-size: 14px;">**Paraphrase detection:** Determine whether two sentences mean the same thing.</span>
* <span style="font-size: 14px;">**Next Sentence Prediction:** The pre-training task itself.</span>

<span style="font-size: 14px;">Pipeline: encoder output at position 0, pooler (dense + tanh), classifier (dense + softmax). Only the [CLS] hidden state matters; all other token representations are discarded.</span>

### <span style="font-size: 14px;">Token Classification</span>

<span style="font-size: 14px;">Every token receives its own label. The pooler is not used. Every hidden state is individually passed through a classification head.</span>

* <span style="font-size: 14px;">**Named Entity Recognition (NER):** Each token labeled as Person, Organization, Location, or Other.</span>
* <span style="font-size: 14px;">**Part-of-Speech tagging:** Each token labeled as noun, verb, adjective, etc.</span>
* <span style="font-size: 14px;">**Slot filling:** Each token tagged with its semantic role in dialogue systems.</span>

<span style="font-size: 14px;">For token classification: $\text{logits}_i = h_i \cdot W_t + b_t$ for each token $i$. The pooler is bypassed entirely. Using it for token tasks collapses all information into one vector.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Toy example with $H = 4$ (BERT-Base uses $H = 768$).</span>

### <span style="font-size: 14px;">Step 1: Extract [CLS]</span>

<span style="font-size: 14px;">Encoder outputs for a 3-token sequence. The pooler takes only $h_0$:</span>

$$
h_0 = [0.8, -0.3, 1.2, -0.5] \quad \text{([CLS])}, \quad h_1 = [0.1, 0.9, -0.4, 0.6], \quad h_2 = [-0.2, 0.5, 0.7, -0.1]
$$

### <span style="font-size: 14px;">Step 2: Linear Projection</span>

$$
W = \begin{bmatrix} 0.5 & 0.1 & -0.3 & 0.2 \\ -0.1 & 0.4 & 0.6 & -0.2 \\ 0.3 & -0.5 & 0.2 & 0.7 \\ 0.2 & 0.3 & -0.1 & 0.4 \end{bmatrix}, \quad b = [0.1, 0.0, -0.1, 0.05]
$$

<span style="font-size: 14px;">$z = h_0 \cdot W + b$. Working out each component:</span>

$$
z_0 = (0.8)(0.5) + (-0.3)(-0.1) + (1.2)(0.3) + (-0.5)(0.2) + 0.1 = 0.79
$$

$$
z_1 = (0.8)(0.1) + (-0.3)(0.4) + (1.2)(-0.5) + (-0.5)(0.3) + 0.0 = -0.79
$$

$$
z_2 = (0.8)(-0.3) + (-0.3)(0.6) + (1.2)(0.2) + (-0.5)(-0.1) - 0.1 = -0.23
$$

$$
z_3 = (0.8)(0.2) + (-0.3)(-0.2) + (1.2)(0.7) + (-0.5)(0.4) + 0.05 = 0.91
$$

### <span style="font-size: 14px;">Step 3: Tanh</span>

$$
\tanh(0.79) = \frac{e^{0.79} - e^{-0.79}}{e^{0.79} + e^{-0.79}} = \frac{2.2034 - 0.4538}{2.2034 + 0.4538} \approx 0.659
$$

$$
\tanh(-0.79) \approx -0.659, \quad \tanh(-0.23) \approx -0.226, \quad \tanh(0.91) \approx 0.723
$$

$$
h_{\text{pooled}} = [0.659, -0.659, -0.226, 0.723]
$$

<span style="font-size: 14px;">All values bounded in $[-1, 1]$. Tanh compressed magnitudes while preserving signs.</span>

### <span style="font-size: 14px;">Step 4: Classifier</span>

<span style="font-size: 14px;">Binary sentiment classifier with $W_c \in \mathbb{R}^{4 \times 2}$, $b_c \in \mathbb{R}^2$:</span>

$$
W_c = \begin{bmatrix} 0.6 & -0.4 \\ -0.3 & 0.5 \\ 0.2 & -0.1 \\ 0.4 & -0.6 \end{bmatrix}, \quad b_c = [0.1, -0.1]
$$

$$
\text{logit}_0 = (0.659)(0.6) + (-0.659)(-0.3) + (-0.226)(0.2) + (0.723)(0.4) + 0.1 = 0.937
$$

$$
\text{logit}_1 = (0.659)(-0.4) + (-0.659)(0.5) + (-0.226)(-0.1) + (0.723)(-0.6) - 0.1 = -1.104
$$

<span style="font-size: 14px;">Softmax: $P(\text{Positive}) = \frac{e^{0.937}}{e^{0.937} + e^{-1.104}} \approx \frac{2.552}{2.552 + 0.332} \approx 0.885$. The model predicts Positive with 88.5% confidence.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">Since BERT's publication, the community has revisited and often moved away from [CLS] pooling.</span>

### <span style="font-size: 14px;">Mean Pooling Often Outperforms [CLS]</span>

<span style="font-size: 14px;">Sentence-BERT (Reimers and Gurevych, 2019) showed mean pooling produces better sentence embeddings than [CLS] for semantic similarity. Averaging distributes the representation across all tokens rather than relying on one position. Mean pooling became the default in Sentence Transformers and models like all-MiniLM.</span>

### <span style="font-size: 14px;">Some Models Drop the Pooler Entirely</span>

<span style="font-size: 14px;">RoBERTa removed NSP, so its pooler is not meaningfully pre-trained. Practitioners bypass it and use raw [CLS] or mean pooling. DistilBERT omits the pooler entirely.</span>

### <span style="font-size: 14px;">Decoder-Only Models Use Last Token</span>

<span style="font-size: 14px;">GPT-style models use causal attention where each position only attends to previous ones. Only the last token has seen the full input, so decoder-only models use last-token pooling, the mirror image of BERT's first-token approach.</span>

### <span style="font-size: 14px;">Contrastive Learning Changes the Landscape</span>

<span style="font-size: 14px;">SimCSE (Gao et al., 2021) showed representation quality depends more on the training objective than pooling strategy. With contrastive pre-training, even [CLS] pooling produces excellent embeddings. What matters is the objective, not the pooling mechanism.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

### <span style="font-size: 14px;">Extracting the Wrong Position</span>

<span style="font-size: 14px;">[CLS] is always at position 0. A common mistake is using position -1 (last token). The extraction should be `hidden_states[:, 0, :]`. Position -1 gives [SEP] or a padding token, neither carrying the right information.</span>

### <span style="font-size: 14px;">Forgetting the Tanh Activation</span>

<span style="font-size: 14px;">Implementing the pooler without tanh changes the output distribution. The representation becomes unbounded, mismatching pre-trained weights that expect inputs in $[-1, 1]$.</span>

### <span style="font-size: 14px;">Wrong Weight Matrix Shape</span>

<span style="font-size: 14px;">The pooler's weight matrix must be $H \times H$ (square). A common error is creating $H \times C$ where $C$ is the number of classes, conflating pooler with classifier. They are separate: the pooler projects $H \to H$ with tanh, then the classifier projects $H \to C$ without tanh.</span>

### <span style="font-size: 14px;">Using the Pooler for Token Classification</span>

<span style="font-size: 14px;">The pooler produces one vector for the entire sequence. For token-level tasks (NER, POS tagging), you need per-token representations. Token classification should use the full hidden state sequence $h_0, h_1, \ldots, h_{T-1}$ directly, bypassing the pooler.</span>

### <span style="font-size: 14px;">Pooler Not Pre-trained When NSP Is Skipped</span>

<span style="font-size: 14px;">Models pre-trained without NSP (RoBERTa) have untrained pooler weights. Options: (1) use raw [CLS] instead, (2) use mean pooling, or (3) accept that the pooler trains from scratch during fine-tuning.</span>

### <span style="font-size: 14px;">Confusing Pooler Output with Raw Hidden State</span>

<span style="font-size: 14px;">In HuggingFace, `outputs.last_hidden_state[:, 0]` is the raw [CLS] hidden state before the pooler; `outputs.pooler_output` is after dense + tanh. These are different tensors with different distributions. Using the wrong one causes subtle bugs.</span>

---