# <span style="font-size: 20px;">Fine-tuning Architecture</span>

<span style="font-size: 14px;">Fine-tuning is the second stage of BERT's two-stage paradigm (Devlin et al., 2019): first pre-train a deep bidirectional Transformer on unlabeled corpora, then fine-tune the entire model on a smaller labeled dataset for a specific downstream task. As the paper states: "Fine-tuning is straightforward since the self-attention mechanism in the Transformer allows BERT to model many downstream tasks. For each task, we simply plug in the task-specific inputs and outputs and fine-tune all the parameters end-to-end."</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">Fine-tuning takes the pre-trained BERT encoder and adds a small, randomly initialized task-specific layer (the "head") on top. The entire system is trained end-to-end on labeled data. This contrasts with feature-based approaches (like ELMo) where pre-trained representations are frozen.</span>

* <span style="font-size: 14px;">**Stage 1 (Pre-training):** BERT learns general language understanding from massive unlabeled text via masked language modeling and next sentence prediction, building deep contextual representations across 12 or 24 Transformer layers.</span>
* <span style="font-size: 14px;">**Stage 2 (Fine-tuning):** All parameters are updated on a small labeled dataset. The pre-trained weights provide an excellent initialization, so the model converges quickly with minimal data.</span>

<span style="font-size: 14px;">The task head is typically a single linear layer. For sequence classification it operates on the [CLS] hidden state; for token classification it operates on every token's hidden state independently. The head is the only component initialized from scratch.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $H \in \mathbb{R}^{L \times d}$ be the final encoder output, where $L$ is sequence length and $d$ is hidden dimension (768 for BERT-Base, 1024 for BERT-Large). $h_{\text{CLS}} \in \mathbb{R}^d$ is the [CLS] hidden state, and $h_i \in \mathbb{R}^d$ is the hidden state at position $i$.</span>

<span style="font-size: 14px;">**Sequence classification** uses only the [CLS] token's representation:</span>

$$
\text{logits} = h_{\text{CLS}} \cdot W + b
$$

<span style="font-size: 14px;">where $W \in \mathbb{R}^{d \times K}$ is the classifier weight matrix, $b \in \mathbb{R}^K$ is the bias vector, and $K$ is the number of classes.</span>

<span style="font-size: 14px;">**Token classification** applies the same linear transformation to every token independently:</span>

$$
\text{logits}_i = h_i \cdot W + b \quad \text{for } i = 1, 2, \ldots, L
$$

<span style="font-size: 14px;">where $W \in \mathbb{R}^{d \times K}$ and $b \in \mathbb{R}^K$ are shared across all positions.</span>

<span style="font-size: 14px;">**Probability distribution** via softmax:</span>

$$
P(y = k \mid h) = \frac{\exp(\text{logits}_k)}{\sum_{j=1}^{K} \exp(\text{logits}_j)}
$$

<span style="font-size: 14px;">**Training loss** is standard cross-entropy. For sequence classification with ground-truth class $c$:</span>

$$
\mathcal{L} = -\log P(y = c \mid h_{\text{CLS}})
$$

<span style="font-size: 14px;">For token classification, the loss sums over all token positions (ignoring padding and special tokens):</span>

$$
\mathcal{L} = -\sum_{i=1}^{L} \log P(y_i = c_i \mid h_i)
$$

<span style="font-size: 14px;">Gradients from this loss flow back through the entire encoder, adjusting all pre-trained parameters. This end-to-end gradient flow is what distinguishes fine-tuning from feature extraction.</span>

---

## <span style="font-size: 16px;">Sequence vs Token Classification</span>

<span style="font-size: 14px;">The same BERT encoder serves both tasks. The difference lies entirely in which hidden states the classifier head reads.</span>

<span style="font-size: 14px;">**Sequence classification** produces one prediction per input:</span>

* <span style="font-size: 14px;">**Input representation:** The [CLS] token aggregates information from the entire sequence into $h_{\text{CLS}}$ via self-attention across all layers.</span>
* <span style="font-size: 14px;">**Classifier:** A single linear layer maps $h_{\text{CLS}} \in \mathbb{R}^d$ to $\mathbb{R}^K$.</span>
* <span style="font-size: 14px;">**Tasks:** Sentiment analysis, natural language inference, topic classification, paraphrase detection.</span>

<span style="font-size: 14px;">**Token classification** produces one prediction per token:</span>

* <span style="font-size: 14px;">**Input representation:** Every token position's hidden state $h_i$ is used, not just [CLS].</span>
* <span style="font-size: 14px;">**Classifier:** The same linear layer is applied independently to each token's hidden state. Weights $W$ and $b$ are shared across positions.</span>
* <span style="font-size: 14px;">**Tasks:** Named entity recognition (NER), part-of-speech tagging, slot filling in dialog systems.</span>

<span style="font-size: 14px;">For sentence-pair tasks, BERT packs both sentences into one input: [CLS] sentence_A [SEP] sentence_B [SEP]. Segment embeddings distinguish which sentence each token belongs to, and [CLS] captures the cross-sentence relationship.</span>

---

## <span style="font-size: 16px;">Layer Freezing</span>

<span style="font-size: 14px;">Layer freezing disables gradient updates for selected parameters by setting `requires_grad = False`, so they retain pre-trained values throughout training.</span>

<span style="font-size: 14px;">**Why freeze layers:**</span>

* <span style="font-size: 14px;">**Prevent catastrophic forgetting:** On small datasets, updating all parameters risks overwriting the general linguistic knowledge learned during pre-training.</span>
* <span style="font-size: 14px;">**Reduce overfitting:** Fewer trainable parameters means lower model capacity relative to dataset size, acting as implicit regularization.</span>
* <span style="font-size: 14px;">**Speed up training:** Frozen layers skip gradient computation and weight updates, reducing memory and training time.</span>

<span style="font-size: 14px;">**Which layers to freeze:**</span>

* <span style="font-size: 14px;">**Bottom layers (0-3 in BERT-Base):** Capture general linguistic features like syntax and morphology. Transfer well across tasks and are safest to freeze.</span>
* <span style="font-size: 14px;">**Middle layers (4-7):** Capture intermediate representations. Whether to freeze depends on domain similarity to pre-training data.</span>
* <span style="font-size: 14px;">**Top layers (8-11):** Capture task-relevant, high-level semantic features. Benefit most from fine-tuning and should generally remain trainable.</span>
* <span style="font-size: 14px;">**Embedding layer:** Often frozen since subword embeddings are well-learned during pre-training.</span>

<span style="font-size: 14px;">**Guidelines by dataset size:**</span>

* <span style="font-size: 14px;">**Very small (< 1K examples):** Freeze all encoder layers, train only the classifier head.</span>
* <span style="font-size: 14px;">**Small (1K-10K examples):** Freeze bottom 6-8 layers, fine-tune top layers and head.</span>
* <span style="font-size: 14px;">**Medium (10K-100K examples):** Freeze bottom 2-4 layers or none. Use a small learning rate.</span>
* <span style="font-size: 14px;">**Large (100K+ examples):** Fine-tune all layers. Sufficient data prevents catastrophic forgetting.</span>

<span style="font-size: 14px;">Gradual unfreezing is a related technique: start with only the top layer unfrozen, train briefly, then unfreeze the next layer down and repeat.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">Devlin et al. (2019) introduced a fine-tuning recipe that became the standard. The recommended hyperparameters were deliberately conservative.</span>

<span style="font-size: 14px;">**Recommended hyperparameters:**</span>

* <span style="font-size: 14px;">**Learning rate:** 2e-5, 3e-5, 4e-5, or 5e-5 -- roughly 100x smaller than typical training-from-scratch rates.</span>
* <span style="font-size: 14px;">**Batch size:** 16 or 32.</span>
* <span style="font-size: 14px;">**Epochs:** 2, 3, or 4. Very few epochs needed because the model starts from strong representations.</span>
* <span style="font-size: 14px;">**Warmup:** Linear warmup over the first 10% of steps, then linear decay.</span>
* <span style="font-size: 14px;">**Dropout:** 0.1 on the classifier head.</span>

<span style="font-size: 14px;">**Key results from the paper:**</span>

* <span style="font-size: 14px;">**GLUE benchmark:** 80.5 accuracy, a 7.7 point improvement over prior state of the art.</span>
* <span style="font-size: 14px;">**SQuAD 1.1:** F1 of 93.2, surpassing human performance (91.2 F1).</span>
* <span style="font-size: 14px;">**SQuAD 2.0:** F1 of 83.1, handling unanswerable questions via [CLS].</span>
* <span style="font-size: 14px;">**CoNLL-2003 NER:** F1 of 92.8, showing the same model excels at both sequence-level and token-level tasks.</span>

<span style="font-size: 14px;">Fine-tuning outperformed feature-based approaches on most tasks, validating the end-to-end strategy.</span>

---

## <span style="font-size: 16px;">Catastrophic Forgetting</span>

<span style="font-size: 14px;">Catastrophic forgetting occurs when fine-tuning overwrites the general knowledge learned during pre-training. The model becomes specialized for the new task at the cost of losing broad linguistic understanding.</span>

<span style="font-size: 14px;">**Why small learning rates matter:**</span>

* <span style="font-size: 14px;">**Small updates preserve knowledge:** A rate of 2e-5 to 5e-5 nudges pre-trained weights rather than replacing them.</span>
* <span style="font-size: 14px;">**Large rates destroy representations:** At 1e-3 or higher, gradients rapidly overwrite attention patterns learned from billions of tokens.</span>

<span style="font-size: 14px;">**Why few epochs matter:**</span>

* <span style="font-size: 14px;">**Drift compounds:** Every step pushes parameters further from pre-trained values. Many epochs on small data cause substantial cumulative drift.</span>

<span style="font-size: 14px;">**Connection to layer freezing:**</span>

* <span style="font-size: 14px;">**Freezing as direct remedy:** If parameters do not update, they cannot forget.</span>
* <span style="font-size: 14px;">**Complementary strategies:** Layer freezing, small learning rates, few epochs, and warmup address the same problem from different angles and are often combined.</span>

<span style="font-size: 14px;">**Discriminative learning rates** offer a middle ground: assign smaller rates to lower layers and larger rates to upper layers. For example, layer 0 uses 1e-6, layer 6 uses 1e-5, layer 11 uses 5e-5.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider 3-class sentiment classification (negative, neutral, positive) with hidden dimension $d = 4$.</span>

<span style="font-size: 14px;">**Step 1: Encoder output.** After passing "[CLS] This movie is great [SEP]" through BERT, the [CLS] hidden state is:</span>

$$
h_{\text{CLS}} = [0.5, -0.3, 0.8, 0.1]
$$

<span style="font-size: 14px;">**Step 2: Classifier weights.** $W \in \mathbb{R}^{4 \times 3}$ and $b \in \mathbb{R}^3$:</span>

$$
W = \begin{bmatrix} 0.2 & -0.1 & 0.4 \\ 0.3 & 0.5 & -0.2 \\ -0.1 & 0.2 & 0.6 \\ 0.4 & -0.3 & 0.1 \end{bmatrix}, \quad b = [0.1, -0.1, 0.2]
$$

<span style="font-size: 14px;">**Step 3: Compute logits.** $\text{logits} = h_{\text{CLS}} \cdot W + b$:</span>

$$
\text{logits}_0 = (0.5)(0.2) + (-0.3)(0.3) + (0.8)(-0.1) + (0.1)(0.4) + 0.1 = 0.07
$$

$$
\text{logits}_1 = (0.5)(-0.1) + (-0.3)(0.5) + (0.8)(0.2) + (0.1)(-0.3) - 0.1 = -0.17
$$

$$
\text{logits}_2 = (0.5)(0.4) + (-0.3)(-0.2) + (0.8)(0.6) + (0.1)(0.1) + 0.2 = 0.95
$$

$$
\text{logits} = [0.07, -0.17, 0.95]
$$

<span style="font-size: 14px;">**Step 4: Softmax.**</span>

$$
\exp(\text{logits}) = [e^{0.07}, e^{-0.17}, e^{0.95}] = [1.073, 0.844, 2.586]
$$

$$
\text{sum} = 1.073 + 0.844 + 2.586 = 4.503
$$

$$
P = [0.238, 0.187, 0.574]
$$

<span style="font-size: 14px;">The model predicts class 2 (positive) with 57.4% probability, matching "This movie is great."</span>

<span style="font-size: 14px;">**Step 5: Loss.** With ground-truth label class 2:</span>

$$
\mathcal{L} = -\log(0.574) = 0.555
$$

<span style="font-size: 14px;">This loss backpropagates through the classifier head and the entire BERT encoder.</span>

### <span style="font-size: 14px;">Token Classification Example</span>

<span style="font-size: 14px;">NER on "John works at Google" with labels {O, PER, ORG} ($K = 3$). The same $W$ and $b$ are applied to every token's hidden state independently.</span>

<span style="font-size: 14px;">After encoding:</span>

$$
h_{\text{John}} = [0.9, -0.2, 0.4, 0.3], \quad h_{\text{works}} = [0.1, 0.5, -0.1, 0.2], \quad h_{\text{Google}} = [0.7, 0.1, 0.6, -0.4]
$$

<span style="font-size: 14px;">**"John":** $\text{logits} = h_{\text{John}} \cdot W + b = [0.30, -0.30, 0.87]$. Softmax: $[0.243, 0.133, 0.624]$. Prediction: ORG. With random weights this is incorrect -- a trained model would predict PER.</span>

<span style="font-size: 14px;">**"works":** $\text{logits} = [0.36, 0.06, 0.10]$. Softmax: $[0.402, 0.298, 0.310]$. Prediction: O. Correct -- "works" is not an entity.</span>

<span style="font-size: 14px;">**"Google":** $\text{logits} = [0.05, 0.12, 0.78]$. Softmax: $[0.225, 0.242, 0.533]$. Prediction: ORG. Correct -- "Google" is an organization.</span>

<span style="font-size: 14px;">The total loss sums cross-entropy across all token positions.</span>

---

## <span style="font-size: 16px;">Modern Context</span>

<span style="font-size: 14px;">As models grew to hundreds of billions of parameters, full fine-tuning became impractical, driving parameter-efficient alternatives.</span>

* <span style="font-size: 14px;">**LoRA (Low-Rank Adaptation):** Freezes pre-trained weights and injects trainable low-rank matrices. The update is $\Delta W = AB$ with $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$, $r \ll d$. Reduces trainable parameters by 1000x while matching full fine-tuning on many tasks.</span>
* <span style="font-size: 14px;">**Adapter layers:** Small bottleneck modules inserted between Transformer layers. Only adapters train; the base model is frozen.</span>
* <span style="font-size: 14px;">**Prompt tuning:** Learnable continuous vectors prepended to input embeddings. The model is completely frozen; only prompt vectors are optimized.</span>
* <span style="font-size: 14px;">**In-context learning (ICL):** Introduced by GPT-3. Task demonstrations given in the input prompt; the model performs zero-shot or few-shot without parameter updates.</span>
* <span style="font-size: 14px;">**QLoRA:** Combines 4-bit quantization with LoRA, enabling fine-tuning of 65B+ parameter models on a single GPU.</span>

<span style="font-size: 14px;">Despite these advances, full fine-tuning remains the gold standard when maximum performance is needed. BERT's core insight -- that pre-trained representations can be adapted with minimal architecture changes -- underpins all of these methods.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Learning rate too high:** Using 1e-3 or higher rapidly destroys pre-trained representations. Always use 2e-5 to 5e-5.</span>
* <span style="font-size: 14px;">**Too many epochs on small data:** Training 10+ epochs on a few thousand examples causes severe overfitting and catastrophic forgetting. Stick to 2-4 epochs.</span>
* <span style="font-size: 14px;">**Forgetting to use [CLS] for sequence tasks:** Averaging all token states or using the last token misaligns with pre-training, which used [CLS] as the aggregate representation.</span>
* <span style="font-size: 14px;">**Applying sequence head to token tasks:** Using only [CLS] for NER or POS tagging discards per-token information. Token classification requires every token's hidden state.</span>
* <span style="font-size: 14px;">**Skipping warmup:** The randomly initialized head produces large, noisy gradients that can permanently damage pre-trained representations. Use warmup for 5-10% of steps.</span>
* <span style="font-size: 14px;">**Freezing too many layers:** Training only the head limits adaptation. Underperforms on domain-specific tasks.</span>
* <span style="font-size: 14px;">**Freezing too few layers on small data:** Fine-tuning all 12 layers on 500 examples gives too many degrees of freedom. Freeze the bottom 8-10 layers.</span>
* <span style="font-size: 14px;">**Ignoring input formatting:** Omitting [CLS]/[SEP], wrong segment IDs, or exceeding 512 tokens produces degraded results without an obvious error.</span>

---