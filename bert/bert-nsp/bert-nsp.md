# <span style="font-size: 20px;">Next Sentence Prediction</span>

<span style="font-size: 14px;">Next Sentence Prediction (NSP) is the second of BERT's two pre-training objectives. While Masked Language Modeling (MLM) teaches the model to understand token-level context, NSP trains the model to understand relationships between sentences. Given a pair (A, B), the model predicts whether B is the actual next sentence following A in the corpus (IsNext) or a randomly sampled sentence (NotNext). It was introduced in Devlin et al. (2019), "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."</span>

---

## <span style="font-size: 16px;">What It Is / What It Does</span>

<span style="font-size: 14px;">NSP is a binary classification task applied to sentence pairs during BERT's pre-training. The model receives two segments separated by a [SEP] token, with a [CLS] token prepended. The final hidden state of [CLS] is fed through a linear classification head to produce a binary prediction.</span>

<span style="font-size: 14px;">The two classes are:</span>

* <span style="font-size: 14px;">**IsNext (label = 1):** Sentence B is the real next sentence following sentence A in the source document. 50% of training pairs use this construction.</span>
* <span style="font-size: 14px;">**NotNext (label = 0):** Sentence B is randomly sampled from a different part of the corpus. The remaining 50% use this construction.</span>

<span style="font-size: 14px;">The balanced 50/50 split means the trivial baseline (always predicting one class) achieves 50% accuracy. BERT's pre-trained NSP classifier reaches approximately 97-98% accuracy, confirming the model learns meaningful inter-sentence relationships.</span>

<span style="font-size: 14px;">NSP trains jointly with MLM. Every training example simultaneously has masked tokens (for MLM) and a sentence-pair label (for NSP). The total pre-training loss is the sum of the MLM loss and the NSP loss.</span>

---

## <span style="font-size: 16px;">Key Equations</span>

<span style="font-size: 14px;">Let $h_{\text{CLS}} \in \mathbb{R}^h$ denote the final hidden state of the [CLS] token from the last Transformer layer, where $h$ is the hidden dimension (768 for BERT-Base, 1024 for BERT-Large).</span>

<span style="font-size: 14px;">The NSP head applies a linear transformation followed by softmax:</span>

$$
\text{logits} = h_{\text{CLS}} \cdot W + b
$$

<span style="font-size: 14px;">where $W \in \mathbb{R}^{h \times 2}$ is the weight matrix and $b \in \mathbb{R}^2$ is the bias vector. This produces two logits for the IsNext and NotNext classes.</span>

<span style="font-size: 14px;">The probability distribution over the two classes via softmax:</span>

$$
P(\text{IsNext}) = \frac{e^{\text{logits}_1}}{e^{\text{logits}_0} + e^{\text{logits}_1}}
$$

$$
P(\text{NotNext}) = \frac{e^{\text{logits}_0}}{e^{\text{logits}_0} + e^{\text{logits}_1}}
$$

<span style="font-size: 14px;">The NSP loss is standard cross-entropy:</span>

$$
\mathcal{L}_{\text{NSP}} = -\left[ y \log P(\text{IsNext}) + (1 - y) \log P(\text{NotNext}) \right]
$$

<span style="font-size: 14px;">where $y = 1$ for IsNext pairs and $y = 0$ for NotNext pairs.</span>

<span style="font-size: 14px;">The total BERT pre-training loss combines both objectives:</span>

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

<span style="font-size: 14px;">Both losses are weighted equally (weight = 1.0) in the original implementation. There is no scaling factor between them.</span>

---

## <span style="font-size: 16px;">How NSP Training Data is Generated</span>

<span style="font-size: 14px;">BERT constructs training data at the sentence-pair level. Two "sentences" (actually spans of contiguous text, which may contain multiple linguistic sentences) are selected to form each input pair.</span>

<span style="font-size: 14px;">The generation procedure:</span>

* <span style="font-size: 14px;">**Step 1:** Choose a random document from the corpus.</span>
* <span style="font-size: 14px;">**Step 2:** Choose a random position in the document and select a text span as sentence A.</span>
* <span style="font-size: 14px;">**Step 3 (IsNext, 50%):** Take the span immediately following A in the same document as sentence B. Label as IsNext (y = 1).</span>
* <span style="font-size: 14px;">**Step 3 (NotNext, 50%):** Sample a random span from a different document as sentence B. Label as NotNext (y = 0).</span>

<span style="font-size: 14px;">The resulting input is formatted with special tokens:</span>

$$
[\text{CLS}] \; A_1 \; A_2 \; \ldots \; A_n \; [\text{SEP}] \; B_1 \; B_2 \; \ldots \; B_m \; [\text{SEP}]
$$

<span style="font-size: 14px;">Three types of embeddings are summed for each token position:</span>

* <span style="font-size: 14px;">**Token embeddings:** The WordPiece embedding for the actual token.</span>
* <span style="font-size: 14px;">**Segment embeddings:** $E_A$ for all tokens in sentence A (including [CLS] and the first [SEP]), $E_B$ for all tokens in sentence B (including the final [SEP]).</span>
* <span style="font-size: 14px;">**Position embeddings:** Learned absolute position embeddings from 0 to 511.</span>

<span style="font-size: 14px;">The combined sequence length (A + B + special tokens) must not exceed 512 tokens.</span>

---

## <span style="font-size: 16px;">Why NSP Was Included</span>

<span style="font-size: 14px;">Devlin et al. included NSP because many downstream NLP tasks require understanding relationships between two pieces of text. Tasks that benefit:</span>

* <span style="font-size: 14px;">**Natural Language Inference (NLI):** Given a premise and hypothesis, determine entailment, contradiction, or neutrality.</span>
* <span style="font-size: 14px;">**Question Answering (QA):** Given a question and passage, locate the answer span. Requires cross-sentence reasoning.</span>
* <span style="font-size: 14px;">**Paraphrase Detection:** Given two sentences, determine if they express the same meaning.</span>

<span style="font-size: 14px;">The paper states: "Many important downstream tasks such as Question Answering and Natural Language Inference are based on understanding the relationship between two sentences, which is not directly captured by language modeling."</span>

<span style="font-size: 14px;">NSP forces the [CLS] token to encode a global, cross-sentence representation that captures whether two segments are coherently related. This representation can then be fine-tuned for sentence-pair tasks.</span>

<span style="font-size: 14px;">Devlin et al. reported ablation results showing that removing NSP hurt performance on QNLI (86.4% to 84.3%), MNLI (84.4% to 83.7%), and SQuAD (88.5 F1 to 87.3 F1), providing direct evidence that NSP pre-training transferred useful knowledge.</span>

---

## <span style="font-size: 16px;">Paper Context</span>

<span style="font-size: 14px;">In the original BERT paper, NSP is introduced alongside MLM as one of two complementary pre-training objectives. No prior approach had captured inter-sentence relationships, making NSP a novel contribution.</span>

<span style="font-size: 14px;">Key details from the paper:</span>

* <span style="font-size: 14px;">**Pre-training accuracy:** The NSP classifier reaches 97-98% accuracy. This high accuracy indicates the task is learnable but foreshadows later criticism that it may be too easy.</span>
* <span style="font-size: 14px;">**Joint training:** MLM and NSP are trained simultaneously on the same input. Every training example has both masked tokens and an IsNext/NotNext label. The two loss terms are summed without any balancing coefficient.</span>
* <span style="font-size: 14px;">**The [CLS] token:** The final hidden state of [CLS] serves as the aggregate sequence representation for NSP during pre-training and for classification tasks during fine-tuning. This dual role is critical to BERT's design.</span>
* <span style="font-size: 14px;">**The pooler layer:** BERT includes a pooler that applies a dense layer with tanh activation to the [CLS] hidden state: $\text{pooled} = \tanh(W_p \cdot h_{\text{CLS}} + b_p)$. The NSP head operates on this pooled representation, not directly on the raw hidden state.</span>
* <span style="font-size: 14px;">**Pre-training data:** BERT is pre-trained on BooksCorpus (800M words) and English Wikipedia (2,500M words). Document-level corpora are essential because the model needs consecutive sentences within documents for IsNext pairs.</span>
* <span style="font-size: 14px;">**Training duration:** BERT-Base is pre-trained for 1M steps with batch size 256. Each sequence contains a sentence pair, so the model sees 256M sentence pairs total.</span>

---

## <span style="font-size: 16px;">The NSP Controversy</span>

<span style="font-size: 14px;">While NSP was presented as beneficial in the original paper, subsequent research challenged its utility significantly.</span>

### <span style="font-size: 14px;">RoBERTa (Liu et al., 2019)</span>

<span style="font-size: 14px;">The most impactful challenge came from RoBERTa, which systematically evaluated NSP through controlled experiments:</span>

* <span style="font-size: 14px;">**SEGMENT-PAIR + NSP:** The original BERT setup with sentence pairs and NSP loss.</span>
* <span style="font-size: 14px;">**SENTENCE-PAIR + NSP:** Using individual sentences (not spans) with NSP loss.</span>
* <span style="font-size: 14px;">**FULL-SENTENCES (no NSP):** Packing full sentences from one or more documents without NSP.</span>
* <span style="font-size: 14px;">**DOC-SENTENCES (no NSP):** Similar but not crossing document boundaries.</span>

<span style="font-size: 14px;">RoBERTa found that removing NSP consistently improved or matched performance. The FULL-SENTENCES configuration without NSP outperformed the original BERT setup on SQuAD, MNLI, SST-2, and RACE.</span>

### <span style="font-size: 14px;">Why NSP Might Hurt</span>

* <span style="font-size: 14px;">**The task is too easy:** NotNext pairs come from random documents with entirely different topics. The model solves NSP by detecting topic shifts rather than learning discourse coherence. A simple topic-matching classifier achieves near-perfect accuracy.</span>
* <span style="font-size: 14px;">**MLM signal pollution:** NotNext pairs produce incoherent concatenated inputs. The MLM objective applied to these inputs may confuse the model, which tries to predict masked tokens in text with no natural flow.</span>
* <span style="font-size: 14px;">**Reduced MLM context:** Using sentence pairs means shorter effective context for MLM. Without NSP, the same input length can be filled with a single long passage, providing richer context for masked token prediction.</span>

### <span style="font-size: 14px;">ALBERT and Sentence Order Prediction (SOP)</span>

<span style="font-size: 14px;">Lan et al. (2020) proposed ALBERT, replacing NSP with Sentence Order Prediction (SOP). SOP uses two consecutive sentences but swaps their order for negatives instead of using random sentences. This forces learning discourse coherence rather than topic detection. ALBERT showed SOP consistently outperformed NSP, confirming the problem was NSP's trivial negative sampling rather than the idea of sentence-level pre-training.</span>

---

## <span style="font-size: 16px;">Numerical Example</span>

<span style="font-size: 14px;">Consider two sentences from a document about climate:</span>

* <span style="font-size: 14px;">**Sentence A:** "Global temperatures have risen by 1.1 degrees Celsius since pre-industrial times."</span>
* <span style="font-size: 14px;">**Sentence B (IsNext):** "This warming is primarily driven by greenhouse gas emissions."</span>
* <span style="font-size: 14px;">**Sentence B' (NotNext):** "The stock market closed at a record high yesterday."</span>

### <span style="font-size: 14px;">IsNext Pair Construction</span>

<span style="font-size: 14px;">The tokenized input (simplified):</span>

$$
[\text{CLS}] \; \text{Global} \; \text{temperatures} \; \text{have} \; \text{risen} \; \ldots \; [\text{SEP}] \; \text{This} \; \text{warming} \; \text{is} \; \ldots \; [\text{SEP}]
$$

* <span style="font-size: 14px;">**segment_ids:** [0, 0, 0, 0, 0, ..., 0, 1, 1, 1, ..., 1]</span>
* <span style="font-size: 14px;">**Segment 0** covers [CLS] through the first [SEP] (sentence A).</span>
* <span style="font-size: 14px;">**Segment 1** covers sentence B through the final [SEP].</span>
* <span style="font-size: 14px;">**Label:** y = 1 (IsNext)</span>

### <span style="font-size: 14px;">NotNext Pair Construction</span>

<span style="font-size: 14px;">Replace sentence B with the random sentence B':</span>

$$
[\text{CLS}] \; \text{Global} \; \text{temperatures} \; \text{have} \; \text{risen} \; \ldots \; [\text{SEP}] \; \text{The} \; \text{stock} \; \text{market} \; \ldots \; [\text{SEP}]
$$

* <span style="font-size: 14px;">**segment_ids:** [0, 0, 0, 0, 0, ..., 0, 1, 1, 1, ..., 1]</span>
* <span style="font-size: 14px;">**Label:** y = 0 (NotNext)</span>

### <span style="font-size: 14px;">Forward Pass Through NSP Head</span>

<span style="font-size: 14px;">Assume BERT-Base ($h = 768$). After the 12 Transformer layers, extract $h_{\text{CLS}} \in \mathbb{R}^{768}$ at position 0.</span>

<span style="font-size: 14px;">First, the pooler transforms $h_{\text{CLS}}$:</span>

$$
h_{\text{pooled}} = \tanh(W_p \cdot h_{\text{CLS}} + b_p) \in \mathbb{R}^{768}
$$

<span style="font-size: 14px;">Then the NSP head produces logits:</span>

$$
\text{logits} = h_{\text{pooled}} \cdot W + b \in \mathbb{R}^2
$$

<span style="font-size: 14px;">where $W \in \mathbb{R}^{768 \times 2}$ and $b \in \mathbb{R}^2$.</span>

<span style="font-size: 14px;">Suppose for the IsNext pair, the logits are:</span>

$$
\text{logits} = [-1.2, \; 2.8]
$$

<span style="font-size: 14px;">Applying softmax:</span>

$$
P(\text{NotNext}) = \frac{e^{-1.2}}{e^{-1.2} + e^{2.8}} = \frac{0.301}{0.301 + 16.445} = \frac{0.301}{16.746} = 0.018
$$

$$
P(\text{IsNext}) = \frac{e^{2.8}}{e^{-1.2} + e^{2.8}} = \frac{16.445}{16.746} = 0.982
$$

<span style="font-size: 14px;">The model predicts IsNext with 98.2% confidence. The cross-entropy loss with $y = 1$:</span>

$$
\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext}) = -\log(0.982) = 0.018
$$

<span style="font-size: 14px;">Now suppose for the NotNext pair, the logits are:</span>

$$
\text{logits} = [3.1, \; -0.9]
$$

$$
P(\text{NotNext}) = \frac{e^{3.1}}{e^{3.1} + e^{-0.9}} = \frac{22.198}{22.198 + 0.407} = 0.982
$$

$$
\mathcal{L}_{\text{NSP}} = -\log P(\text{NotNext}) = -\log(0.982) = 0.018
$$

<span style="font-size: 14px;">Both examples show low loss, confirming the model correctly distinguishes IsNext from NotNext pairs.</span>

---

## <span style="font-size: 16px;">Pitfalls</span>

* <span style="font-size: 14px;">**Using the wrong token position:** The NSP head must use the hidden state at position 0 (the [CLS] token). Using the final token, averaging all positions, or using a [SEP] hidden state will produce incorrect results.</span>

* <span style="font-size: 14px;">**Wrong output dimensionality:** The NSP head outputs 2 logits (NotNext and IsNext), not 1. BERT uses a two-class softmax, not a single-output sigmoid. The weight matrix $W$ has shape $(h, 2)$, not $(h, 1)$.</span>

* <span style="font-size: 14px;">**NotNext pairs from the same document:** Original BERT samples NotNext sentences from different documents. Sampling from the same document produces harder negatives and does not match the setup. This is what ALBERT's SOP does intentionally.</span>

* <span style="font-size: 14px;">**Forgetting segment embeddings:** Segment embeddings ($E_A$ and $E_B$) are essential for NSP. Without them, the model has no signal distinguishing sentence A from B. The [SEP] token alone is not sufficient. Omitting them silently degrades performance.</span>

* <span style="font-size: 14px;">**Ignoring the pooler layer:** The NSP head operates on the pooler output, not directly on $h_{\text{CLS}}$. The pooler applies a dense + tanh transformation. Skipping it changes the effective architecture.</span>

* <span style="font-size: 14px;">**Assuming NSP loss needs a weighting factor:** In original BERT, the total loss is $\mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$ with no scaling. Adding a balancing coefficient departs from the original setup.</span>

* <span style="font-size: 14px;">**Conflating "sentence" with linguistic sentence:** In BERT's data generation, a "sentence" is a contiguous text span that can contain multiple linguistic sentences. Splitting on sentence boundaries produces shorter spans than what BERT uses.</span>

* <span style="font-size: 14px;">**Overlooking the 50/50 balance:** Training data must maintain a 50/50 split between IsNext and NotNext. Deviating introduces class imbalance that shifts the model's prior and defeats the purpose of the objective.</span>

---