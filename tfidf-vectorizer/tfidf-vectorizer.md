## What is TF-IDF?

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document within a collection. It combines two factors: how frequently a term appears in a document (TF) and how rare the term is across all documents (IDF). Words that are frequent in a document but rare overall receive high scores.

---

## The Intuition Behind TF-IDF

**Problem with raw counts**: Common words like "the", "is", "and" appear frequently in all documents. Using raw counts gives them high importance despite carrying little meaning.

**TF component**: Captures local importance - a word mentioned many times in a document is likely important to that document.

**IDF component**: Captures global rarity - a word appearing in few documents is more discriminative than one appearing everywhere.

**Combined effect**: Balances frequency with distinctiveness.

---

## Term Frequency (TF)

Several variants exist:

**Raw count**:

$$
\text{tf}(t, d) = f_{t,d}
$$

Where $f_{t,d}$ is the number of times term $t$ appears in document $d$.

**Boolean**:

$$
\text{tf}(t, d) = \begin{cases} 1 & \text{if } t \in d \\ 0 & \text{otherwise} \end{cases}
$$

**Log normalization**:

$$
\text{tf}(t, d) = 1 + \log(f_{t,d}) \quad \text{if } f_{t,d} > 0
$$

**Augmented frequency** (prevents bias toward long documents):

$$
\text{tf}(t, d) = 0.5 + 0.5 \cdot \frac{f_{t,d}}{\max\{f_{t',d} : t' \in d\}}
$$

---

## Inverse Document Frequency (IDF)

Measures how common or rare a term is across documents:

$$
\text{idf}(t) = \log\left(\frac{N}{n_t}\right)
$$

Where:
- $N$ = total number of documents
- $n_t$ = number of documents containing term $t$

**Properties**:
- Terms in all documents: idf → 0
- Terms in one document: idf → log(N) (maximum)
- Rare terms get high IDF, common terms get low IDF

**Smoothed variant** (avoids division by zero):

$$
\text{idf}(t) = \log\left(\frac{N + 1}{n_t + 1}\right) + 1
$$

---

## The TF-IDF Formula

Combining TF and IDF:

$$
\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)
$$

**Interpretation**: High TF-IDF means the term is frequent in this document but rare across the corpus - likely a key term for this document.

---

## Worked Example

**Corpus** (3 documents):
- D1: "cat sat mat"
- D2: "dog ran park"
- D3: "cat dog play"

**Vocabulary**: [cat, dog, mat, park, play, ran, sat]

**Document Frequencies**:
- cat: 2 documents (D1, D3)
- dog: 2 documents (D2, D3)
- mat: 1 document (D1)
- park: 1 document (D2)
- play: 1 document (D3)
- ran: 1 document (D2)
- sat: 1 document (D1)

**IDF calculations** (using standard formula):

$$
\text{idf(cat)} = \log(3/2) \approx 0.405
$$

$$
\text{idf(mat)} = \log(3/1) \approx 1.099
$$

**TF-IDF for D1**:
- cat: tf=1, idf=0.405, tf-idf = 0.405
- sat: tf=1, idf=1.099, tf-idf = 1.099
- mat: tf=1, idf=1.099, tf-idf = 1.099

**Observation**: "mat" and "sat" have higher TF-IDF than "cat" because they are unique to D1.

---

## The TF-IDF Matrix

For $N$ documents and vocabulary of size $V$:

**Matrix shape**: $(N, V)$

**Each row**: TF-IDF vector for one document

**Each column**: TF-IDF values for one term across documents

**Sparsity**: Most entries are zero (documents do not contain most vocabulary terms)

---

## Vocabulary Building

**Step 1**: Tokenize all documents into terms

**Step 2**: Build vocabulary (unique terms)

**Step 3**: Optionally filter vocabulary:
- Remove terms appearing in too few documents (min_df)
- Remove terms appearing in too many documents (max_df)
- Limit vocabulary size (max_features)

**Result**: Mapping from term to column index

---

## Fitting and Transforming

**Fit** (learn from training corpus):
- Build vocabulary
- Compute IDF values for each term
- Store vocabulary and IDF weights

**Transform** (apply to documents):
- Count term frequencies in document
- Multiply by stored IDF values
- Return TF-IDF vector

**Fit-Transform**: Combines both on training data

**Important**: Use same vocabulary and IDF values for training and test data.

---

## L2 Normalization

Often applied after TF-IDF:

$$
\text{normalized}(d) = \frac{\text{tf-idf}(d)}{||\text{tf-idf}(d)||_2}
$$

**Benefits**:
- Document vectors have unit length
- Cosine similarity becomes dot product
- Removes bias from document length

---

## N-gram Extension

Instead of single words, consider sequences:

**Unigrams**: "the", "cat", "sat"

**Bigrams**: "the cat", "cat sat"

**Trigrams**: "the cat sat"

**Combined**: Often use (1,2) or (1,3) range to capture both words and phrases

**Tradeoff**: More features, potentially better representation, but higher dimensionality

---

## TF-IDF vs Word Embeddings

**TF-IDF**:
- Sparse, high-dimensional vectors
- No semantic similarity (synonyms are different dimensions)
- Interpretable (each dimension is a word)
- No training required

**Word embeddings (Word2Vec, GloVe)**:
- Dense, low-dimensional vectors
- Capture semantic similarity
- Less interpretable
- Require pre-training or training

**Modern approach**: Often combine both or use transformer embeddings

---

## Handling Out-of-Vocabulary Terms

Terms in test documents but not in training vocabulary:

**Standard approach**: Ignore them (they contribute nothing to the vector)

**Implication**: New documents with mostly OOV terms get sparse representations

**Solutions**:
- Use subword tokenization
- Include rare term handling in preprocessing
- Use character n-grams

---

## Where TF-IDF Shows Up

- **Search Engines**: Ranking documents by query relevance

- **Text Classification**: Feature vectors for document categorization

- **Document Clustering**: Grouping similar documents

- **Keyword Extraction**: Terms with highest TF-IDF are key terms

- **Duplicate Detection**: Comparing document similarity

- **Recommendation Systems**: Content-based filtering using text features

- **Information Retrieval**: Core component of many retrieval systems

- **Spam Detection**: Features for email classification

- **Sentiment Analysis**: Baseline text representation

- **Topic Modeling**: Often used alongside or compared with LDA
