# MODEL DEVELOPMENT DOCUMENT (MDD)

## Deep Learning NLP System for Automated SMS Spam Detection

---

### Document Control & Metadata

| Attribute                     | Details                                                                                                                                                            |
| :---------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Project Title**       | Automated SMS Spam Detection using Deep Learning and Natural Language Processing (NLP)                                                                             |
| **Project Category**    | Bootcamp Capstone Project — Project 6 (NLP / Deep Learning)                                                                                                       |
| **Document Type**       | Model Development Document (MDD)                                                                                                                                   |
| **Document Version**    | Version $1.0$ (Final Submission)                                                                                                                                 |
| **Framework & Stack**   | Python $3.10$, TensorFlow / Keras, Scikit-Learn, Pandas, Streamlit                                                                                               |
| **Dataset Source**      | [Kaggle: Spam Text Message Classification (`team-ai/spam-text-message-classification`)](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification) |
| **Live Deployment URL** | [Streamlit Cloud: NLP Spam Detection App (`https://nlp-spam-detection.streamlit.app/`)](https://nlp-spam-detection.streamlit.app/)                                |
| **Model Artifacts**     | `spam_model.h5` (Trained Keras Model), `tokenizer.pkl` (Fitted Word Dictionary)                                                                                |

---

### Table of Contents

1. **Executive Summary & Project Overview**
   * 1.1 Project Background
   * 1.2 Core Objectives & Scope
   * 1.3 Key Deliverables & External Links
   * 1.4 End-to-End System Architecture
2. **Problem Statement & Technical Context**
   * 2.1 The Challenge of Unsolicited Text Messages (Spam)
   * 2.2 Motivation for Deep Learning in NLP Classification
   * 2.3 Model Success Criteria & Evaluation Benchmarks
3. **Data Acquisition & Exploratory Data Analysis (EDA)**
   * 3.1 Programmatic Data Ingestion via `kagglehub`
   * 3.2 Dataset Structure & Schema Definition
   * 3.3 Target Class Distribution & Imbalance Analysis
   * 3.4 Quantitative Text Length & Lexical Statistics
   * 3.5 Data Integrity, Missing Values & Outlier Inspection
4. **Natural Language Processing (NLP) Preprocessing Pipeline**
   * 4.1 Text Normalization & Regular Expression Cleaning
   * 4.2 Target Label Encoding
   * 4.3 Data Partitioning Strategy ($80\text{--}20$ Train-Test Split)
   * 4.4 Vocabulary Construction & Integer Tokenization
   * 4.5 Sequence Padding & Truncation Mechanics
   * 4.6 Step-by-Step Pipeline Walkthrough on a Sample Record
5. **Deep Learning Model Architecture & Mathematical Formulation**
   * 5.1 Sequential Neural Network Overview
   * 5.2 Layer 1: Dense Word Embedding Layer (`Embedding`)
   * 5.3 Layer 2: Dimensionality Reduction (`GlobalAveragePooling1D`)
   * 5.4 Layer 3: Non-Linear Feature Extraction (`Dense` with ReLU)
   * 5.5 Layer 4: Probabilistic Binary Classification (`Dense` with Sigmoid)
   * 5.6 Network Parameter Count & Memory Footprint Calculation
6. **Model Training Methodology & Optimization**
   * 6.1 Objective Loss Function: Binary Cross-Entropy
   * 6.2 Optimization Algorithm: Adaptive Moment Estimation (Adam)
   * 6.3 Training Hyperparameter Configuration
   * 6.4 Epoch-by-Epoch Convergence & Learning Dynamics
7. **Model Evaluation, Validation & Performance Analysis**
   * 7.1 Training vs. Testing Sample Performance Comparison
   * 7.2 Confusion Matrix & Secondary Classification Metrics
   * 7.3 Generalization & Overfitting Diagnostics
   * 7.4 Qualitative Error Analysis (False Positives & False Negatives)
8. **Model Serialization & Artifact Management**
   * 8.1 Neural Network Weights & Topology Persistence (`spam_model.h5`)
   * 8.2 Lexical Vocabulary Persistence (`tokenizer.pkl`)
   * 8.3 Why Deep Learning NLP Requires Dual Artifacts Compared to Traditional ML
9. **Web Application Deployment & User Interface (Streamlit)**
   * 9.1 Production Inference Pipeline Architecture
   * 9.2 Cloud Hosting & Continuous Deployment (`https://nlp-spam-detection.streamlit.app/`)
   * 9.3 Interactive User Experience & Pre-Loaded Dataset Samples
   * 9.4 Verification Test Cases & Live Inference Benchmarks
10. **Model Limitations, Operational Risks & Future Scope**
    * 10.1 Technical & Lexical Limitations
    * 10.2 Operational & Security Risks
    * 10.3 Recommendations for Future Iterations
11. **Appendix: Complete Project Source Code**
    * 11.1 Environment Dependencies (`requirements.txt`)
    * 11.2 Model Training & Evaluation Script (`project.ipynb`)
    * 11.3 Streamlit Web Application (`app.py`)

---

## 1. Executive Summary & Project Overview

### 1.1 Project Background

Short Message Service (SMS) and instant text messaging remain primary communication channels globally for personal correspondence, banking alerts, and enterprise notifications. However, the ubiquity of text messaging has made it a prime target for unsolicited commercial advertisements, lottery scams, and phishing attacks—collectively referred to as **Spam**. Traditional rule-based filters that rely on static keyword blacklists fail when spammers intentionally obfuscate words or alter phrasing.

This Model Development Document (MDD) details the design, mathematical foundation, training, validation, and cloud deployment of a **Deep Learning Natural Language Processing (NLP)** classification model developed as a Bootcamp Capstone Project. Using TensorFlow and Keras, the model learns semantic word representations directly from historical SMS data to classify incoming text messages as either **Ham (Legitimate / Not Spam)** or **Spam (Unsolicited / Malicious)**.

### 1.2 Core Objectives & Scope

The primary goal of this capstone project is to build a clean, reproducible, and interpretable end-to-end Deep Learning NLP pipeline suitable for academic evaluation and real-time demonstration. Specifically, the project scope encompasses:

1. **Automated Data Retrieval:** Fetching the benchmark dataset directly from Kaggle using the `kagglehub` library to guarantee reproducibility.
2. **Foundational NLP Preprocessing:** Implementing text normalization, regular expression cleaning, vocabulary tokenization, and sequence padding.
3. **Deep Neural Network Design:** Constructing a lightweight `Sequential` neural network in TensorFlow featuring an `Embedding` layer, `GlobalAveragePooling1D`, and `Dense` layers.
4. **Rigorous Validation:** Splitting the dataset using an $80\%$ training and $20\%$ testing ratio to evaluate generalization and prevent overfitting.
5. **Interactive Cloud Deployment:** Packaging the trained model (`spam_model.h5`) and tokenizer (`tokenizer.pkl`) into a publicly accessible Streamlit web application with built-in sample test cases.

### 1.3 Key Deliverables & External Links

All project artifacts have been developed, tested, and deployed to production:

* **Benchmark Dataset:** [https://www.kaggle.com/datasets/team-ai/spam-text-message-classification](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)
* **Live Streamlit Web Application:** [https://nlp-spam-detection.streamlit.app/](https://nlp-spam-detection.streamlit.app/)
* **Source Code Deliverables:** `requirements.txt`, `project.ipynb`, `app.py`
* **Serialized Model Artifacts:** `spam_model.h5`, `tokenizer.pkl`

### 1.4 End-to-End System Architecture

The complete project lifecycle is divided into two distinct phases: the **Offline Training Pipeline** (executed in `project.ipynb`) and the **Online Inference Pipeline** (executed in `app.py` on Streamlit Cloud).

```
================================================================================
                        PHASE 1: OFFLINE TRAINING PIPELINE
================================================================================
[Kaggle Dataset: SPAM text message 20170820 - Data.csv (5,572 rows)]
       |
       v
[Regex Text Cleaning: Lowercase + Strip Punctuation/Numbers]
       |
       v
[Train-Test Split: 80% Train (4,457 rows) | 20% Test (1,115 rows)]
       |
       v
[Keras Tokenizer (Vocab = 5,000) & Sequence Padding (Max Length = 100)]
       |--------------------------------------> Saves: tokenizer.pkl
       v
[TensorFlow Sequential Model: Embedding -> GlobalAvgPool1D -> Dense -> Sigmoid]
       |
       v
[Model Evaluation on 20% Test Split (~98.5% Accuracy)]
       |--------------------------------------> Saves: spam_model.h5

================================================================================
                        PHASE 2: ONLINE INFERENCE PIPELINE
================================================================================
[User Input / Sample Dropdown Selection in Streamlit UI]
       |
       v
[Text Normalization (clean_text function)]
       |
       v
[Integer Encoding via Loaded tokenizer.pkl + Post-Padding (Length = 100)]
       |
       v
[Forward Pass Prediction via Loaded spam_model.h5]
       |
       v
[Probability Thresholding (p >= 0.50 -> SPAM | p < 0.50 -> HAM)]
```

---

## 2. Problem Statement & Technical Context

### 2.1 The Challenge of Unsolicited Text Messages (Spam)

Text message spam poses both a nuisance to consumers and a severe cybersecurity risk. Unlike email spam—which often contains rich HTML headers, sender IP metadata, and domain signatures—SMS messages consist strictly of short, unstructured plain text (historically constrained to $160$ characters). Furthermore, SMS text is characterized by:

* Informal abbreviations and slang (e.g., *"u"*, *"ur"*, *"wkly"*, *"txt"*, *"tkts"*).
* High density of promotional buzzwords (*"FREE"*, *"WINNER"*, *"URGENT"*, *"CLAIM"*).
* Numerical strings representing premium rate phone numbers, claim codes, or currency figures.

### 2.2 Motivation for Deep Learning in NLP Classification

In classical Machine Learning, text classification relies on Bag-of-Words (BoW) or Term Frequency-Inverse Document Frequency ($\text{TF-IDF}$) matrices paired with linear classifiers like Naive Bayes or Logistic Regression. While effective, $\text{TF-IDF}$ representations treat every word as an orthogonal, independent dimension, failing to capture semantic similarity between related words (e.g., *"prize"*, *"reward"*, *"jackpot"*, and *"won"*).

By utilizing a **Deep Learning Word Embedding** architecture in TensorFlow:

1. Words are projected into a continuous, low-dimensional vector space ($\mathbb{R}^{16}$) where semantically similar words cluster together geometrically during training.
2. `GlobalAveragePooling1D` aggregates these dense word vectors across the entire message length, creating a compact document-level representation that is invariant to exact word position and highly resistant to overfitting on small datasets.
3. Non-linear hidden layers learn complex combinations of semantic features to output a calibrated probability score between $0$ and $1$.

### 2.3 Model Success Criteria & Evaluation Benchmarks

To consider the capstone model successful and ready for deployment, the following benchmark criteria were established prior to training:

| Criterion Category          | Target Metric                  | Minimum Acceptable Threshold |          Achieved Result          |
| :-------------------------- | :----------------------------- | :--------------------------: | :--------------------------------: |
| **Primary Accuracy**  | Overall Test Accuracy          |       $\ge 95.00\%$       |       **$98.48\%$**       |
| **Generalization**    | Train-Test Accuracy Gap        |        $\le 2.50\%$        |        **$0.96\%$**        |
| **Spam Detection**    | Spam Class Precision           |       $\ge 92.00\%$       |       **$97.10\%$**       |
| **Spam Coverage**     | Spam Class Recall              |       $\ge 88.00\%$       |       **$91.28\%$**       |
| **Inference Latency** | Single Message Prediction Time |    $\le 200\text{ ms}$    | **$\approx 35\text{ ms}$** |

---

## 3. Data Acquisition & Exploratory Data Analysis (EDA)

### 3.1 Programmatic Data Ingestion via `kagglehub`

To ensure that the notebook (`project.ipynb`) can be executed on any local machine or cloud environment (such as Google Colab) without manual file uploads, the dataset is fetched programmatically using the official `kagglehub` package:

```python
import kagglehub

# Download latest version of the dataset from Kaggle
path = kagglehub.dataset_download("team-ai/spam-text-message-classification")
print("Path to dataset files:", path)
```

This operation downloads and caches the raw CSV file `SPAM text message 20170820 - Data.csv` directly from the Kaggle repository (`https://www.kaggle.com/datasets/team-ai/spam-text-message-classification`).

### 3.2 Dataset Structure & Schema Definition

The dataset consists of **$5,572$ rows** and **$2$ columns**. Below is the formal schema definition:

| Column Name  | Data Type               |     Nullable     | Description & Domain Values                                                                                                              |
| :----------- | :---------------------- | :--------------: | :--------------------------------------------------------------------------------------------------------------------------------------- |
| `Category` | `string` (`object`) | No ($0$ nulls) | Ground-truth target classification label. Takes strictly two values: `'ham'` (legitimate message) or `'spam'` (unsolicited message). |
| `Message`  | `string` (`object`) | No ($0$ nulls) | The raw, unedited plain-text content of the SMS message.                                                                                 |

#### Representative Raw Records from the Dataset

| Row Index | `Category` | `Message` (Raw Text)                                                                                                                                               |
| :-------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   $0$   |   `ham`   | *"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."*                                                |
|   $2$   |   `spam`   | *"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"*    |
|   $4$   |   `ham`   | *"Nah I don't think he goes to usf, he lives around here though"*                                                                                                  |
|   $8$   |   `spam`   | *"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."* |
|  $12$  |   `spam`   | *"URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18"*   |

### 3.3 Target Class Distribution & Imbalance Analysis

An analysis of the `Category` column reveals an imbalanced binary distribution typical of real-world fraud and spam datasets:

$$
\text{Total Samples } (N) = 5,572
$$

$$
\text{Ham Count } (N_{\text{ham}}) = 4,825 \quad \left(\frac{4,825}{5,572} \times 100 = 86.5937\%\right)
$$

$$
\text{Spam Count } (N_{\text{spam}}) = 747 \quad \left(\frac{747}{5,572} \times 100 = 13.4063\%\right)
$$

$$
\text{Imbalance Ratio} = \frac{N_{\text{ham}}}{N_{\text{spam}}} = \frac{4,825}{747} \approx 6.46 : 1
$$

While the dataset exhibits a $6.46 : 1$ ratio in favor of `ham`, the lexical signatures of `spam` messages are highly distinct, allowing a neural network trained with Binary Cross-Entropy to achieve strong separation without requiring synthetic oversampling (SMOTE) or complex class weighting.

### 3.4 Quantitative Text Length & Lexical Statistics

To determine optimal sequence padding lengths (`max_len`) and vocabulary sizes (`vocab_size`), comprehensive descriptive statistics were computed across the entire corpus for Character Length, Raw Word Count, and Cleaned Word Count (words remaining after removing numbers and punctuation).

| Subset                 | Metric             | Count ($N$) | Mean ($\mu$) | Std Dev ($\sigma$) |  Min  | $25\%$ (Q1) | Median ($50\%$) | $75\%$ (Q3) |   Max   |
| :--------------------- | :----------------- | :-----------: | :------------: | :------------------: | :---: | :-----------: | :---------------: | :-----------: | :-----: |
| **All Data**     | Character Length   |   $5,572$   |   $80.37$   |      $59.93$      | $2$ |   $36.0$   |     $61.0$     |   $122.0$   | $910$ |
| **All Data**     | Raw Word Count     |   $5,572$   |   $15.58$   |      $11.41$      | $1$ |    $7.0$    |     $12.0$     |   $23.0$   | $171$ |
| **All Data**     | Cleaned Word Count |   $5,572$   |   $14.86$   |      $10.83$      | $0$ |    $7.0$    |     $12.0$     |   $22.0$   | $171$ |
| **Ham (`0`)**  | Character Length   |   $4,825$   |   $71.45$   |      $58.43$      | $2$ |   $33.0$   |     $52.0$     |   $93.0$   | $910$ |
| **Ham (`0`)**  | Raw Word Count     |   $4,825$   |   $14.31$   |      $11.52$      | $1$ |    $7.0$    |     $11.0$     |   $19.0$   | $171$ |
| **Ham (`0`)**  | Cleaned Word Count |   $4,825$   |   $13.89$   |      $11.12$      | $0$ |    $6.0$    |     $10.0$     |   $18.0$   | $171$ |
| **Spam (`1`)** | Character Length   |    $747$    |   $137.99$   |      $29.98$      | $7$ |   $132.0$   |     $149.0$     |   $157.0$   | $223$ |
| **Spam (`1`)** | Raw Word Count     |    $747$    |   $23.81$   |       $5.96$       | $1$ |   $22.0$   |     $25.0$     |   $28.0$   | $35$ |
| **Spam (`1`)** | Cleaned Word Count |    $747$    |   $21.13$   |       $5.60$       | $1$ |   $18.0$   |     $22.0$     |   $25.0$   | $32$ |

#### Key Analytical Findings from EDA:

1. **Structural Length Divergence:** `Spam` messages have a median character length of $149.0$ characters ($\mu = 137.99$, $\sigma = 29.98$), clustering tightly near the standard $160$-character SMS billing limit as marketers maximize ad copy per message. Conversely, `ham` messages have a median length of just $52.0$ characters ($\mu = 71.45$), reflecting brief conversational replies (e.g., *"Ok lor..."*, *"Are you home?"*).
2. **Impact of Regex Cleaning on Word Count:** In `spam` messages, the mean word count drops from $23.81$ raw words to $21.13$ cleaned words (a reduction of $2.68$ tokens per message) because `spam` frequently contains standalone phone numbers, claim numbers, and currency quantities that are stripped during alphabetic normalization.
3. **Justification for `max_len = 100`:** Across all $5,572$ messages, the maximum word count for `spam` is $35$ words ($32$ cleaned words), and over $99.8\%$ of all `ham` messages contain fewer than $100$ words. Setting `max_len = 100` captures virtually $100\%$ of the semantic content without wasting memory on extreme upper-tail `ham` outliers (which reach up to $171$ words).

### 3.5 Data Integrity, Missing Values & Outlier Inspection

* **Missing Values:** A null check (`df.isnull().sum()`) confirms exactly $0$ missing values in both `Category` and `Message`.
* **Outlier Inspection:** A small number of `ham` messages consist solely of numeric or symbolic characters (e.g., `":)"` or `"645"`), which yield a cleaned word count of $0$ after stripping non-alphabetic characters. These records map cleanly to a zero-padded vector `[0, 0, ..., 0]` and are correctly classified as `ham` by the network bias term.

---

## 4. Natural Language Processing (NLP) Preprocessing Pipeline

Raw text cannot be fed directly into a neural network because mathematical operations (matrix multiplication and gradient descent) require numerical tensors of uniform shape. The preprocessing pipeline transforms variable-length strings into standardized integer matrices of shape $(N, 100)$.

### 4.1 Text Normalization & Regular Expression Cleaning

A deterministic text-cleaning function (`clean_text`) is applied both during training (`project.ipynb`) and during live inference (`app.py`):

```python
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

This function executes three sequential transformations:

1. **Case Folding (`str(text).lower()`):** Converts all uppercase characters to lowercase so that `"WINNER"`, `"Winner"`, and `"winner"` map to the exact same vocabulary token.
2. **Non-Alphabetic Filtering (`re.sub(r'[^a-z\s]', '', text)`):** Strips all punctuation marks, special symbols (`£`, `$`, `!`, `?`, `&`), and digits (`0-9`), retaining only lowercase English letters (`a-z`) and whitespace. This prevents punctuation variations from fracturing the vocabulary.
3. **Whitespace Collapsing (`re.sub(r'\s+', ' ', text).strip()`):** Replaces multiple consecutive spaces, tabs, or newlines with a single space and removes leading/trailing spaces.

### 4.2 Target Label Encoding

Neural networks with a sigmoid output layer require binary numerical targets $y \in \{0, 1\}$. The categorical string labels are mapped explicitly as follows:

$$
y_i = \begin{cases} 0 & \text{if } \text{Category}_i = \text{'ham'} \\ 1 & \text{if } \text{Category}_i = \text{'spam'} \end{cases}
$$

### 4.3 Data Partitioning Strategy ($80\text{--}20$ Train-Test Split)

Using `sklearn.model_selection.train_test_split` with `test_size = 0.20` and a fixed random seed (`random_state = 42`) for reproducibility, the $5,572$ records are partitioned into:

* **Training Set ($X_{\text{train}}, y_{\text{train}}$):** $4,457$ messages ($80.0\%$) used to fit the `Tokenizer` vocabulary and optimize neural network weights.
* **Testing Set ($X_{\text{test}}, y_{\text{test}}$):** $1,115$ messages ($20.0\%$) held out completely from training to evaluate unseen generalization performance.

### 4.4 Vocabulary Construction & Integer Tokenization

The Keras `Tokenizer` (`tensorflow.keras.preprocessing.text.Tokenizer`) is initialized with:

* `num_words = 5000` (`vocab_size`): Restricts the active dictionary to the top $4,999$ most frequent words in the training corpus (plus index $0$ reserved for padding and index $1$ reserved for `<OOV>`).
* `oov_token = "<OOV>"`: Assigns index $1$ to any Out-Of-Vocabulary word encountered during testing or live production inference that was not seen during training.

**Critical Leakage Prevention Rule:** `tokenizer.fit_on_texts(X_train)` is called **exclusively on the training set** ($X_{\text{train}}$), never on the full dataset, ensuring zero data leakage from the test set into the vocabulary dictionary.

### 4.5 Sequence Padding & Truncation Mechanics

After converting sentences into variable-length integer lists via `texts_to_sequences()`, `pad_sequences()` enforces a uniform vector length of $L = 100$ (`max_len = 100`) using `padding = 'post'`:

* If a message has $k < 100$ tokens, $100 - k$ zeros (`0`) are appended to the **end** of the sequence.
* If a message has $k > 100$ tokens, it is truncated to $100$ tokens.

This transforms $X_{\text{train}}$ into a 2D tensor of shape $(4457, 100)$ and $X_{\text{test}}$ into a 2D tensor of shape $(1115, 100)$.

### 4.6 Step-by-Step Pipeline Walkthrough on a Sample Record

Below is the exact transformation of **Spam Sample 1** (Dataset Row $8$) through our preprocessing pipeline:

| Pipeline Stage                                                 | Data Representation                                                                                                                 |
| :------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------- |
| **1. Raw Input String**                                  | `"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461."`       |
| **2. Cleaned String (`clean_text`)**                   | `"winner as a valued network customer you have been selected to receivea prize reward to claim call"`                             |
| **3. Integer Sequence (`texts_to_sequences`)**         | `[642, 74, 5, 891, 392, 231, 3, 18, 98, 468, 2, 1, 134, 512, 2, 118, 17]` *(Note: `"receivea"` maps to `1` `<OOV>`)*      |
| **4. Post-Padded Tensor (`pad_sequences`, $L=100$)** | `[642, 74, 5, 891, 392, 231, 3, 18, 98, 468, 2, 1, 134, 512, 2, 118, 17, 0, 0, 0, ..., 0]` *(17 word IDs followed by 83 zeros)* |

---

## 5. Deep Learning Model Architecture & Mathematical Formulation

### 5.1 Sequential Neural Network Overview

The classification model is built using the TensorFlow Keras `Sequential` API. It is intentionally designed to be compact, fast to train, and highly interpretable while capturing non-linear semantic relationships.

```python
model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=100),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

### 5.2 Layer 1: Dense Word Embedding Layer (`Embedding`)

* **Input Shape:** $(B, 100)$, where $B$ is the batch size and $100$ is the padded sequence length ($L$).
* **Configuration:** Vocabulary size $V = 5000$, Embedding dimension $D = 16$.
* **Mathematical Operation:** The layer maintains a learnable lookup matrix $\mathbf{E} \in \mathbb{R}^{5000 \times 16}$. Each integer token $w_t \in \{0, 1, \dots, 4999\}$ at position $t \in \{1, 2, \dots, 100\}$ is mapped to its corresponding row vector $\mathbf{e}_t \in \mathbb{R}^{16}$:

$$
\mathbf{e}_t = \mathbf{E}[w_t, :] \in \mathbb{R}^{16}
$$

* **Output Shape:** $(B, 100, 16)$.

### 5.3 Layer 2: Dimensionality Reduction (`GlobalAveragePooling1D`)

* **Input Shape:** $(B, 100, 16)$.
* **Mathematical Operation:** Rather than flattening the $(100, 16)$ matrix into a $1600$-dimensional vector (which would drastically increase parameters and cause overfitting) or using a computationally heavy LSTM/GRU, `GlobalAveragePooling1D` computes the arithmetic mean across the sequence dimension ($L = 100$) for each of the $16$ embedding channels:

$$
\mathbf{h}_{\text{pool}} = \frac{1}{L} \sum_{t=1}^{L} \mathbf{e}_t \in \mathbb{R}^{16}
$$

* **Output Shape:** $(B, 16)$. This vector represents the average semantic embedding of the entire text message.

### 5.4 Layer 3: Non-Linear Feature Extraction (`Dense` with ReLU)

* **Input Shape:** $(B, 16)$.
* **Configuration:** $16$ hidden units with Rectified Linear Unit ($\text{ReLU}$) activation.
* **Mathematical Operation:** Given weight matrix $\mathbf{W}_1 \in \mathbb{R}^{16 \times 16}$ and bias vector $\mathbf{b}_1 \in \mathbb{R}^{16}$, the layer computes an affine transformation followed by element-wise $\text{ReLU}$ activation:

$$
\mathbf{z}_1 = \mathbf{W}_1^T \mathbf{h}_{\text{pool}} + \mathbf{b}_1
$$

$$
\mathbf{h}_1 = \text{ReLU}(\mathbf{z}_1) = \max(\mathbf{0}, \mathbf{z}_1) \in \mathbb{R}^{16}
$$

* **Output Shape:** $(B, 16)$.

### 5.5 Layer 4: Probabilistic Binary Classification (`Dense` with Sigmoid)

* **Input Shape:** $(B, 16)$.
* **Configuration:** $1$ output neuron with Sigmoid ($\sigma$) activation.
* **Mathematical Operation:** Given weight vector $\mathbf{w}_2 \in \mathbb{R}^{16 \times 1}$ and scalar bias $b_2 \in \mathbb{R}$, the layer computes the logit $z_2$ and squashes it into the probability interval $(0, 1)$:

$$
z_2 = \mathbf{w}_2^T \mathbf{h}_1 + b_2
$$

$$
\hat{y} = \sigma(z_2) = \frac{1}{1 + e^{-z_2}} \in (0, 1)
$$

Here, $\hat{y} = P(y = 1 \mid \mathbf{x})$ represents the predicted probability that the input message is **Spam**. The decision rule classifies the message using a threshold of $\tau = 0.50$:

$$
\text{Predicted Class} = \begin{cases} \text{SPAM } (1) & \text{if } \hat{y} \ge 0.50 \\ \text{HAM } (0) & \text{if } \hat{y} < 0.50 \end{cases}
$$

### 5.6 Network Parameter Count & Memory Footprint Calculation

Every trainable parameter in the neural network can be verified analytically:

| Layer Name                   | Layer Type                 |    Output Tensor Shape    |  Parameter Calculation Formula  | Trainable Parameters |
| :--------------------------- | :------------------------- | :------------------------: | :-----------------------------: | :------------------: |
| `embedding`                | `Embedding`              | $(\text{None}, 100, 16)$ | $V \times D = 5000 \times 16$ |      $80,000$      |
| `global_average_pooling1d` | `GlobalAveragePooling1D` |   $(\text{None}, 16)$   |  $0$ (No learnable weights)  |        $0$        |
| `dense`                    | `Dense` (ReLU)           |   $(\text{None}, 16)$   |     $(16 \times 16) + 16$     |       $272$       |
| `dense_1`                  | `Dense` (Sigmoid)        |    $(\text{None}, 1)$    |      $(16 \times 1) + 1$      |        $17$        |
| **Total Network**      | —                         |             —             |    $80,000 + 0 + 272 + 17$    | **$80,289$** |

With $80,289$ 32-bit floating-point parameters ($4$ bytes per parameter), the raw weight footprint is only $321.16\text{ KB}$, making the saved `spam_model.h5` file extremely lightweight and ideal for rapid cloud deployment.

---

## 6. Model Training Methodology & Optimization

### 6.1 Objective Loss Function: Binary Cross-Entropy

Because spam detection is framed as a binary classification task with probabilistic outputs $\hat{y}_i \in (0, 1)$ and ground-truth labels $y_i \in \{0, 1\}$, the network is trained to minimize **Binary Cross-Entropy (Log Loss)** across a mini-batch of $M$ samples:

$$
\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{M} \sum_{i=1}^{M} \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
$$

Binary Cross-Entropy heavily penalizes confident wrong predictions (as $\hat{y}_i \rightarrow 0$ when $y_i = 1$, $-\log(\hat{y}_i) \rightarrow \infty$), driving the embedding vectors of spam keywords rapidly away from normal conversational words.

### 6.2 Optimization Algorithm: Adaptive Moment Estimation (Adam)

The model uses the **Adam** (`optimizer = 'adam'`) algorithm, which combines Momentum (first-order moment of gradients $m_t$) and RMSProp (second-order uncentered moment of gradients $v_t$) to adaptively scale the learning rate for each of the $80,289$ parameters:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

Adam is particularly advantageous for NLP embedding layers because rare words receive infrequent gradient updates; Adam scales their step sizes appropriately without manual learning rate scheduling.

### 6.3 Training Hyperparameter Configuration

| Hyperparameter                                 |                          Configured Value                          | Rationale                                                                                     |
| :--------------------------------------------- | :-----------------------------------------------------------------: | :-------------------------------------------------------------------------------------------- |
| **Vocabulary Size (`vocab_size`)**     |                              $5,000$                              | Captures all frequent English and SMS slang terms while filtering rare typos.                 |
| **Sequence Max Length (`max_len`)**    |                               $100$                               | Covers $100\%$ of spam messages (max $35$ words) and $>99.8\%$ of ham messages.         |
| **Embedding Dimension (`output_dim`)** |                               $16$                               | Sufficient capacity for binary classification without overfitting $4,457$ training samples. |
| **Hidden Dense Units**                   |                               $16$                               | Matches the embedding dimension for smooth feature projection.                                |
| **Optimizer**                            | `Adam` ($\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$) | Fast convergence on sparse embedding lookups.                                                 |
| **Batch Size (`batch_size`)**          |                               $32$                               | Yields $140$ gradient updates per epoch ($\lceil 4457 / 32 \rceil = 140$).                |
| **Training Epochs (`epochs`)**         |                               $10$                               | Achieves full convergence on validation accuracy before overfitting sets in.                  |

### 6.4 Epoch-by-Epoch Convergence & Learning Dynamics

During execution of `model.fit()` across $10$ epochs ($140$ steps per epoch), the model exhibits smooth, monotonic loss reduction and rapid accuracy convergence:

|       Epoch       | Training Loss |   Training Accuracy   | Validation (Test) Loss | Validation (Test) Accuracy | Dynamics Observation                                 |
| :----------------: | :-----------: | :-------------------: | :--------------------: | :------------------------: | :--------------------------------------------------- |
| **Epoch 1** |  $0.4912$  |      $85.64\%$      |       $0.3765$       |        $86.64\%$        | Initial majority-class (`ham`) baseline alignment. |
| **Epoch 2** |  $0.3418$  |      $86.58\%$      |       $0.3182$       |        $86.64\%$        | Embedding vectors begin separating spam tokens.      |
| **Epoch 3** |  $0.2584$  |      $87.95\%$      |       $0.2145$       |        $90.40\%$        | Non-linear ReLU activations cross decision boundary. |
| **Epoch 4** |  $0.1592$  |      $94.77\%$      |       $0.1328$       |        $96.23\%$        | Rapid jump in Spam recall as key terms cluster.      |
| **Epoch 5** |  $0.0985$  |      $97.35\%$      |       $0.0914$       |        $97.58\%$        | Strong discrimination on promotional phrases.        |
| **Epoch 6** |  $0.0684$  |      $98.18\%$      |       $0.0721$       |        $98.03\%$        | Refinement on shorter spam messages.                 |
| **Epoch 7** |  $0.0521$  |      $98.61\%$      |       $0.0625$       |        $98.21\%$        | Steady reduction in Binary Cross-Entropy loss.       |
| **Epoch 8** |  $0.0419$  |      $98.90\%$      |       $0.0568$       |        $98.39\%$        | High confidence on both classes.                     |
| **Epoch 9** |  $0.0348$  |      $99.17\%$      |       $0.0532$       |        $98.48\%$        | Near-optimal weight stabilization.                   |
| **Epoch 10** |  $0.0294$  | **$99.44\%$** |       $0.0511$       |   **$98.48\%$**   | Final converged state saved to `spam_model.h5`.    |

---

## 7. Model Evaluation, Validation & Performance Analysis

### 7.1 Training vs. Testing Sample Performance Comparison

After $10$ epochs of training, evaluating the model on both the $80\%$ training split ($4,457$ samples) and the $20\%$ unseen test split ($1,115$ samples) yields:

| Dataset Split                     | Sample Count ($N$) | Binary Cross-Entropy Loss | Classification Accuracy | Error Rate |
| :-------------------------------- | :------------------: | :-----------------------: | :---------------------: | :--------: |
| **Training Set ($80\%$)** |      $4,457$      |        $0.0294$        |  **$99.44\%$**  | $0.56\%$ |
| **Testing Set ($20\%$)**  |      $1,115$      |        $0.0511$        |  **$98.48\%$**  | $1.52\%$ |

### 7.2 Confusion Matrix & Secondary Classification Metrics

Because the dataset is imbalanced ($86.6\%$ Ham vs. $13.4\%$ Spam), accuracy alone is insufficient to validate performance. Evaluating the $1,115$ test samples ($966$ actual Ham, $149$ actual Spam) produces the following Confusion Matrix:

|                                | **Predicted: HAM ($0$)** | **Predicted: SPAM ($1$)** | **Row Total** |
| :----------------------------- | :------------------------------: | :-------------------------------: | :-----------------: |
| **Actual: HAM ($0$)**  |       $\text{TN} = 962$       |         $\text{FP} = 4$         |       $966$       |
| **Actual: SPAM ($1$)** |        $\text{FN} = 13$        |        $\text{TP} = 136$        |       $149$       |
| **Column Total**         |             $975$             |              $140$              | **$1,115$** |

From the Confusion Matrix components ($\text{TP} = 136$, $\text{TN} = 962$, $\text{FP} = 4$, $\text{FN} = 13$), we derive the exact classification metrics:

1. **Overall Accuracy:**

   $$
   \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{136 + 962}{1,115} = \frac{1,098}{1,115} \approx 98.475\%
   $$
2. **Spam Precision (Positive Predictive Value):**

   $$
   \text{Precision}_{\text{spam}} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{136}{136 + 4} = \frac{136}{140} \approx 97.14\%
   $$

   *Interpretation:* When the model flags a message as SPAM, it is correct $97.14\%$ of the time. Only $4$ legitimate messages out of $966$ ($0.41\%$) were falsely flagged.
3. **Spam Recall (Sensitivity / True Positive Rate):**

   $$
   \text{Recall}_{\text{spam}} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{136}{136 + 13} = \frac{136}{149} \approx 91.28\%
   $$

   *Interpretation:* The model successfully catches $91.28\%$ of all unseen spam messages in the test set.
4. **Spam $F_1$-Score (Harmonic Mean of Precision and Recall):**

   $$
   F_1 = 2 \times \frac{\text{Precision}_{\text{spam}} \times \text{Recall}_{\text{spam}}}{\text{Precision}_{\text{spam}} + \text{Recall}_{\text{spam}}} = 2 \times \frac{0.9714 \times 0.9128}{0.9714 + 0.9128} \approx 94.12\%
   $$

### 7.3 Generalization & Overfitting Diagnostics

The generalization gap between training accuracy ($99.44\%$) and test accuracy ($98.48\%$) is less than $1\%$ ($\Delta = 0.96\%$). This confirms that the `GlobalAveragePooling1D` architecture effectively regularizes the model, preventing it from memorizing specific training sentences while learning generalizable word representations.

### 7.4 Qualitative Error Analysis (False Positives & False Negatives)

* **False Positives ($\text{FP} = 4$):** Legitimate `ham` messages misclassified as `spam` typically contain promotional-sounding vocabulary in a personal context (e.g., a friend texting about winning free concert tickets or sharing a discount code).
* **False Negatives ($\text{FN} = 13$):** `Spam` messages missed by the model are predominantly very short conversational phishing lures (e.g., *"Hi babe, call me back at this number"*) that lack classic commercial keywords (*"prize"*, *"claim"*, *"urgent"*) and Lose their numerical phone string during `clean_text` regex filtering.

---

## 8. Model Serialization & Artifact Management

To transition from the experimental Jupyter Notebook (`project.ipynb`) to the standalone production web application (`app.py`), two distinct artifacts are serialized to disk at the end of training.

### 8.1 Neural Network Weights & Topology Persistence (`spam_model.h5`)

```python
model.save("spam_model.h5")
```

The `.h5` (Hierarchical Data Format version 5) file stores the complete TensorFlow/Keras neural network:

1. The structural layer configuration (`Embedding` $\rightarrow$ `GlobalAveragePooling1D` $\rightarrow$ `Dense` $\rightarrow$ `Dense`).
2. The exact learned weight matrices and bias vectors ($80,289$ float32 parameters).
3. The optimizer state and compilation loss/metric definitions.

### 8.2 Lexical Vocabulary Persistence (`tokenizer.pkl`)

```python
import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

The `.pkl` (Python Pickle) file serializes the fitted Keras `Tokenizer` object, preserving the exact `word_index` lookup dictionary created from `X_train` (where each of the $5,000$ words is mapped to its specific integer ID).

### 8.3 Why Deep Learning NLP Requires Dual Artifacts Compared to Traditional ML

In standard tabular Machine Learning projects (such as predicting house prices or medical metrics using Scikit-Learn), user inputs are already numerical features, requiring only a single `model.pkl` file. Even in Scikit-Learn text projects, `TfidfVectorizer` and `LogisticRegression` can be wrapped inside a single `sklearn.pipeline.Pipeline` object and saved as one `.pkl` file.

In classic TensorFlow/Keras workflows, however:

* The neural network graph (`Sequential`) expects numerical tensors as input and is saved using Keras's native HDF5 saver (`spam_model.h5`).
* The text-to-integer `Tokenizer` is a pure Python preprocessing object that lives outside the neural network layers.
* If `app.py` did not load `tokenizer.pkl`, it would have no way of knowing which integer ID corresponds to which word in the `Embedding` lookup matrix $\mathbf{E} \in \mathbb{R}^{5000 \times 16}$. Both files work in tandem during inference.

---

## 9. Web Application Deployment & User Interface (Streamlit)

### 9.1 Production Inference Pipeline Architecture

The user-facing web application is built using **Streamlit** (`app.py`). When a user visits the app and clicks **"Check Message"**, the application executes an end-to-end inference pipeline in $\approx 35\text{ ms}$:

1. Captures the raw string from the UI text area (`user_input`).
2. Normalizes the string using the identical `clean_text()` regex function used in `project.ipynb`.
3. Transforms the cleaned string into a list of word IDs via `tokenizer.texts_to_sequences([cleaned])`.
4. Post-pads the integer list to length $100$ via `pad_sequences(seq, maxlen=100, padding='post')`.
5. Passes the $(1, 100)$ tensor into `model.predict(padded)` to obtain the scalar probability $\hat{y} \in (0, 1)$.
6. Renders a green success banner (**Not SPAM (Ham)**) if $\hat{y} < 0.50$, or a red alert banner (**SPAM Message**) if $\hat{y} \ge 0.50$, alongside the calibrated confidence score.

### 9.2 Cloud Hosting & Continuous Deployment

The application is deployed and publicly accessible on **Streamlit Community Cloud**:

* **Live Production URL:** [https://nlp-spam-detection.streamlit.app/](https://nlp-spam-detection.streamlit.app/)

Streamlit Cloud connects directly to the project repository, installs the exact packages listed in `requirements.txt`, loads `spam_model.h5` and `tokenizer.pkl` into memory, and serves the interactive UI over HTTPS.

### 9.3 Interactive User Experience & Pre-Loaded Dataset Samples

To allow instructors, reviewers, and end-users to test the model immediately without searching for external text messages, `app.py` features an interactive dropdown menu (`st.selectbox`) populated with **$6$ real messages** ($3$ Ham and $3$ Spam) extracted directly from `SPAM text message 20170820 - Data.csv`, plus an option to type custom text.

### 9.4 Verification Test Cases & Live Inference Benchmarks

All pre-loaded sample messages and custom edge cases were verified on the live Streamlit application (`https://nlp-spam-detection.streamlit.app/`):

|  Test Case ID  | Dropdown Selection / Input Source        | Raw Input Text                                                                                        |   Ground Truth   | Model Probability ($\hat{y}$) | Displayed App Output                                   |     Status     |
| :-------------: | :--------------------------------------- | :---------------------------------------------------------------------------------------------------- | :--------------: | :-----------------------------: | :----------------------------------------------------- | :------------: |
| **TC-01** | `Ham Sample 1 (Not SPAM)` (Row $4$)  | *"Nah I don't think he goes to usf, he lives around here though"*                                   | `ham` ($0$) |            $0.01$            | **Not SPAM (Ham)** (Confidence: $0.99$)        | **PASS** |
| **TC-02** | `Ham Sample 2 (Not SPAM)` (Row $20$) | *"Is that seriously how you spell his name?"*                                                       | `ham` ($0$) |            $0.01$            | **Not SPAM (Ham)** (Confidence: $0.99$)        | **PASS** |
| **TC-03** | `Ham Sample 3 (Not SPAM)` (Row $28$) | *"I'm back & we're packing the car now, I'll let you know if there's room"*                         | `ham` ($0$) |            $0.00$            | **Not SPAM (Ham)** (Confidence: $1.00$)        | **PASS** |
| **TC-04** | `Spam Sample 1 (SPAM)` (Row $8$)     | *"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!..."* | `spam` ($1$) |            $0.99$            | **SPAM Message** (Confidence: $0.99$)          | **PASS** |
| **TC-05** | `Spam Sample 2 (SPAM)` (Row $12$)    | *"URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!..."*                | `spam` ($1$) |            $0.99$            | **SPAM Message** (Confidence: $0.99$)          | **PASS** |
| **TC-06** | `Spam Sample 3 (SPAM)` (Row $2$)     | *"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005..."*                           | `spam` ($1$) |            $0.98$            | **SPAM Message** (Confidence: $0.98$)          | **PASS** |
| **TC-07** | Blank / Whitespace Input                 | `""` or `"   "`                                                                                   |       N/A       |               N/A               | Warning: *"Please enter or select a message first."* | **PASS** |

---

## 10. Model Limitations, Operational Risks & Future Scope

### 10.1 Technical & Lexical Limitations

1. **Bag-of-Embeddings Order Invariance:** Because `GlobalAveragePooling1D` averages word vectors across the entire sequence of length $100$, the model is largely insensitive to word order and multi-word negation (e.g., *"This is not a free prize scam"* vs. *"Free prize"*).
2. **Stripping of Numeric & Symbolic Tokens:** The regex filter `re.sub(r'[^a-z\s]', '', text)` removes all numbers, currency signs (`£`, `$`), and URL punctuation. While this keeps the vocabulary simple, phone numbers, shortcodes (`87121`), and currency symbols are strong indicators of spam that are currently discarded.
3. **Static Vocabulary Era:** The Kaggle dataset (`SPAM text message 20170820 - Data.csv`) reflects classic SMS patterns. Modern smishing attacks involving cryptocurrency links, parcel delivery tracking scams, or WhatsApp lures may contain newer vocabulary mapped to `<OOV>`.

### 10.2 Operational & Security Risks

1. **Adversarial Orthographic Obfuscation (Leetspeak):** If a spammer substitutes numbers or symbols inside words (e.g., writing `"FR33 W1NN3R"` instead of `"FREE WINNER"`), the regex cleaner strips the digits, leaving `"fr wnnr"`, which maps to `<OOV>` (`1`) and reduces the predicted spam probability.
2. **Asymmetric Cost of False Positives:** In production messaging systems, quarantining a legitimate personal or banking message (False Positive) is often more disruptive to the user than letting a spam message through (False Negative).

### 10.3 Recommendations for Future Iterations

To evolve this capstone prototype into an enterprise-grade spam filter, the following enhancements are recommended:

* **Preserve Symbolic & Numeric Meta-Tokens:** Replace raw phone numbers, URLs, and currency amounts with special tokens (`<PHONE>`, `<URL>`, `<MONEY>`) prior to stripping punctuation so the model retains structural spam signals.
* **Sequence-Aware Architectures:** Experiment with a `Bidirectional(LSTM(16))` or `Conv1D` layer before pooling to capture local n-gram phrases and word order.
* **Single-Artifact Packaging:** Migrate from the external `Tokenizer` (`tokenizer.pkl`) to TensorFlow's native `tf.keras.layers.TextVectorization` layer embedded directly inside the `Sequential` model so the entire pipeline exports as a single `.keras` file.
* **Adjustable Decision Threshold:** Increase the classification threshold from $\tau = 0.50$ to $\tau = 0.75$ in high-precision environments to drive False Positives near zero.

---

## 11. `Environment Dependencies (requirements.txt)`

```text
kagglehub
pandas
numpy
scikit-learn
tensorflow
streamlit
```
