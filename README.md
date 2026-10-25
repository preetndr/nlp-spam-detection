# Automated SMS Spam Detection System (Deep Learning NLP)

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)
![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

A simple, robust, and end-to-end Deep Learning Natural Language Processing (NLP) capstone project that classifies SMS messages as either **Ham (Legitimate)** or **Spam (Unsolicited)** using TensorFlow/Keras and deployed via Streamlit.

---

## 🚀 Streamlit Live Demo

Test the live production model directly in your browser:
👉 **[https://nlp-spam-detection.streamlit.app/](https://nlp-spam-detection.streamlit.app/)**

---

## 📂 Project Structure

```text
├── project.ipynb                # Jupyter Notebook for EDA, training, and evaluation
├── app.py                       # Streamlit web application interface
├── requirements.txt             # Python package dependencies
├── spam_model.h5                # Trained TensorFlow Sequential model weights
├── tokenizer.pkl                # Fitted Keras Tokenizer vocabulary dictionary
└── model_development_document.md# Comprehensive 10-page Model Development Document (MDD)
```

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

The project utilizes the **Spam Text Message Classification** dataset from Kaggle (`team-ai/spam-text-message-classification`), downloaded automatically via `kagglehub`.

* **Total Messages:** $5,572$ SMS records
* **Class Imbalance:** $4,825$ `ham` messages ($86.59\%$) vs. $747$ `spam` messages ($13.41\%$) ($6.46:1$ ratio).
* **Length Statistics:** `Spam` messages cluster tightly near the $160$-character SMS limit (median $149$ characters / $25$ words), whereas `ham` messages average shorter conversational lengths (median $52$ characters / $11$ words).

---

## 🧠 Deep Learning Model Architecture

Built using the Keras `Sequential` API with an $80\text{--}20$ training/test split:

1. **`Embedding` Layer:** Maps $5,000$ vocabulary tokens into a continuous $16$-dimensional vector space ($\mathbb{R}^{16}$).
2. **`GlobalAveragePooling1D`:** Averages word vectors across the padded sequence length ($L=100$) for position invariance.
3. **`Dense` Layer ($16$ units, ReLU):** Non-linear feature extraction.
4. **`Dense` Layer ($1$ unit, Sigmoid):** Probabilistic binary classification output ($\hat{y} \in [0, 1]$).

---

## ⚙️ Local Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/preetndr/nlp-spam-detection.git
cd nlp-spam-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook (Optional - to re-train model)

Open `project.ipynb` in Jupyter Lab or Google Colab and run all cells to download the data, train the model, and generate `spam_model.h5` and `tokenizer.pkl`.

### 4. Run the Streamlit App Locally

```bash
python -m streamlit run app.py
```

---

## 📈 Model Performance

Evaluated on an unseen $20\%$ test split ($1,115$ samples):

* **Test Accuracy:** **$98.48\%$**
* **Spam Precision:** **$97.14\%$**
* **Spam Recall:** **$91.28\%$**
* **Spam $F_1$-Score:** **$94.12\%$**
