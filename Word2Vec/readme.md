# 🧠 Word2Vec Text Embedding Using Gensim (NLP Project)

This project demonstrates how to preprocess text data and train a **Word2Vec model** using the **Gensim library**. The model learns meaningful vector representations of words that can be used for various Natural Language Processing (NLP) tasks such as sentiment analysis, document similarity, and recommendation systems.

Link to the Dataset: 
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz
---

## 📌 Features

* Text preprocessing using `gensim.simple_preprocess`
* Vocabulary building with frequency filtering
* Training Word2Vec embeddings
* Finding similar words
* Word similarity calculation
* Optimized multi-core training

---

## 🛠 Technologies Used

* Python
* Pandas
* Gensim
* NumPy

---

## 📂 Project Workflow

```
Raw Text Data
      ↓
Text Cleaning & Tokenization
(simple_preprocess)
      ↓
Vocabulary Building
(build_vocab)
      ↓
Word2Vec Training
      ↓
Word Embeddings Output
```

---

## 📊 Dataset Format

Your dataset should contain a text column such as:

| reviewText                  |
| --------------------------- |
| "This product is very good" |
| "Battery life is poor"      |

---

## ⚙ Installation

Install required libraries:

```bash
pip install gensim pandas numpy
```

---

## 🚀 How It Works

### Step 1: Import Libraries

```python
import gensim
import pandas as pd
```

---

### Step 2: Text Preprocessing

```python
review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
```

This performs:

* Lowercasing
* Tokenization
* Removing punctuation
* Removing special characters
* Cleaning text

---

### Step 3: Initialize Word2Vec Model

```python
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)
```

---

### Step 4: Build Vocabulary

```python
model.build_vocab(review_text, progress_per=1000)
```

This step:

* Scans dataset
* Counts word frequencies
* Removes rare words
* Prepares training structure

---

### Step 5: Train Model

```python
model.train(
    review_text,
    total_examples=model.corpus_count,
    epochs=10
)
```

---

## 📈 Model Usage Examples

### Get Word Vector

```python
model.wv["good"]
```

---

### Find Similar Words

```python
model.wv.most_similar("good")
```

---

### Calculate Similarity

```python
model.wv.similarity("good", "excellent")
```

---

## 💡 Applications

This project can be extended to:

* Sentiment Analysis
* Resume Screening System
* Chatbots
* Recommendation Systems
* Search Engines
* Text Classification
* LSTM Input Embeddings

---

## 📁 Save & Load Model

### Save Model

```python
model.save("word2vec.model")
```

---

### Load Model

```python
from gensim.models import Word2Vec
model = Word2Vec.load("word2vec.model")
```

---

## 🎯 Results

* Generates dense word embeddings
* Captures semantic meaning of words
* Improves NLP model performance
* Efficient on large datasets

---



