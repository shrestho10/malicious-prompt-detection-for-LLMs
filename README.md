# 🔒 Detecting Malicious Prompts for Large Language Models (LLMs)
#It is also part of "Prompter Says": A Linguistic Approach to Understanding and Detecting Jailbreak Attacks Against Large-Language Models paper
[Link](https://dl.acm.org/doi/10.1145/3689217.3690618)

![Project Banner](plot0001_.png)

---

## 📖 Project Overview

Large language models (LLMs) are vulnerable to "jailbreak" prompts that induce undesired or harmful behavior. This project develops a machine learning pipeline to detect such malicious prompts using linguistic analysis and various classifiers, enabling early filtering before prompts reach the models.

---

## 🎯 Objectives

- Collect and preprocess a labeled prompt dataset (benign vs. malicious).
- Extract linguistic features: tokenization, lemmatization, stemming, stopword removal.
- Use vectorizers: CountVectorizer, TF-IDF, and Word2Vec embeddings.
- Train and evaluate classifiers including Logistic Regression, SVM, Random Forest, and others.
- Measure performance with accuracy, precision, recall, and F1-score.

---


WorkFlow:
+-------------------+
| 1. Data Collection|
+-------------------+
         |
         v
+------------------------------+
| Load raw CSV into DataFrame |
|  (FinalRaWData.csv)         |
+------------------------------+
         |
         v
+-------------------------------+
| 2. Data Cleaning & Preprocess |
+-------------------------------+
| - Drop Unnamed, NaN, Duplicates|
| - Clean text (tokenize,       |
|   lowercase, stem, lemmatize, |
|   remove stopwords)           |
| - Save cleaned CSV            |
+-------------------------------+
         |
         v
+--------------------------------------+
| 3. Text Vectorization (3 alternatives)|
+--------------------------------------+
| a) CountVectorizer (Bag-of-Words)    |
| b) TF-IDF Vectorizer                 |
| c) Word2Vec Embeddings               |
+--------------------------------------+
         |
         v
+------------------------------+
| 4. Train/Test Split (80/20) |
+------------------------------+
         |
         v
+--------------------------------------------------+
| 5. Model Training & Evaluation                   |
+--------------------------------------------------+
| Multiple Classifiers:                            |
| - Logistic Regression                            |
| - SVM (Support Vector Machine)                   |
| - Random Forest                                  |
| - k-NN (k-Nearest Neighbors)                     |
| - Naive Bayes                                    |
| - Decision Tree                                  |
| - Gradient Boosting                              |
| - Extra Trees                                    |
+--------------------------------------------------+
| Metrics Computed:                                |
| - Accuracy, Precision, Recall, F1 Score          |
+--------------------------------------------------+
         |
         v
+----------------------------+
| 6. Performance Comparison  |
+----------------------------+
| - Print and compare scores |
| - Select best model        |
+----------------------------+



## 🗂️ Project Structure

├── data/
│ ├── FinalRawData.csv
│ └── FinalCleanedData.csv
├── notebooks/
│ ├── cs205_project_data_clean.ipynb
│ └── cs205_project_classifiers.ipynb
├── README.md



---

## 🧹 Data Preprocessing

- Tokenize text with `nltk.RegexpTokenizer`
- Lowercase all words
- Lemmatize using `WordNetLemmatizer`
- Stem using `PorterStemmer`
- Remove English stopwords with NLTK
- Store cleaned text in `cleaned` column

---

## 💡 Feature Extraction Methods

| Method            | Description                              |
|-------------------|----------------------------------------|
| CountVectorizer   | Bag-of-words frequency counts          |
| TF-IDF Vectorizer  | Term frequency - inverse document freq |
| Word2Vec Embedding | Semantic word embeddings from training data |

---

## 🤖 Classifiers Evaluated

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- k-Nearest Neighbors (k-NN)
- Naive Bayes
- Decision Tree
- Gradient Boosting
- Extra Trees

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **F1 Score** (weighted)

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas scikit-learn gensim nltk
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Run data cleaning notebook:

notebooks/cs205_project_data_clean.ipynb

Run model training and evaluation notebook:

notebooks/cs205_project_classifiers.ipynb
