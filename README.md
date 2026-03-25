# CS481 – Amazon Reviews Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using machine learning models such as Naive Bayes and kNN.

---

## Project Pipeline

The project is divided into steps:

### Step 1: Data Loading
- Load Amazon reviews dataset
- Keep only relevant columns (Text, Score)
- Convert scores into labels:
  - POSITIVE (score ≥ 4)
  - NEGATIVE (score ≤ 2)

---

### Step 2: Preprocessing
- Convert text to lowercase
- Remove punctuation
- Tokenize text into words
- Store tokens in a new column: `Tokens`

---

### Step 3: Vocabulary Building
- Build a vocabulary from all tokens
- Use a Python set to keep only unique words
- Convert it into a sorted list

**Result:**
- Vocabulary size ≈ 23,000 words

**Note:**
The vocabulary includes numbers and product-related terms because numeric values were not removed during preprocessing.

---

## Step 4 – Vectorize Documents (Bag-of-Words)

Each preprocessed document is converted into a numerical representation using a Non-Binary Bag-of-Words model. For every review, the vectorizer counts how many times each vocabulary word appears and stores the result as a dictionary mapping words to their counts. For example, the token list `["good", "food", "good"]` becomes `{"good": 2, "food": 1}`. Words that do not appear in the vocabulary are ignored. This vector representation is what the classifiers will use to make predictions.

---

## Step 5 – Split Dataset

The dataset is split into a training set and a test set following the assignment rules.
The first `TRAIN_SIZE%` of samples are used for training and the remaining samples form
the test set. For example, with 1000 samples and `TRAIN_SIZE = 80`, the first 800 samples
are used for training and the last 200 for testing. The split is done by position, not
randomly, to ensure consistency across runs.

---

## Project Structure
