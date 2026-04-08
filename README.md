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

### Step 4: Vectorize Documents (Bag-of-Words)
- Convert each tokenized document into a numerical vector
- Use Non-Binary Bag-of-Words (count how many times each word appears)
- Store the result as a dictionary mapping words to their counts
- Words not in the vocabulary are ignored

**Example:**
- Tokens: `["good", "food", "good"]`
- Vector: `{"good": 2, "food": 1}`

---

### Step 5: Split Dataset
- Split the dataset by position (not randomly)
- Training set: first `TRAIN_SIZE%` of samples (between 50% and 80%)
- Test set: always the **last 20%** of samples (fixed)
- Note: if `TRAIN_SIZE < 80`, there will be a gap between train and test that is not used

**Example:**
- Dataset size = 1000, `TRAIN_SIZE = 80`
- Training = first 800 samples
- Testing = last 200 samples (always)

---

### Step 6A: Naïve Bayes Classifier
- Compute prior probabilities: P(POSITIVE) and P(NEGATIVE)
- Compute likelihood probabilities: P(word | POSITIVE) and P(word | NEGATIVE)
- Apply Add-1 (Laplace) smoothing to handle unseen words
- Store all probabilities in dictionaries
- Use log probabilities during prediction to avoid underflow
- Convert log scores back to real probabilities for output

**Output includes:**
- Predicted class label
- P(POSITIVE | S)
- P(NEGATIVE | S)

---

### Step 6B: k-Nearest Neighbors (kNN) Classifier

- Choose a value for k (number of neighbors), e.g., k = 3
- For each test document:
  - Compare it with all training documents
  - Compute distance between their vectors (Euclidean distance)
- Sort all distances from smallest to largest
- Select the k closest documents (nearest neighbors)
- Count how many neighbors are POSITIVE and NEGATIVE
- Assign the label using majority voting

*Example:*

- k = 3
- Nearest neighbors labels = [POSITIVE, POSITIVE, NEGATIVE]
- Prediction = POSITIVE

**Output includes:**

- Predicted class label for the test document

Note:

- kNN does not train a model in advance
- It stores all training data and makes predictions based on similarity
- Smaller distance means more similar documents

---

### Step 7A: Naïve Bayes Evaluation
- Test the Naïve Bayes classifier on all test documents
- Compare predicted labels with actual labels
- Count:
  - **True Positives (TP)**
  - **True Negatives (TN)**
  - **False Positives (FP)**
  - **False Negatives (FN)**

**Metrics computed:**
- Sensitivity (Recall)
- Specificity
- Precision
- Negative Predictive Value (NPV)
- Accuracy
- F-score

**Result summary (TRAIN_SIZE = 80):**
- True Positives: 1451
- True Negatives: 145
- False Positives: 181
- False Negatives: 51
- Sensitivity (Recall): 0.9660
- Specificity: 0.4448
- Precision: 0.8891
- Negative Predictive Value: 0.7398
- Accuracy: 0.8731
- F-score: 0.9260

**Observation:**
- The model performs well overall and is especially strong at identifying positive reviews.
- Specificity is lower, which means it is weaker at correctly identifying negative reviews.
- This is largely explained by the class imbalance in the dataset (83.3% POSITIVE vs 16.7% NEGATIVE).

---

### Step 7B: kNN Evaluation
- Test the kNN classifier on all documents in the test set
- Compare predicted labels with actual labels
- Count:
  - **True Positives (TP)**
  - **True Negatives (TN)**
  - **False Positives (FP)**
  - **False Negatives (FN)**

**Metrics computed:**
- Sensitivity (Recall)
- Specificity
- Precision
- Negative Predictive Value (NPV)
- Accuracy
- F-score

**Result summary (TRAIN_SIZE = 80):**
- True Positives: 1457
- True Negatives: 37
- False Positives: 289
- False Negatives: 45
- Sensitivity (Recall): 0.9700
- Specificity: 0.1135
- Precision: 0.8345
- Negative Predictive Value: 0.4512
- Accuracy: 0.8173
- F-score: 0.8972

**Observation:**
- kNN struggles significantly with identifying negative reviews (Specificity = 0.1135).
- Almost all test documents were classified as POSITIVE due to class imbalance in the dataset.
- kNN is less suited for high-dimensional text data compared to Naïve Bayes.

---

### Step 8: Display Results

- Output evaluation metrics in the exact format required by the assignment
- Metrics are printed after testing is complete:
  - **Number of true positives**
  - **Number of true negatives**
  - **Number of false positives**
  - **Number of false negatives**
  - **Sensitivity (recall)**
  - **Specificity**
  - **Precision**
  - **Negative predictive value**
  - **Accuracy**
  - **F-score**

---

### Step 9: Interactive Sentence Classification
- After testing, the user is prompted to enter a sentence
- The sentence goes through the same pipeline:
  - Preprocessing (lowercase, remove punctuation, tokenize)
  - Vectorization (Bag-of-Words)
  - Classification using the selected algorithm
- Output format:
```
  Sentence/document S: This product tastes amazing
  was classified as POSITIVE.
```
- For **Naïve Bayes only**, also prints:
```
  P(POSITIVE | S) = xxxx
  P(NEGATIVE | S) = xxxx
```

---

### Step 10: Repeat Classification Loop
- After each classification, the user is asked:
Do you want to enter another sentence [Y/N]?
- If **Y**: classify a new sentence without retraining
- If **N**: program exits

---

### Step 11: Exploratory Data Analysis (EDA)
- After seeing the results from the KNN which classified every review as POSITIVE, we tried analyzing the dataset before training to understand its structure and distribution

**Results:**
- Total samples: 9,138
- POSITIVE: 7,616 samples (83.3%)
- NEGATIVE: 1,522 samples (16.7%)
- Average review length (POSITIVE): 72.6 words
- Average review length (NEGATIVE): 86.7 words
- Shortest review: 10 words
- Longest review: 1,513 words

**Observation:**
- The dataset is heavily imbalanced with a 5:1 ratio of positive to negative reviews.
- Negative reviews tend to be longer on average, likely because unhappy customers write more to explain their complaints.
- This imbalance directly explains the low specificity of both classifiers on the original dataset.

---

### Step 12: Balanced Dataset Experiment
- To address class imbalance, the dataset was balanced by taking equal numbers of POSITIVE and NEGATIVE samples
- 1,522 POSITIVE and 1,522 NEGATIVE reviews were kept (3,044 total)
- The balanced dataset was shuffled before splitting to ensure even distribution across train and test sets

**Naïve Bayes results (Balanced, TRAIN_SIZE = 80):**
- True Positives: 248
- True Negatives: 278
- False Positives: 33
- False Negatives: 50
- Sensitivity (Recall): 0.8322
- Specificity: 0.8939
- Precision: 0.8826
- Negative Predictive Value: 0.8476
- Accuracy: 0.8637
- F-score: 0.8566

**kNN results (Balanced, TRAIN_SIZE = 80):**
- True Positives: 245
- True Negatives: 155
- False Positives: 156
- False Negatives: 53
- Sensitivity (Recall): 0.8221
- Specificity: 0.4984
- Precision: 0.6110
- Negative Predictive Value: 0.7452
- Accuracy: 0.6568
- F-score: 0.7010

**Observation:**
- Balancing the dataset significantly improved Naïve Bayes specificity from 0.4448 to 0.8939 — it now identifies negative reviews much more accurately.
- kNN improved slightly but still struggled, classifying most sentences as POSITIVE.
- Naïve Bayes is clearly better suited for this text classification task.

---

## Full Results Comparison

| Metric | NB Original | NB Balanced | kNN Original | kNN Balanced |
|---|---|---|---|---|
| True Positives | 1451 | 248 | 1457 | 245 |
| True Negatives | 145 | 278 | 37 | 155 |
| False Positives | 181 | 33 | 289 | 156 |
| False Negatives | 51 | 50 | 45 | 53 |
| Sensitivity | 0.9660 | 0.8322 | 0.9700 | 0.8221 |
| Specificity | 0.4448 | 0.8939 | 0.1135 | 0.4984 |
| Precision | 0.8891 | 0.8826 | 0.8345 | 0.6110 |
| NPV | 0.7398 | 0.8476 | 0.4512 | 0.7452 |
| Accuracy | 0.8731 | 0.8637 | 0.8173 | 0.6568 |
| F-score | 0.9260 | 0.8566 | 0.8972 | 0.7010 |


---

## Project Structure

CS481--amazon-reviews-sentiment-analysis/
├── data/
│   └── Reviews.csv              # Dataset (not tracked by git)
├── src/
│   ├── data_loader.py           # Step 1: Load and label dataset
│   ├── preprocessing.py         # Step 2: Lowercase, remove punctuation, tokenize
│   ├── vocabulary.py            # Step 3: Build vocabulary from all tokens
│   ├── vectorizer.py            # Step 4: Bag-of-Words vectorization
│   ├── splitter.py              # Step 5: Train/test split + balance dataset
│   ├── naive_bayes.py           # Step 6A: Naïve Bayes classifier
│   ├── knn.py                   # Step 6B: kNN classifier
│   ├── nb_evaluation.py         # Step 7A: Evaluate Naïve Bayes
│   ├── knn_evaluation.py        # Step 7B: Evaluate kNN
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── main.py                  # Main entry point (original dataset)
│   └── main_balanced.py         # Main entry point (balanced dataset)
├── .gitignore
└── README.md