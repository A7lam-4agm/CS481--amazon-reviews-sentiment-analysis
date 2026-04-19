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
  - Score = 3 (neutral) is dropped

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

**Note:**
The vocabulary includes numbers and product-related terms because numeric values were not removed during preprocessing. Per professor instructions, vocabulary is built from the entire dataset before splitting.

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

**Example:**
- k = 3
- Nearest neighbors labels = [POSITIVE, POSITIVE, NEGATIVE]
- Prediction = POSITIVE

**Note:**
- kNN does not train a model in advance
- It stores all training data and makes predictions based on similarity
- Smaller distance means more similar documents

---

### Step 7A: Naïve Bayes Evaluation
- Test the Naïve Bayes classifier on all test documents
- Compare predicted labels with actual labels

**Result summary (Full dataset, TRAIN_SIZE = 80):**
- True Positives: 84,902
- True Negatives: 11,411
- False Positives: 4,349
- False Negatives: 4,501
- Sensitivity (Recall): 0.9497
- Specificity: 0.7240
- Precision: 0.9513
- Negative Predictive Value: 0.7171
- Accuracy: 0.9158
- F-score: 0.9505

**Observation:**
- Naïve Bayes achieves 91.6% accuracy on the full dataset
- Specificity improved significantly with more data (0.44 → 0.72)
- Still slightly biased toward POSITIVE due to class imbalance (84.4% vs 15.6%)

---

### Step 7B: kNN Evaluation
- Test the kNN classifier on all documents in the test set
- Compare predicted labels with actual labels

**Result summary (50,000 rows, TRAIN_SIZE = 80):**
- True Positives: 7,409
- True Negatives: 230
- False Positives: 1,279
- False Negatives: 273
- Sensitivity (Recall): 0.9645
- Specificity: 0.1524
- Precision: 0.8528
- Negative Predictive Value: 0.4573
- Accuracy: 0.8311
- F-score: 0.9052

**Note on dataset size:**
Due to kNN's O(n²) complexity, we limited kNN to 50,000 rows. Naïve Bayes was run on the full 568,454 row dataset since it trains in minutes regardless of dataset size.

**Observation:**
- kNN struggles significantly with identifying negative reviews (Specificity = 0.1524)
- Almost all test documents classified as POSITIVE due to class imbalance
- kNN is less suited for high-dimensional text data compared to Naïve Bayes

---

### Step 8: Display Results
- Output evaluation metrics in the exact format required by the assignment
- Metrics are printed after testing is complete

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
```
  Do you want to enter another sentence [Y/N]?
```
- If **Y**: classify a new sentence without retraining
- If **N**: program exits

---

### Step 11: Exploratory Data Analysis (EDA)
- Analyzed the full dataset before training

**Results (Full dataset):**
- Total samples: 525,814
- POSITIVE: 443,777 samples (84.4%)
- NEGATIVE: 82,037 samples (15.6%)
- Average review length (POSITIVE): 77.3 words
- Average review length (NEGATIVE): 88.3 words
- Shortest review: 3 words
- Longest review: 2,520 words

**Observation:**
- Dataset is heavily imbalanced with a 5:1 ratio of positive to negative reviews
- Negative reviews are longer on average — unhappy customers write more detailed complaints
- This imbalance directly explains the low specificity of both classifiers

---

### Step 12: Balanced Dataset Experiment
- To address class imbalance, the dataset was balanced by taking equal numbers of POSITIVE and NEGATIVE samples
- 82,037 POSITIVE and 82,037 NEGATIVE reviews kept (164,074 total)
- The balanced dataset was shuffled before splitting to ensure even distribution

**Naïve Bayes results (Balanced, TRAIN_SIZE = 80):**
- True Positives: 14,538
- True Negatives: 14,333
- False Positives: 2,168
- False Negatives: 1,776
- Sensitivity (Recall): 0.8911
- Specificity: 0.8686
- Precision: 0.8702
- Negative Predictive Value: 0.8898
- Accuracy: 0.8798
- F-score: 0.8806

**kNN results (Balanced, 50k rows, TRAIN_SIZE = 80):**
- True Positives: 1,191
- True Negatives: 819
- False Positives: 706
- False Negatives: 298
- Sensitivity (Recall): 0.7999
- Specificity: 0.5370
- Precision: 0.6278
- Negative Predictive Value: 0.7332
- Accuracy: 0.6669
- F-score: 0.7035

**Observation:**
- Balancing dramatically improved NB specificity from 0.7240 to 0.8686
- NB balanced is now nearly symmetric — performs similarly on both classes
- kNN improved from 0.1524 to 0.5370 specificity but still underperforms
- Naïve Bayes is clearly better suited for this text classification task

---

## Full Results Comparison

| Metric | NB Full Dataset | NB Balanced | kNN 50k | kNN Balanced |
|---|---|---|---|---|
| True Positives | 84,902 | 14,538 | 7,409 | 1,191 |
| True Negatives | 11,411 | 14,333 | 230 | 819 |
| False Positives | 4,349 | 2,168 | 1,279 | 706 |
| False Negatives | 4,501 | 1,776 | 273 | 298 |
| Sensitivity | 0.9497 | 0.8911 | 0.9645 | 0.7999 |
| Specificity | 0.7240 | 0.8686 | 0.1524 | 0.5370 |
| Precision | 0.9513 | 0.8702 | 0.8528 | 0.6278 |
| NPV | 0.7171 | 0.8898 | 0.4573 | 0.7332 |
| Accuracy | 0.9158 | 0.8798 | 0.8311 | 0.6669 |
| F-score | 0.9505 | 0.8806 | 0.9052 | 0.7035 |

---

### ROC Curve Analysis

Real ROC curves were computed from actual classifier probability outputs:

- **Naïve Bayes** — 8,736 real threshold points computed from continuous 
  P(POSITIVE|S) scores. AUC = 0.9067.
- **kNN (k=3)** — 4 operating points computed from vote proportion scores 
  (0.0, 0.333, 0.667, 1.0). AUC = 0.5714.

ROC scores were computed using numpy vectorized matrix multiplication 
for efficiency. This is not part of the classifier implementation — 
it was used solely for visualization purposes.

---

## Project Structure

```
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
```
