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
- Training set: first `TRAIN_SIZE%` of samples
- Test set: remaining samples after training split

**Example:**
- Dataset size = 1000, `TRAIN_SIZE = 80`
- Training = first 800 samples
- Testing = last 200 samples

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

Output includes:

- Predicted class label for the test document

Note:

- kNN does not train a model in advance
- It stores all training data and makes predictions based on similarity
- Smaller distance means more similar documents


## Step 7A: Naïve Bayes Evaluation

- Test the Naïve Bayes classifier on all test documents
- Compare predicted labels with actual labels
- Count:
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)

Metrics computed:
- Accuracy
- Precision
- Recall
- Specificity
- Negative Predictive Value (NPV)
- F1 Score

Result summary:
- Accuracy ≈ 87%
- Precision ≈ 0.89
- Recall ≈ 0.97
- Specificity ≈ 0.44
- NPV ≈ 0.74
- F1 Score ≈ 0.93

Observation:
- The model performs well overall and is especially strong at identifying positive reviews.
- Specificity is lower, which means it is weaker at correctly identifying negative reviews.  
## Project Structure
