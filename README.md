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

## Project Structure
