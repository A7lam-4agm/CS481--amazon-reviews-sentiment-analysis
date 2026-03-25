import math

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}           # P(POSITIVE), P(NEGATIVE)
        self.likelihood = {}      # P(word | class)
        self.vocab_size = 0
        self.word_counts = {}     # raw word counts per class
        self.class_totals = {}    # total word count per class

    # ─────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────

    def train(self, train_df):
        """
        Computes prior and likelihood probabilities from the training set.
        Uses Add-1 (Laplace) smoothing for likelihoods.
        """
        print("Training classifier...")

        total_docs = len(train_df)
        classes = train_df["Label"].unique()  # ["POSITIVE", "NEGATIVE"]

        # self.word_counts stores how many times each word appears in each class
        self.word_counts = {c: {} for c in classes}

        # self.class_totals stores the TOTAL number of words in each class
        # This is used as the denominator when computing likelihoods
        self.class_totals = {c: 0 for c in classes}

        # ── Loop through every document in the training set ──────────────────
        for _, row in train_df.iterrows():
            label  = row["Label"] 
            vector = row["Vector"]  

            # for each word and its count in this document
            for word, count in vector.items():

                # add the word to the class word count dict if not seen before
                if word not in self.word_counts[label]:
                    self.word_counts[label][word] = 0

                # add the word's count to the class total
                self.word_counts[label][word] += count

                # also add to the total word count for this class
                self.class_totals[label] += count

        # ── Build vocabulary from all words seen across all classes ──────────
        # We need the vocab size |V| for Add-1 smoothing
        all_words = set()
        for c in classes:
            all_words.update(self.word_counts[c].keys())
        self.vocab_size = len(all_words)  # |V|

        # ── Compute prior probabilities ───────────────────────────────────────
        # P(POSITIVE) = number of POSITIVE docs / total docs
        # P(NEGATIVE) = number of NEGATIVE docs / total docs
        class_counts = train_df["Label"].value_counts()
        for c in classes:
            self.prior[c] = class_counts[c] / total_docs

        # ── Compute likelihood probabilities with Add-1 smoothing ────────────
        # P(word | class) = (count(word in class) + 1) / (total words in class + |V|)
        # Use add-1 smoothing to pretend every word appeared at least once.
        self.likelihood = {c: {} for c in classes}
        for c in classes:
            denominator = self.class_totals[c] + self.vocab_size
            for word in all_words:
                count = self.word_counts[c].get(word, 0)  # 0 if word never appeared in this class
                self.likelihood[c][word] = (count + 1) / denominator

    # ─────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────

    def predict(self, vector):
        """
        Predicts the class of a single document vector.
        Uses log probabilities to avoid underflow.
        Returns (predicted_label, {class: probability})
        """
        classes = list(self.prior.keys())
        log_scores = {}

        for c in classes:
            # start with log prior
            log_score = math.log(self.prior[c])
            denominator = self.class_totals[c] + self.vocab_size

            for word, count in vector.items():
                if word in self.likelihood[c]:
                    # add log likelihood for each word occurrence
                    log_score += count * math.log(self.likelihood[c][word])
                else:
                    # word not seen in training, apply smoothing
                    smoothed = 1 / denominator
                    log_score += count * math.log(smoothed)

            log_scores[c] = log_score

        # ── Convert log scores back to probabilities ──────────────────────
        # subtract max log score for numerical stability before exponentiating
        max_log = max(log_scores.values())
        raw = {c: math.exp(log_scores[c] - max_log) for c in classes}

        # normalize so probabilities sum to 1
        total = sum(raw.values())
        probabilities = {c: raw[c] / total for c in classes}

        # predicted label is the class with the highest probability
        predicted_label = max(probabilities, key=probabilities.get)
        return predicted_label, probabilities


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import preprocess_dataset
    from vocabulary import build_vocabulary
    from vectorizer import vectorize_dataset
    from splitter import split_dataset

    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)
    train_df, test_df = split_dataset(df)

    nb = NaiveBayesClassifier()
    nb.train(train_df)

    # test on first 3 samples
    print("\nSample predictions:")
    for i, row in test_df.head(3).iterrows():
        predicted, probs = nb.predict(row["Vector"])
        print(f"\nActual: {row['Label']} | Predicted: {predicted}")
        print(f"  P(POSITIVE | S) = {probs.get('POSITIVE', 0):.6f}")
        print(f"  P(NEGATIVE | S) = {probs.get('NEGATIVE', 0):.6f}")