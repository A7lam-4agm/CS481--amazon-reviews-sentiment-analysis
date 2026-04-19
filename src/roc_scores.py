import numpy as np
from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset
from splitter import split_dataset
from naive_bayes import NaiveBayesClassifier
import csv

TRAIN_SIZE = 80

def get_nb_scores(nb, test_df):
    scores = []
    print("Getting NB scores...")
    for i, (_, row) in enumerate(test_df.iterrows()):
        actual = 1 if row["Label"] == "POSITIVE" else 0
        _, probs = nb.predict(row["Vector"])
        scores.append((actual, probs.get("POSITIVE", 0)))
        if i % 1000 == 0:
            print(f"NB progress: {i}/{len(test_df)}")
    return scores

def get_knn_scores_fast(train_df, test_df, k=3):
    print("Building vocabulary index...")
    all_words = sorted(set(
        word for vec in train_df["Vector"] for word in vec.keys()
    ))
    word_to_idx = {word: i for i, word in enumerate(all_words)}
    vocab_size = len(all_words)
    print(f"Vocab size: {vocab_size}")

    print("Converting training vectors to matrix...")
    train_matrix = np.zeros((len(train_df), vocab_size), dtype=np.float32)
    for i, vec in enumerate(train_df["Vector"]):
        for word, count in vec.items():
            if word in word_to_idx:
                train_matrix[i, word_to_idx[word]] = count
        if i % 5000 == 0:
            print(f"  Train matrix progress: {i}/{len(train_df)}")
    train_labels = np.array([1 if l == "POSITIVE" else 0 for l in train_df["Label"]])

    print("Converting test vectors to matrix...")
    test_matrix = np.zeros((len(test_df), vocab_size), dtype=np.float32)
    for i, vec in enumerate(test_df["Vector"]):
        for word, count in vec.items():
            if word in word_to_idx:
                test_matrix[i, word_to_idx[word]] = count
    test_labels = np.array([1 if l == "POSITIVE" else 0 for l in test_df["Label"]])

    print("Computing ALL distances at once...")
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    train_sq = (train_matrix ** 2).sum(axis=1)
    test_sq  = (test_matrix ** 2).sum(axis=1)
    dot      = test_matrix @ train_matrix.T
    dist_sq  = test_sq[:, None] + train_sq[None, :] - 2 * dot
    dist_sq  = np.maximum(dist_sq, 0)

    print("Getting k nearest neighbors for all test docs...")
    nearest_indices = np.argsort(dist_sq, axis=1)[:, :k]
    neighbor_labels = train_labels[nearest_indices]
    pos_votes = neighbor_labels.sum(axis=1)
    probs = pos_votes / k

    scores = list(zip(test_labels.tolist(), probs.tolist()))
    print(f"Done! {len(scores)} scores computed.")
    return scores

def compute_roc(scores):
    thresholds = sorted(set([s[1] for s in scores]), reverse=True)
    roc_points = [(0.0, 0.0)]
    total_pos = sum(1 for actual, _ in scores if actual == 1)
    total_neg = sum(1 for actual, _ in scores if actual == 0)
    for thresh in thresholds:
        tp = sum(1 for actual, prob in scores if prob >= thresh and actual == 1)
        fp = sum(1 for actual, prob in scores if prob >= thresh and actual == 0)
        tpr = tp / total_pos if total_pos > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        roc_points.append((fpr, tpr))
    roc_points.append((1.0, 1.0))
    return roc_points

def save_roc(points, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr"])
        for fpr, tpr in points:
            writer.writerow([round(fpr, 6), round(tpr, 6)])
    print(f"Saved {len(points)} points to {filename}")

if __name__ == "__main__":
    print("Loading data...")
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)
    train_df, test_df = split_dataset(df, TRAIN_SIZE)

    # ── Naïve Bayes ──────────────────────────────
    print("\nTraining Naïve Bayes...")
    nb = NaiveBayesClassifier()
    nb.train(train_df)
    nb_scores = get_nb_scores(nb, test_df)
    nb_roc = compute_roc(nb_scores)
    save_roc(nb_roc, "../nb_roc.csv")
    print(f"NB done — {len(nb_roc)} ROC points")

    # ── kNN fast ─────────────────────────────────
    print("\nRunning fast kNN...")
    knn_scores = get_knn_scores_fast(train_df, test_df, k=3)
    knn_roc = compute_roc(knn_scores)
    save_roc(knn_roc, "../knn_roc.csv")
    print(f"kNN done — {len(knn_roc)} ROC points")

    print("\nAll done! Share nb_roc.csv and knn_roc.csv.")