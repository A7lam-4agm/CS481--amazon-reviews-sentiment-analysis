from knn import predict_knn

def evaluate_knn(train_df, test_df, k=3):
    """
    Runs the kNN classifier on every document in the test set
    and computes evaluation metrics.
    
    True Positive  (TP): predicted POSITIVE, actual POSITIVE
    True Negative  (TN): predicted NEGATIVE, actual NEGATIVE
    False Positive (FP): predicted POSITIVE, actual NEGATIVE
    False Negative (FN): predicted NEGATIVE, actual POSITIVE
    """
    print("Testing classifier...")

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(test_df)):
        test_vector  = test_df["Vector"].iloc[i]
        actual_label = test_df["Label"].iloc[i]

        predicted_label = predict_knn(train_df, test_vector, k)

        if predicted_label == "POSITIVE" and actual_label == "POSITIVE":
            TP += 1
        elif predicted_label == "NEGATIVE" and actual_label == "NEGATIVE":
            TN += 1
        elif predicted_label == "POSITIVE" and actual_label == "NEGATIVE":
            FP += 1
        elif predicted_label == "NEGATIVE" and actual_label == "POSITIVE":
            FN += 1

    # ── Compute metrics ───────────────────────────────────────────────────

    # Sensitivity (Recall) = TP / (TP + FN)
    # "Of all actual POSITIVES, how many did we catch?"
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity = TN / (TN + FP)
    # "Of all actual NEGATIVES, how many did we correctly reject?"
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Precision = TP / (TP + FP)
    # "Of all predicted POSITIVES, how many were actually POSITIVE?"
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Negative Predictive Value = TN / (TN + FN)
    # "Of all predicted NEGATIVES, how many were actually NEGATIVE?"
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0

    # Accuracy = (TP + TN) / total
    # "How many predictions were correct overall?"
    accuracy = (TP + TN) / len(test_df) if len(test_df) > 0 else 0

    # F-score = 2 * (precision * recall) / (precision + recall)
    # "Harmonic mean of precision and recall"
    fscore = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # ── Display results ───────────────────────────────────────────────────
    print("\nTest results / metrics:")
    print(f"Number of true positives:  {TP}")
    print(f"Number of true negatives:  {TN}")
    print(f"Number of false positives: {FP}")
    print(f"Number of false negatives: {FN}")
    print(f"Sensitivity (recall):      {sensitivity:.4f}")
    print(f"Specificity:               {specificity:.4f}")
    print(f"Precision:                 {precision:.4f}")
    print(f"Negative predictive value: {npv:.4f}")
    print(f"Accuracy:                  {accuracy:.4f}")
    print(f"F-score:                   {fscore:.4f}")

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "accuracy": accuracy,
        "fscore": fscore
    }


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

    evaluate_knn(train_df, test_df, k=3)