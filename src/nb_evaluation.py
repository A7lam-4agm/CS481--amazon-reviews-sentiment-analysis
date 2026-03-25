from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset
from splitter import split_dataset
from naive_bayes import NaiveBayesClassifier


if __name__ == "__main__":
    # Load and prepare data
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)

    train_df, test_df = split_dataset(df)

    # Train Naive Bayes
    nb = NaiveBayesClassifier()
    nb.train(train_df)

    # Counters
    tp = tn = fp = fn = 0

    # Test on ALL test data
    for _, row in test_df.iterrows():
        actual = row["Label"]
        predicted, _ = nb.predict(row["Vector"])

        if actual == "POSITIVE" and predicted == "POSITIVE":
            tp += 1
        elif actual == "NEGATIVE" and predicted == "NEGATIVE":
            tn += 1
        elif actual == "NEGATIVE" and predicted == "POSITIVE":
            fp += 1
        elif actual == "POSITIVE" and predicted == "NEGATIVE":
            fn += 1

    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nResults (Naive Bayes):")
    print("TP:", tp)
    print("TN:", tn)
    print("FP:", fp)
    print("FN:", fn)

    print("\nMetrics:")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("Specificity:", specificity)
    print("NPV      :", npv)
    print("F1 Score :", f1)
    