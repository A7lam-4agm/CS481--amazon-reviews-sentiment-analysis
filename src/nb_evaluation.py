from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset
from splitter import split_dataset
from naive_bayes import NaiveBayesClassifier


def prepare_data(train_size=80):
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)

    train_df, test_df = split_dataset(df, train_size)
    return train_df, test_df, vocab


def safe_divide(a, b):
    return a / b if b != 0 else 0


def compute_metrics(tp, tn, fp, fn):
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    f_score = safe_divide(2 * precision * sensitivity, precision + sensitivity)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "accuracy": accuracy,
        "f_score": f_score
    }


def evaluate_naive_bayes(train_df, test_df):
    nb = NaiveBayesClassifier()
    nb.train(train_df)
    print("Testing classifier...")

    tp = tn = fp = fn = 0

    for _, row in test_df.iterrows():
        test_vector = row["Vector"]
        actual_label = row["Label"]

        predicted_label, _ = nb.predict(test_vector)

        if predicted_label == "POSITIVE" and actual_label == "POSITIVE":
            tp += 1
        elif predicted_label == "NEGATIVE" and actual_label == "NEGATIVE":
            tn += 1
        elif predicted_label == "POSITIVE" and actual_label == "NEGATIVE":
            fp += 1
        elif predicted_label == "NEGATIVE" and actual_label == "POSITIVE":
            fn += 1

    metrics = compute_metrics(tp, tn, fp, fn)
    return nb, metrics


def print_metrics(metrics):
    print("\nTest results / metrics:")
    print(f"Number of true positives: {metrics['tp']}")
    print(f"Number of true negatives: {metrics['tn']}")
    print(f"Number of false positives: {metrics['fp']}")
    print(f"Number of false negatives: {metrics['fn']}")
    print(f"Sensitivity (recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Negative predictive value: {metrics['npv']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F-score: {metrics['f_score']:.4f}")


def run_nb_evaluation(train_size=80):
    train_df, test_df, vocab = prepare_data(train_size)
    nb_model, metrics = evaluate_naive_bayes(train_df, test_df)
    return nb_model, metrics, vocab


if __name__ == "__main__":
    nb_model, metrics, vocab = run_nb_evaluation(train_size=80)
    print_metrics(metrics)
    