import sys
from data_loader import load_dataset
from preprocessing import preprocess_dataset, preprocess
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset, vectorize_document
from splitter import split_dataset
from nb_evaluation import evaluate_naive_bayes, print_metrics
from knn_evaluation import evaluate_knn
from knn import predict_knn

# ─────────────────────────────────────────────
# COMMAND LINE ARGUMENTS
# ─────────────────────────────────────────────

def parse_args():
    """
    Parses command line arguments.
    Usage: python main.py ALGO TRAIN_SIZE
        ALGO       : 0 = Naïve Bayes, 1 = kNN (default: 0)
        TRAIN_SIZE : 50–80 (default: 80)
    """
    ALGO       = 0
    TRAIN_SIZE = 80

    if len(sys.argv) == 3:
        try:
            algo       = int(sys.argv[1])
            train_size = int(sys.argv[2])
            if algo in [0, 1] and 50 <= train_size <= 80:
                ALGO       = algo
                TRAIN_SIZE = train_size
        except ValueError:
            pass  # use defaults if invalid

    return ALGO, TRAIN_SIZE


# ─────────────────────────────────────────────
# INTERACTIVE CLASSIFICATION (Steps 9 & 10)
# ─────────────────────────────────────────────

def interactive_classification(algo, nb_classifier, train_df, vocab):
    """
    Prompts the user to enter a sentence and classifies it.
    Loops until the user enters N.
    Does not retrain the classifier between inputs.
    """
    while True:
        print("\nEnter your sentence/document:")
        sentence = input().strip()

        # Step 1 — preprocess the sentence
        tokens = preprocess(sentence)

        # Step 2 — convert to Bag-of-Words vector
        vector = vectorize_document(tokens, set(vocab))

        # Step 3 — classify using selected algorithm
        if algo == 0:
            # Naïve Bayes — also prints probabilities
            predicted, probs = nb_classifier.predict(vector)
            print(f"\nSentence/document S: {sentence}")
            print(f"was classified as {predicted}.")
            print(f"P(POSITIVE | S) = {probs.get('POSITIVE', 0):.6f}")
            print(f"P(NEGATIVE | S) = {probs.get('NEGATIVE', 0):.6f}")
        else:
            # kNN — just prints the label
            predicted = predict_knn(train_df, vector)
            print(f"\nSentence/document S: {sentence}")
            print(f"was classified as {predicted}.")

        # Step 10 — ask to repeat
        print("\nDo you want to enter another sentence [Y/N]?")
        answer = input().strip().upper()
        if answer != "Y":
            break


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ALGO, TRAIN_SIZE = parse_args()

    algo_name = "Naive Bayes" if ALGO == 0 else "K Nearest Neighbors"

    print("Last Name, First Name, AXXXXXXXX solution:")
    print(f"Training set size: {TRAIN_SIZE} %")
    print(f"Classifier type: {algo_name}")

    # ── Load and prepare data ─────────────────
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)
    train_df, test_df = split_dataset(df, TRAIN_SIZE)

    # ── Train and Test ────────────────────────
    nb = None
    if ALGO == 0:
        print("Training classifier...")
        nb, metrics = evaluate_naive_bayes(train_df, test_df)
        print_metrics(metrics)
    else:
        print("Training classifier...")
        evaluate_knn(train_df, test_df, k=3)

    # ── Interactive classification ────────────
    interactive_classification(ALGO, nb, train_df, vocab)


if __name__ == "__main__":
    main()