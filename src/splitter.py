import pandas as pd
from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset

TRAIN_SIZE = 80

def split_dataset(df, train_size=TRAIN_SIZE):
    n = len(df)
    train_end  = int(n * train_size / 100)
    test_start = int(n * 0.80)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df  = df.iloc[test_start:].reset_index(drop=True)

    return train_df, test_df

def balance_dataset(df):
    """
    Balances the dataset by taking equal numbers of
    POSITIVE and NEGATIVE samples.
    Shuffles to ensure even distribution across train/test split.
    """
    positive = df[df["Label"] == "POSITIVE"]
    negative = df[df["Label"] == "NEGATIVE"]

    min_count = min(len(positive), len(negative))

    balanced = pd.concat([
        positive.iloc[:min_count],
        negative.iloc[:min_count]
    ]).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"Balanced dataset: {min_count} POSITIVE, {min_count} NEGATIVE")
    print(f"Total samples after balancing: {len(balanced)}")
    return balanced

if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)

    balanced_df = balance_dataset(df)
    train_df, test_df = split_dataset(balanced_df)

    print(f"\nTraining set : {len(train_df)} samples")
    print(f"Test set     : {len(test_df)} samples")