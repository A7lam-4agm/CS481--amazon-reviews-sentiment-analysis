from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset

TRAIN_SIZE = 80  # percentage

def split_dataset(df, train_size=TRAIN_SIZE):
    """
    Splits the dataset into training and test sets.
    Training set: first train_size% of samples
    Test set: last (100 - train_size)% of samples

    Example:
        Dataset size = 1000, train_size = 80
        Training = first 800 samples
        Testing  = last  200 samples
    """
    n = len(df)
    train_end = int(n * train_size / 100)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df  = df.iloc[train_end:].reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)

    train_df, test_df = split_dataset(df)

    print(f"Total samples : {len(df)}")
    print(f"Training set  : {len(train_df)} samples (first {TRAIN_SIZE}%)")
    print(f"Test set      : {len(test_df)} samples (last {100 - TRAIN_SIZE}%)")