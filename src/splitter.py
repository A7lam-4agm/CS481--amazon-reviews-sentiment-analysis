from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary
from vectorizer import vectorize_dataset

TRAIN_SIZE = 80  # percentage

def split_dataset(df, train_size=TRAIN_SIZE):
    n = len(df)
    train_end  = int(n * train_size / 100)
    test_start = int(n * 0.80)          # test is ALWAYS last 20%

    train_df = df.iloc[:train_end].reset_index(drop=True)
    test_df  = df.iloc[test_start:].reset_index(drop=True)

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