from data_loader import load_dataset
from preprocessing import preprocess_dataset

def build_vocabulary(df):
    vocab = set()
    for tokens in df["Tokens"]:
        for word in tokens:
            vocab.add(word)

    return sorted(list(vocab))

if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)

    vocab = build_vocabulary(df)

    print("Vocabulary size:", len(vocab))
    print(vocab[:20])