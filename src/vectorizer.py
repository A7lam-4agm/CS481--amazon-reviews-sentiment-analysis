from data_loader import load_dataset
from preprocessing import preprocess_dataset
from vocabulary import build_vocabulary

def vectorize_document(tokens, vocab):
    vector = {}
    for word in tokens:
        if word in vocab:
            if word in vector:
                vector[word] += 1
            else:
                vector[word] = 1
    return vector

def vectorize_dataset(df, vocab):
    vocab_set = set(vocab)  # faster lookup
    df = df.copy()
    df["Vector"] = df["Tokens"].apply(lambda tokens: vectorize_document(tokens, vocab_set))
    return df

if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)
    vocab = build_vocabulary(df)
    df = vectorize_dataset(df, vocab)

    print("Vocabulary size:", len(vocab))
    print("\nFirst document tokens:")
    print(df["Tokens"].iloc[0][:10])
    print("\nFirst document vector (first 10 entries):")
    first_vector = df["Vector"].iloc[0]
    for word in list(first_vector.keys())[:10]:
        print(f"  {word} → {first_vector[word]}")