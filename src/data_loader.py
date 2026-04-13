import pandas as pd

def load_dataset(file_path):
    # NOTE: nrows is set to 50000 for kNN due to computational constraints
    # (kNN is O(n²) and the full 568k dataset would take days to run)
    # Naïve Bayes was run on the full dataset (remove nrows limit below to do so)
    df = pd.read_csv(file_path, nrows=50000)  # change to None for full dataset
    df = df[["Text", "Score"]]
    df = df.dropna(subset=["Text"])

    df = df[df["Score"] != 3].copy()

    df["Label"] = df["Score"].apply(
        lambda score: "POSITIVE" if score >= 4 else "NEGATIVE"
    )

    return df[["Text", "Label"]]


if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")

    print("First 5 rows:")
    print(df.head())

    print("\nLabel counts:")
    print(df["Label"].value_counts())
