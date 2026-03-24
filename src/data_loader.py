import pandas as pd

def load_dataset(file_path):
    df = pd.read_csv(file_path, nrows=10000)
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
