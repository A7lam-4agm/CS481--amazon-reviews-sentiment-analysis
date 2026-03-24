import string
from data_loader import load_dataset

def preprocess(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 3. Tokenize
    tokens = text.split()
    return tokens

def preprocess_dataset(df):
    """
    Takes the DataFrame from load_dataset()
    and adds a 'Tokens' column with the preprocessed words.
    Returns the updated DataFrame.
    """
    df = df.copy()
    df["Tokens"] = df["Text"].apply(preprocess)
    return df

if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    df = preprocess_dataset(df)

    print("First 3 rows:")
    print(df[["Text", "Label", "Tokens"]].head(3))

    print("\nSample tokens from first review:")
    print(df["Tokens"].iloc[0][:10])