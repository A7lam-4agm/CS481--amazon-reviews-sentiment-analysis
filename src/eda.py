import pandas as pd
from data_loader import load_dataset

def run_eda(df):
    print("=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # ── Dataset size ──────────────────────────────
    print(f"\nTotal samples: {len(df)}")

    # ── Label distribution ────────────────────────
    counts = df["Label"].value_counts()
    percentages = df["Label"].value_counts(normalize=True) * 100

    print("\nLabel distribution:")
    for label in counts.index:
        print(f"  {label}: {counts[label]} samples ({percentages[label]:.1f}%)")

    # ── Average review length ─────────────────────
    df["review_length"] = df["Text"].apply(lambda x: len(x.split()))
    print(f"\nAverage review length (words):")
    for label in counts.index:
        avg = df[df["Label"] == label]["review_length"].mean()
        print(f"  {label}: {avg:.1f} words")

    # ── Shortest and longest reviews ──────────────
    print(f"\nShortest review: {df['review_length'].min()} words")
    print(f"Longest review:  {df['review_length'].max()} words")

    print("=" * 50)


if __name__ == "__main__":
    df = load_dataset("../data/Reviews.csv")
    run_eda(df)