import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def analyze_sentence_distribution(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["phoneme"])
    df = df[df["phoneme"].str.strip().astype(bool)]
    df["length"] = df["phoneme"].str.len()

    if "sentence" not in df.columns:
        print("Error: CSV must contain a 'sentence' column.")
        return

    lengths = df["sentence"].dropna().str.split().str.len()

    mean_len = lengths.mean()
    std_len = lengths.std()

    print(f"Mean length: {mean_len:.2f} words")
    print(f"Std: {std_len:.2f} words")
    print(f"Count: {len(lengths)} sentences")

    stat, p = stats.normaltest(lengths)

    print(f"D'Agostino's K^2: {stat:.3f}, p-value: {p:.3e}")
    if p > 0.05:
        print("Normal distribution")
    else:
        print("Not a normal distribution")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(lengths, kde=True, color="skyblue")
    plt.title("Sentence length distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    stats.probplot(lengths, dist="norm", plot=plt)
    plt.title("Q-Q plot")

    plt.tight_layout()
    plt.show()


analyze_sentence_distribution("data/manual_set.csv")
