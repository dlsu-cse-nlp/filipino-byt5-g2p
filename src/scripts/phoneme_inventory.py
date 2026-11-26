import os

import pandas as pd


def get_unique_phoneme_chars(file_paths):
    unique_chars = set()

    for file_path in file_paths:
        sep = "\t" if file_path.endswith(".tsv") else ","

        df = pd.read_csv(file_path, sep=sep, usecols=["phoneme"])

        phonemes = df["phoneme"].dropna().astype(str)

        for entry in phonemes:
            unique_chars.update(list(entry))

    return sorted(list(unique_chars))


files_to_process = [
    "data/manual_test.csv",
    "data/multitask/multitask_ds.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split.csv",
    "data/stress-minimal/stress-minimal_single_split.csv",
    "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
    "data/wikipron/wikipron_tl.tsv",
]

chars = get_unique_phoneme_chars(files_to_process)

print("=" * 40)
print(f"Unique characters ({len(chars)} total):")
print("=" * 40)
print(chars)
