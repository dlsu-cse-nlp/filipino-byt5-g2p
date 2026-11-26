"""
Splits the Wiktionary-guided dataset into train, test, and validation splits by
taking 2 samples per word-phoneme pair
"""

import os

import pandas as pd


def split_csv_by_pronunciation(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    df = pd.read_csv(file_path)
    df["group_index"] = df.groupby("pronunciation").cumcount()

    test_df = df[df["group_index"] < 2].copy()
    val_df = df[(df["group_index"] >= 2) & (df["group_index"] < 4)].copy()
    train_df = df[df["group_index"] >= 4].copy()

    for data_split in [test_df, val_df, train_df]:
        data_split.drop(columns=["group_index"], inplace=True)

    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    train_path = os.path.join(dir_name, f"{file_name}_train{file_ext}")
    test_path = os.path.join(dir_name, f"{file_name}_test{file_ext}")
    val_path = os.path.join(dir_name, f"{file_name}_validation{file_ext}")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train: {len(train_df)} -> {os.path.basename(train_path)}")
    print(f"Test: {len(test_df)} -> {os.path.basename(test_path)}")
    print(f"Validation: {len(val_df)} -> {os.path.basename(val_path)}")


if __name__ == "__main__":
    target_file = "data/wiktionary-scrape/transcribed/homographs_final.csv"
    split_csv_by_pronunciation(target_file)
