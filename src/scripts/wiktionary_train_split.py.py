import os

import pandas as pd

# TODO: Do an audit of this and other similar scripts


def split_csv_by_pronunciation(file_path):
    # 1. Load the CSV file
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    df = pd.read_csv(file_path)

    # 2. Assign a row index count for each unique pronunciation group
    # This helps us identify the 1st, 2nd, 3rd, 4th, etc., occurrence of each pronunciation
    df["group_index"] = df.groupby("pronunciation").cumcount()

    # 3. Split the data based on the group index rules
    # First 2 entries (index 0 and 1) -> Test
    test_df = df[df["group_index"] < 2].copy()

    # Next 2 entries (index 2 and 3) -> Validation
    val_df = df[(df["group_index"] >= 2) & (df["group_index"] < 4)].copy()

    # Remaining entries (index 4 and above) -> Train
    train_df = df[df["group_index"] >= 4].copy()

    # 4. Drop the temporary helper column
    for data_split in [test_df, val_df, train_df]:
        data_split.drop(columns=["group_index"], inplace=True)

    # 5. Generate dynamic output file names based on your input path
    dir_name, base_name = os.path.split(file_path)
    file_name, file_ext = os.path.splitext(base_name)

    train_path = os.path.join(dir_name, f"{file_name}_train{file_ext}")
    test_path = os.path.join(dir_name, f"{file_name}_test{file_ext}")
    val_path = os.path.join(dir_name, f"{file_name}_validation{file_ext}")

    # 6. Save the new datasets to CSV
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)

    # Print out summary
    print(f"Splitting complete! Files saved in '{dir_name or './'}'")
    print(f" - Train rows: {len(train_df)} -> {os.path.basename(train_path)}")
    print(f" - Test rows: {len(test_df)} -> {os.path.basename(test_path)}")
    print(f" - Validation rows: {len(val_df)} -> {os.path.basename(val_path)}")


# --- HOW TO RUN IT ---
if __name__ == "__main__":
    # Replace this with whatever file path you want to process
    target_file = "data/wiktionary-scrape/transcribed/homographs_final.csv"

    split_csv_by_pronunciation(target_file)
