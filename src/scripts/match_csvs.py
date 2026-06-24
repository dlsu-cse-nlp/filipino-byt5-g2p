import csv


def load_csv_pairs(filename):
    """Reads a CSV file and returns a set of (word, original_sentence) tuples."""
    pairs = set()

    with open(filename, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Check if the required headers exist
        if (
            not reader.fieldnames
            or "word" not in reader.fieldnames
            or "original_sentence" not in reader.fieldnames
        ):
            raise ValueError(
                f"Error: '{filename}' must contain both 'word' and 'original_sentence' columns."
            )

        for row in reader:
            # Safely get the values, default to empty string if missing, and strip whitespace
            word = row.get("word", "").strip()
            original_sentence = row.get("original_sentence", "").strip()

            # Add as a composite tuple key
            pairs.add((word, original_sentence))

    return pairs


def compare_csvs(csv1_path, csv2_path):
    # Load data from both files
    try:
        set1 = load_csv_pairs(csv1_path)
        set2 = load_csv_pairs(csv2_path)
    except Exception as e:
        print(e)
        return

    print("--- Dataset Sizes ---")
    print(f"File 1 ('{csv1_path}'): {len(set1)} unique entries.")
    print(f"File 2 ('{csv2_path}'): {len(set2)} unique entries.")
    print("-" * 21)

    # 1. Check if everything in one is in the other
    missing_in_file2 = set1 - set2  # Items in 1 but not in 2
    missing_in_file1 = set2 - set1  # Items in 2 but not in 1

    if not missing_in_file2 and not missing_in_file1:
        print("✅ Perfect Match: Both files contain the exact same entries.")
        return

    # 2. Check who has more entries
    if len(set1) > len(set2):
        print(
            f"📊 File 1 has MORE entries than File 2 (+{len(set1) - len(set2)} net differences)."
        )
    elif len(set2) > len(set1):
        print(
            f"📊 File 2 has MORE entries than File 1 (+{len(set2) - len(set1)} net differences)."
        )
    else:
        print("📊 Both files have the same total count, but contain different entries.")

    # 3. Find and display the missing ones (Capped at 10 previews to prevent terminal flooding)
    MAX_PREVIEWS = 10

    if missing_in_file2:
        print(
            f"\n❌ Missing from File 2 (Found in File 1): {len(missing_in_file2)} items"
        )
        for i, (w, p) in enumerate(sorted(missing_in_file2), 1):
            if i <= MAX_PREVIEWS:
                print(f"   {i}. Word: '{w}' | original_sentence: '{p}'")
            else:
                print(f"   ... and {len(missing_in_file2) - MAX_PREVIEWS} more items.")
                break

    if missing_in_file1:
        print(
            f"\n❌ Missing from File 1 (Found in File 2): {len(missing_in_file1)} items"
        )
        for i, (w, p) in enumerate(sorted(missing_in_file1), 1):
            if i <= MAX_PREVIEWS:
                print(f"   {i}. Word: '{w}' | original_sentence: '{p}'")
            else:
                print(f"   ... and {len(missing_in_file1) - MAX_PREVIEWS} more items.")
                break


# --- Run the Script ---
if __name__ == "__main__":
    # Adjust these file paths to match your local files
    FILE_ONE = "ipa_perfect_line_data.csv"
    FILE_TWO = "no_ipa_perfect_line_data.csv"

    compare_csvs(FILE_ONE, FILE_TWO)
