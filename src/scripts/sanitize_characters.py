import csv
import os
import unicodedata

from src.utils.normalize_characters import normalize_characters

# TODO: Do an audit of this and other similar scripts

PHONEME_INVENTORY = [
    " ",
    "'",
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "s",
    "t",
    "u",
    "v",
    "w",
    "z",
    "ŋ",
    "ɕ",
    "ə",
    "ɡ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",
    "ˌ",
    "\u0361",  # Tie bar: Unicode U+0361
]


def sanitize_and_copy_csv(input_filepath, output_filepath):
    # Convert the inventory to a set for fast O(1) lookups
    allowed_chars = set(PHONEME_INVENTORY)

    # Trackers for our final summary
    all_removed_chars = set()
    rows_cleaned_count = 0
    total_chars_removed = 0

    # Use a temporary file path to prevent truncating the file mid-read
    temp_filepath = output_filepath + ".tmp"

    try:
        with open(input_filepath, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)

            # Verify the 'phoneme' column exists
            if "phoneme" not in reader.fieldnames:
                print(
                    f"Error: The CSV '{input_filepath}' does not contain a 'phoneme' column."
                )
                return

            # Write to the temporary file first
            with open(temp_filepath, mode="w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row_index, row in enumerate(
                    reader, start=2
                ):  # Start at 2 for header
                    phoneme_string = row.get("phoneme", "")

                    if not phoneme_string:
                        writer.writerow(row)
                        continue

                    # Apply normalization to the string
                    normalized_string = normalize_characters(phoneme_string)

                    # Identify bad characters
                    bad_chars_in_row = set(normalized_string) - allowed_chars

                    if bad_chars_in_row:
                        rows_cleaned_count += 1
                        all_removed_chars.update(bad_chars_in_row)

                        # Rebuild the string, keeping ONLY the allowed characters
                        cleaned_string = "".join(
                            char for char in normalized_string if char in allowed_chars
                        )

                        # Calculate how many characters are being dropped
                        chars_removed_in_row = len(normalized_string) - len(
                            cleaned_string
                        )
                        total_chars_removed += chars_removed_in_row

                        # Update the row dictionary with the clean string
                        row["phoneme"] = cleaned_string

                        print(
                            f"[!] Cleaned row {row_index}: Dropped {chars_removed_in_row} invalid char(s)."
                        )

                    # Write the (potentially cleaned) row to the temporary file
                    writer.writerow(row)

        # Atomic swap: safely replace the target file with our clean, completed temp file
        os.replace(temp_filepath, output_filepath)

    except FileNotFoundError:
        print(f"Error: The file '{input_filepath}' was not found.")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return

    print("\n" + "=" * 60)
    if all_removed_chars:
        print("CLEANING SUMMARY:")
        print(f" - Total invalid characters removed: {total_chars_removed}")
        print(f" - Total rows modified/cleaned: {rows_cleaned_count}")
        print(f" - Unique invalid characters filtered out: {len(all_removed_chars)}\n")

        print(f"{'Char':<8} | {'Unicode':<10} | {'Name'}")
        print("-" * 60)

        for char in sorted(all_removed_chars):
            unicode_hex = f"U+{ord(char):04X}"
            try:
                char_name = unicodedata.name(char)
            except ValueError:
                char_name = "UNKNOWN OR CONTROL CHARACTER"

            print(f"{repr(char):<8} | {unicode_hex:<10} | {char_name}")

        print(f"\nSUCCESS: Cleaned data safely applied to '{output_filepath}'")
    else:
        print("SUCCESS: No invalid characters were found. Target file left unchanged.")
    print("=" * 60 + "\n")


if __name__ == "__main__":

    csv_paths = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        "data/stress-minimal/stress-minimal_single_split.csv",
        "data/wiktionary-scrape/transcribed/homographs_final_fixed.csv",
    ]

    for x in csv_paths:
        if os.path.exists(x):
            sanitize_and_copy_csv(x, x)
        else:
            print(f"Skipping: '{x}' does not exist.")
