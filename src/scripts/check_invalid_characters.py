import csv
import os
import unicodedata

from src.utils.normalize_characters import normalize_characters

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
    # "ə",
    "ɡ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",
    "ˌ",
    "\u0361",
]


def audit_and_wipe_csv(filepath):
    allowed_chars = set(PHONEME_INVENTORY)
    all_invalid_chars = {}  # char -> count
    rows_wiped = 0
    total_rows = 0

    temp_filepath = filepath + ".tmp"

    try:
        with open(filepath, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if "phoneme" not in reader.fieldnames:
                print(f"  Error: No 'phoneme' column found.")
                return

            with open(temp_filepath, mode="w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()

                for row in reader:
                    total_rows += 1
                    phoneme_string = row.get("phoneme", "")

                    if phoneme_string:
                        normalized = normalize_characters(phoneme_string)
                        bad_chars = [c for c in normalized if c not in allowed_chars]

                        if bad_chars:
                            rows_wiped += 1
                            for c in bad_chars:
                                all_invalid_chars[c] = all_invalid_chars.get(c, 0) + 1
                            continue  # drop the row

                    writer.writerow(row)

        os.replace(temp_filepath, filepath)

    except FileNotFoundError:
        print(f"  Error: File not found.")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return
    except Exception as e:
        print(f"  Error: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        return

    print(f"  Rows scanned:    {total_rows}")
    print(f"  Rows wiped:      {rows_wiped}")
    print(f"  Rows kept:       {total_rows - rows_wiped}")

    if all_invalid_chars:
        print(f"\n  {'Char':<8} {'Count':<8} {'Unicode':<12} Name")
        print(f"  {'-'*60}")
        for char, count in sorted(all_invalid_chars.items(), key=lambda x: -x[1]):
            unicode_hex = f"U+{ord(char):04X}"
            try:
                name = unicodedata.name(char)
            except ValueError:
                name = "UNKNOWN OR CONTROL CHARACTER"
            print(f"  {repr(char):<8} {count:<8} {unicode_hex:<12} {name}")
    else:
        print(f"  No invalid characters found.")


if __name__ == "__main__":
    csv_paths = [
        # "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        # "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
        # "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        # "data/stress-minimal/stress-minimal_single_split.csv",
        # "data/wiktionary-scrape/transcribed/homographs_flattened.csv",
        "output_normalized.csv"
    ]

    for path in csv_paths:
        print(f"\n{'='*64}")
        print(f"FILE: {path}")
        print(f"{'='*64}")
        if os.path.exists(path):
            audit_and_wipe_csv(path)
        else:
            print(f"  Skipping: file does not exist.")
    print()
