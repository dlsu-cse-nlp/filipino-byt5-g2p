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
    "\u0361",
]


def audit_csv(filepath):
    allowed_chars = set(PHONEME_INVENTORY)
    all_removed_chars = {}  # char -> count
    rows_with_invalid = 0
    total_invalid = 0
    total_rows = 0

    try:
        with open(filepath, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            if "phoneme" not in reader.fieldnames:
                print(f"  Error: No 'phoneme' column found.")
                return

            for row in reader:
                total_rows += 1
                phoneme_string = row.get("phoneme", "")
                if not phoneme_string:
                    continue

                normalized = normalize_characters(phoneme_string)
                bad_chars = [c for c in normalized if c not in allowed_chars]

                if bad_chars:
                    rows_with_invalid += 1
                    total_invalid += len(bad_chars)
                    for c in bad_chars:
                        all_removed_chars[c] = all_removed_chars.get(c, 0) + 1

    except FileNotFoundError:
        print(f"  Error: File not found.")
        return

    print(f"  Rows scanned:           {total_rows}")
    if not all_removed_chars:
        print(f"  No invalid characters found.")
        return

    print(f"  Rows with invalid chars: {rows_with_invalid}")
    print(f"  Total invalid chars:     {total_invalid}")
    print(f"  Unique invalid chars:    {len(all_removed_chars)}\n")
    print(f"  {'Char':<8} {'Count':<8} {'Unicode':<12} Name")
    print(f"  {'-'*60}")
    for char, count in sorted(all_removed_chars.items(), key=lambda x: -x[1]):
        unicode_hex = f"U+{ord(char):04X}"
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "UNKNOWN OR CONTROL CHARACTER"
        print(f"  {repr(char):<8} {count:<8} {unicode_hex:<12} {name}")


if __name__ == "__main__":
    csv_paths = [
        # "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        # "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
        # "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        # "data/stress-minimal/stress-minimal_single_split.csv",
        "data/wiktionary-scrape/transcribed/homographs_flattened.csv",
    ]

    for path in csv_paths:
        print(f"\n{'='*64}")
        print(f"FILE: {path}")
        print(f"{'='*64}")
        if os.path.exists(path):
            audit_csv(path)
        else:
            print(f"  Skipping: file does not exist.")
    print()
