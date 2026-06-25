"""
strip_vowel_diacritics.py

Reads a CSV file and replaces vowels with diacritics (accent/grave/circumflex/etc.)
with their plain ASCII counterparts, writing the result to a new CSV file.

Usage:
    python strip_vowel_diacritics.py input.csv [output.csv]

If output.csv is omitted, the result is written to <input>_stripped.csv.
"""

import csv
import sys
import unicodedata
from pathlib import Path

# Only normalize characters that are vowels (case-insensitive base).
VOWEL_BASES = set("aeiouAEIOU")


def strip_vowel_diacritics(text: str) -> str:
    """
    Replace accented/diacritical vowels in *text* with their plain ASCII
    equivalents, leaving all other characters (including accented consonants)
    untouched.

    Strategy: decompose each character via NFD (base letter + combining marks),
    keep the base letter if it is a vowel, otherwise keep the original character.
    """
    result = []
    for char in text:
        # Decompose into base character + combining diacritical marks
        decomposed = unicodedata.normalize("NFD", char)
        base = decomposed[0]  # first code point is always the base letter

        if base in VOWEL_BASES and len(decomposed) > 1:
            # It's a vowel with at least one diacritic — keep only the base
            result.append(base)
        else:
            result.append(char)

    return "".join(result)


def process_csv(input_path: Path, output_path: Path) -> tuple[int, int]:
    """
    Read *input_path*, apply strip_vowel_diacritics to every cell, and write
    to *output_path*.  Returns (rows_processed, cells_changed).
    """
    rows_processed = 0
    cells_changed = 0

    # Sniff the dialect so we preserve the original delimiter, quoting, etc.
    with input_path.open(newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel  # fall back to standard comma-separated

    with (
        input_path.open(newline="", encoding="utf-8") as infile,
        output_path.open("w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.reader(infile, dialect)
        writer = csv.writer(
            outfile, dialect, quoting=csv.QUOTE_MINIMAL, escapechar="\\"
        )

        for row in reader:
            new_row = []
            for cell in row:
                cleaned = strip_vowel_diacritics(cell)
                if cleaned != cell:
                    cells_changed += 1
                new_row.append(cleaned)
            writer.writerow(new_row)
            rows_processed += 1

    return rows_processed, cells_changed


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python strip_vowel_diacritics.py input.csv [output.csv]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found — {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_stem(input_path.stem + "_stripped")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    rows, changed = process_csv(input_path, output_path)

    print(f"Done  : {rows} rows processed, {changed} cells modified.")


if __name__ == "__main__":
    main()
