"""
This script allows you to inspect a CSV dataset (with a 'phoneme' column) for
invalid IPA characters.
"""

import csv
import os
import unicodedata

import questionary

from src.utils.dataset_files import CSV_PATHS
from src.utils.normalize_characters import normalize_characters
from src.utils.phoneme_inventory import PHONEME_INVENTORY
from src.utils.phoneme_utils import write_csv


def audit_and_wipe_csv(filepath):
    allowed_chars = set(PHONEME_INVENTORY)

    all_rows = []
    all_invalid_chars = {}
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
                if phoneme_string:
                    normalized = normalize_characters(phoneme_string)
                    bad_chars = [c for c in normalized if c not in allowed_chars]
                    for c in bad_chars:
                        all_invalid_chars[c] = all_invalid_chars.get(c, 0) + 1
                all_rows.append(row)
    except FileNotFoundError:
        print(f"  Error: File not found.")
        return
    except Exception as e:
        print(f"  Error: {e}")
        return

    print(f"  Rows scanned:    {total_rows}")
    if not all_invalid_chars:
        print(f"  No invalid characters found.")
        return

    rows_with_any_invalid = sum(
        1
        for row in all_rows
        if any(
            c in all_invalid_chars for c in normalize_characters(row.get("phoneme", ""))
        )
    )
    print(f"  Rows with invalid chars: {rows_with_any_invalid}")
    print(f"\n  {'Char':<8} {'Count':<8} {'Unicode':<12} Name")
    print(f"  {'-'*60}")

    for char, count in sorted(all_invalid_chars.items(), key=lambda x: -x[1]):
        unicode_hex = f"U+{ord(char):04X}"
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "UNKNOWN OR CONTROL CHARACTER"
        print(f"  {repr(char):<8} {count:<8} {unicode_hex:<12} {name}")

    # Checkbox selection

    choices = [
        questionary.Choice(
            title=f"{repr(char)}  ({all_invalid_chars[char]}x)  U+{ord(char):04X}",
            value=char,
        )
        for char, _ in sorted(all_invalid_chars.items(), key=lambda x: -x[1])
    ]
    selected = questionary.checkbox(
        "Select characters whose rows should be deleted:",
        choices=choices,
    ).ask()

    if not selected:
        print("  Nothing selected — file unchanged.")
        return

    selected_set = set(selected)
    kept_rows = []
    rows_wiped = 0

    for row in all_rows:
        phoneme_string = row.get("phoneme", "")
        normalized = normalize_characters(phoneme_string) if phoneme_string else ""
        if any(c in selected_set for c in normalized):
            rows_wiped += 1
        else:
            kept_rows.append(row)

    write_csv(kept_rows, filepath)
    print(f"\n  Rows wiped:      {rows_wiped}")
    print(f"  Rows kept:       {total_rows - rows_wiped}")


if __name__ == "__main__":
    for path in CSV_PATHS:
        print(f"\n{'='*64}")
        print(f"FILE: {path}")
        print(f"{'='*64}")
        if os.path.exists(path):
            audit_and_wipe_csv(path)
        else:
            print(f"  Skipping: file does not exist.")

    print()
