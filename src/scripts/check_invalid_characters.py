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


def browse_rows_with_char(char, all_rows, fieldnames):
    """Page through rows whose phoneme field contains `char`."""
    matching = [
        row for row in all_rows if char in normalize_characters(row.get("phoneme", ""))
    ]
    if not matching:
        print("  No matching rows found.")
        return

    PAGE = 10
    total = len(matching)
    offset = 0
    sentence_key = next(
        (k for k in ("sentence", "text", "utterance") if k in fieldnames), None
    )

    while offset < total:
        page_rows = matching[offset : offset + PAGE]
        print(f"\n  Showing {offset + 1}–{min(offset + PAGE, total)} of {total} rows:")
        print(
            f"  {'#':<6} {'phoneme':<40}" + (f" {sentence_key}" if sentence_key else "")
        )
        print(f"  {'-'*80}")
        for i, row in enumerate(page_rows, start=offset + 1):
            phoneme = row.get("phoneme", "")
            display = phoneme[:38] + "…" if len(phoneme) > 38 else phoneme
            line = f"  {i:<6} {display:<40}"
            if sentence_key:
                sent = row.get(sentence_key, "")
                line += " " + (sent[:60] + "…" if len(sent) > 60 else sent)
            print(line)

        offset += PAGE
        if offset < total:
            cont = questionary.confirm("  Show next page?", default=True).ask()
            if not cont:
                break


def audit_and_wipe_csv(filepath):
    allowed_chars = set(PHONEME_INVENTORY)
    all_rows = []
    all_invalid_chars = {}
    total_rows = 0
    fieldnames = []

    try:
        with open(filepath, mode="r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            fieldnames = list(reader.fieldnames or [])
            if "phoneme" not in fieldnames:
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

    # Per-character action selection
    # actions: "delete" | "replace:<str>" | "skip"
    char_actions = {}  # char -> ("delete" | "replace", replacement)

    sorted_chars = sorted(all_invalid_chars.items(), key=lambda x: -x[1])

    for char, count in sorted_chars:
        unicode_hex = f"U+{ord(char):04X}"
        print(f"\n  {repr(char)}  ({count}x)  {unicode_hex}")

        action = questionary.select(
            "  Action:",
            choices=[
                questionary.Choice("Skip (leave rows unchanged)", value="skip"),
                questionary.Choice(
                    "Browse rows containing this character", value="browse"
                ),
                questionary.Choice(
                    "Replace all occurrences with another string", value="replace"
                ),
                questionary.Choice(
                    "Delete all rows containing this character", value="delete"
                ),
            ],
        ).ask()

        if action == "browse":
            browse_rows_with_char(char, all_rows, fieldnames)
            # After browsing, ask again what to do
            action = questionary.select(
                "  Now what?",
                choices=[
                    questionary.Choice("Skip (leave rows unchanged)", value="skip"),
                    questionary.Choice(
                        "Replace all occurrences with another string", value="replace"
                    ),
                    questionary.Choice(
                        "Delete all rows containing this character", value="delete"
                    ),
                ],
            ).ask()

        if action == "replace":
            replacement = questionary.text(
                f"  Replace {repr(char)} with (leave blank for empty string):",
            ).ask()
            if replacement is None:
                replacement = ""
            char_actions[char] = ("replace", replacement)
        elif action == "delete":
            char_actions[char] = ("delete", None)
        # "skip" -> not added to char_actions

    if not char_actions:
        print("  No actions selected — file unchanged.")
        return

    # Apply actions
    delete_chars = {c for c, (a, _) in char_actions.items() if a == "delete"}
    replace_map = {c: r for c, (a, r) in char_actions.items() if a == "replace"}

    kept_rows = []
    rows_wiped = 0
    rows_modified = 0

    for row in all_rows:
        phoneme_string = row.get("phoneme", "")
        normalized = normalize_characters(phoneme_string) if phoneme_string else ""

        if delete_chars and any(c in delete_chars for c in normalized):
            rows_wiped += 1
            continue

        if replace_map:
            new_phoneme = normalized
            changed = False
            for bad_char, replacement in replace_map.items():
                if bad_char in new_phoneme:
                    new_phoneme = new_phoneme.replace(bad_char, replacement)
                    changed = True
            if changed:
                row = dict(row)
                row["phoneme"] = new_phoneme
                rows_modified += 1

        kept_rows.append(row)

    write_csv(kept_rows, filepath)

    if rows_wiped:
        print(f"\n  Rows deleted:    {rows_wiped}")
    if rows_modified:
        print(f"  Rows modified:   {rows_modified}")
    print(f"  Rows kept:       {len(kept_rows)}")


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
