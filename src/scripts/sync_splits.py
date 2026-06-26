"""
sync_phonemes.py

For each split CSV (train/test/validation), find its reference file by
stripping the _train/_test/_validation suffix, then overwrite any mismatched
"phoneme" values using the reference file as the source of truth.
"""

import csv
import os
import re

from src.utils.phoneme_utils import write_csv

CSV_PATHS = [
    "data/tatoeba/phonetic_tatoeba_gemini_3_train.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite_train.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split_train.csv",
    "data/stress-minimal/stress-minimal_single_split_train.csv",
    # --------------------
    "data/tatoeba/phonetic_tatoeba_gemini_3_test.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite_test.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split_test.csv",
    "data/stress-minimal/stress-minimal_single_split_test.csv",
    "data/tatoeba/phonetic_tatoeba_gemini_3_validation.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite_validation.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split_validation.csv",
    "data/stress-minimal/stress-minimal_single_split_validation.csv",
]

SPLIT_SUFFIX_RE = re.compile(r"_(train|test|validation)(?=\.csv$)")


def get_reference_path(split_path: str) -> str:
    """
    Derive the reference CSV path by removing the _train/_test/_validation
    suffix from the filename.

    e.g. "data/tatoeba/phonetic_tatoeba_gemini_3_train.csv"
      -> "data/tatoeba/phonetic_tatoeba_gemini_3.csv"
    """
    ref_path, n = SPLIT_SUFFIX_RE.subn("", split_path)
    if n == 0:
        raise ValueError(
            f"Could not find a _train/_test/_validation suffix in: {split_path!r}"
        )
    return ref_path


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sync_phonemes(split_path: str) -> None:
    ref_path = get_reference_path(split_path)

    if not os.path.exists(ref_path):
        print(f"  [SKIP] Reference file not found: {ref_path}")
        return

    ref_rows = read_csv(ref_path)
    split_rows = read_csv(split_path)

    # Build a lookup: sentence -> phoneme from the reference file.
    # If a sentence appears multiple times in the reference with different
    # phonemes, the last occurrence wins (consistent with simple dict lookup).
    ref_lookup: dict[str, str] = {row["sentence"]: row["phoneme"] for row in ref_rows}

    n_fixed = 0
    for row in split_rows:
        sentence = row["sentence"]
        if sentence not in ref_lookup:
            continue  # no matching sentence in reference — leave untouched

        ref_phoneme = ref_lookup[sentence]
        if row["phoneme"] != ref_phoneme:
            row["phoneme"] = ref_phoneme
            n_fixed += 1

    if n_fixed:
        write_csv(split_rows, split_path)
        print(f"  [FIXED] {n_fixed} row(s) updated  ->  {split_path}")
    else:
        print(f"  [OK]    No mismatches found       ->  {split_path}")


def main() -> None:
    for path in CSV_PATHS:
        print(f"Processing: {path}")
        try:
            sync_phonemes(path)
        except Exception as exc:
            print(f"  [ERROR] {exc}")


if __name__ == "__main__":
    main()
