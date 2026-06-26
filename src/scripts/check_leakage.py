#!/usr/bin/env python3
"""
check_leakage.py — Check for sentence-level data leakage between one or more
test CSVs and one or more train/split CSV files.

Usage:
    python check_leakage.py --tests test.csv --splits train.csv dev.csv
    python check_leakage.py --tests test1.csv test2.csv --splits train.csv dev.csv
    python check_leakage.py --tests *.test.csv --splits *.train.csv --ignore-case
    python check_leakage.py --tests test.csv --splits train.csv --output report.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


def load_sentences(path: Path, column: str, ignore_case: bool) -> dict[str, list[int]]:
    """
    Read `column` from a CSV file.
    Returns a dict mapping (normalised) sentence → list of 1-based row numbers.
    Raises FileNotFoundError / KeyError with clear messages.
    """
    sentences: dict[str, list[int]] = {}
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or column not in reader.fieldnames:
                available = reader.fieldnames or []
                raise KeyError(
                    f"Column '{column}' not found in {path}.\n"
                    f"  Available columns: {available}"
                )
            for row_num, row in enumerate(reader, start=2):  # row 1 = header
                raw = row[column]
                key = raw.strip().lower() if ignore_case else raw.strip()
                sentences.setdefault(key, []).append(row_num)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except KeyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    return sentences


def find_leaks(
    test_sentences: dict[str, list[int]],
    split_sentences: dict[str, list[int]],
) -> list[tuple[str, list[int], list[int]]]:
    """
    Return list of (sentence, test_rows, split_rows) for every overlap.
    """
    leaks = []
    for sentence, test_rows in test_sentences.items():
        if sentence in split_sentences:
            leaks.append((sentence, test_rows, split_sentences[sentence]))
    return leaks


def fmt_rows(rows: list[int]) -> str:
    return ", ".join(str(r) for r in rows)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check for sentence-level leakage between a test CSV "
        "and one or more split CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--column",
        default="sentence",
        metavar="COL",
        help="Name of the sentence column to compare (default: 'sentence').",
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Normalise sentences to lowercase before comparing.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Optional path to write a CSV leak report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file progress messages.",
    )
    args = parser.parse_args()

    split_paths = [
        "data/tatoeba/phonetic_tatoeba_gemini_3_train.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite_train.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split_train.csv",
        "data/stress-minimal/stress-minimal_single_split_train.csv",
        "data/wiktionary-scrape/transcribed/homographs_flattened_train.csv",
    ]
    test_paths = [
        "data/tatoeba/phonetic_tatoeba_gemini_3_test.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite_test.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split_test.csv",
        "data/stress-minimal/stress-minimal_single_split_test.csv",
        "data/wiktionary-scrape/transcribed/homographs_flattened_test.csv",
        "data/tatoeba/phonetic_tatoeba_gemini_3_validation.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite_validation.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split_validation.csv",
        "data/stress-minimal/stress-minimal_single_split_validation.csv",
        "data/wiktionary-scrape/transcribed/homographs_flattened_validation.csv",
        "data/manual_set_new.csv",
        "data/manual_set_old.csv",
    ]
    test_paths = [Path(p) for p in test_paths]
    split_paths = [Path(p) for p in split_paths]
    column = args.column
    ignore_case = args.ignore_case

    total_leaked_sentences: set[str] = set()
    all_results: list[dict] = []
    any_leakage = False

    for test_path in test_paths:
        # ── load test set ────────────────────────────────────────────────────
        if not args.quiet:
            print(f"\n{'═' * 60}")
            print(f"Test file          : {test_path}")
        test_sentences = load_sentences(test_path, column, ignore_case)
        if not args.quiet:
            print(f"  Unique sentences  : {len(test_sentences):,}")

        # ── check each split ─────────────────────────────────────────────────
        for split_path in split_paths:
            if not args.quiet:
                print(f"\n  Checking split   : {split_path}")
            split_sentences = load_sentences(split_path, column, ignore_case)
            if not args.quiet:
                print(f"  Unique sentences  : {len(split_sentences):,}")

            leaks = find_leaks(test_sentences, split_sentences)

            if not leaks:
                print(f"  ✓  No leakage found in {split_path.name}")
            else:
                any_leakage = True
                print(
                    f"  ✗  {len(leaks):,} leaked sentence(s) found in {split_path.name}"
                )
                for sentence, test_rows, split_rows in leaks:
                    total_leaked_sentences.add(sentence)
                    preview = sentence if len(sentence) <= 80 else sentence[:77] + "..."
                    print(
                        f"     Row(s) in test [{fmt_rows(test_rows)}] | "
                        f"Row(s) in split [{fmt_rows(split_rows)}] | "
                        f'"{preview}"'
                    )
                    all_results.append(
                        {
                            "test_file": str(test_path),
                            "split_file": str(split_path),
                            "sentence": sentence,
                            "test_rows": fmt_rows(test_rows),
                            "split_rows": fmt_rows(split_rows),
                        }
                    )

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    if any_leakage:
        print(
            f"SUMMARY  {len(total_leaked_sentences):,} unique leaked sentence(s) "
            f"across {len(test_paths)} test file(s) and {len(split_paths)} split file(s)."
        )
    else:
        print(
            f"SUMMARY  No leakage detected "
            f"({len(test_paths)} test file(s) × {len(split_paths)} split file(s))."
        )
    print("─" * 60)

    # ── optional CSV report ───────────────────────────────────────────────────
    if args.output and all_results:
        out_path = Path(args.output)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "test_file",
                    "split_file",
                    "sentence",
                    "test_rows",
                    "split_rows",
                ],
            )
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nLeak report written : {out_path}")
    elif args.output and not all_results:
        print(f"\n(No leaks to report; '{args.output}' was not created.)")

    sys.exit(1 if any_leakage else 0)


if __name__ == "__main__":
    main()
