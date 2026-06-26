#!/usr/bin/env python3
"""
vowel_mismatches.py — Browse and fix word-IPA pairs where vowel counts disagree,
or where a word ends in 'ng' but the IPA does not end in 'ŋ'.

Usage:
    python vowel_mismatches.py <input.csv>

CSV must have columns: index, word, pronunciation, sentence, phoneme

Vowels counted on both sides: a e i o u (case-insensitive).
IPA diacritics (stress marks, length marks, etc.) are ignored when counting.
Pairs are sorted by descending frequency (most common mismatches first).
"""

import argparse
import csv
import re
import sys
from copy import deepcopy

import questionary
from questionary import Style

from src.utils.phoneme_utils import (
    apply_replacement,
    build_index,
    tokenize_phonemes,
    tokenize_sentence,
    write_csv,
)
from src.utils.questionary_style import STYLE

VOWELS = set("aeiou")
PAGE_SIZE = 45


def count_vowels(s: str) -> int:
    return sum(1 for ch in s.lower() if ch in VOWELS)


def has_ng_tail_mismatch(word: str, ipa: str) -> bool:
    """True when the grapheme ends in 'ng' but the IPA does not end in 'ŋ'."""
    return word.lower().endswith("ng") and not ipa.endswith("ŋ")


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"index", "sentence", "phoneme"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            sys.exit(f"[ERROR] CSV is missing required columns: {missing}")
        return list(reader)


def find_mismatches(rows: list[dict]) -> list[dict]:
    """
    Returns a list of dicts, one per unique (word, ipa) pair with either:
      • a vowel-count mismatch, or
      • a word ending in 'ng' whose IPA does not end in 'ŋ'
    Sorted by descending frequency.
    """
    pair_info: dict[tuple[str, str], dict] = {}

    for row_idx, row in enumerate(rows):
        words = tokenize_sentence(row["sentence"])
        ipas = tokenize_phonemes(row["phoneme"])

        if len(words) != len(ipas):
            continue

        for tok_idx, (w, ipa) in enumerate(zip(words, ipas)):
            wv = count_vowels(w)
            iv = count_vowels(ipa)

            vowel_mismatch = wv != iv
            ng_mismatch = has_ng_tail_mismatch(w, ipa)

            if not vowel_mismatch and not ng_mismatch:
                continue

            key = (w, ipa)
            if key not in pair_info:
                pair_info[key] = {
                    "word": w,
                    "ipa": ipa,
                    "word_vowels": wv,
                    "ipa_vowels": iv,
                    "vowel_mismatch": vowel_mismatch,
                    "ng_mismatch": ng_mismatch,
                    "count": 0,
                    "examples": [],
                }
            pair_info[key]["count"] += 1
            pair_info[key]["examples"].append((row_idx, tok_idx))

    return sorted(pair_info.values(), key=lambda x: x["count"], reverse=True)


def mismatch_label(m: dict) -> str:
    tags = []
    if m["vowel_mismatch"]:
        delta = m["ipa_vowels"] - m["word_vowels"]
        arrow = f"+{delta}" if delta > 0 else str(delta)
        tags.append(f"{m['word_vowels']}v→{m['ipa_vowels']}v {arrow}")
    if m["ng_mismatch"]:
        tags.append("ng≠ŋ")
    tag_str = "  [" + ", ".join(tags) + f"]  ×{m['count']}"
    return f"{m['word']:<20} →  {m['ipa']:<24}{tag_str}"


def browse_pair(
    m: dict,
    rows: list[dict],
    pending_rows: list[dict],
    occurrence_map: dict,
    csv_path: str,
) -> list[dict]:
    """
    Page through sentences for this pair and optionally replace the IPA.
    Returns the (possibly updated) pending_rows.
    """
    examples = m["examples"]
    total = len(examples)
    page = 0

    ACTION_REPLACE = "__REPLACE__"
    ACTION_BACK = "__BACK__"
    ACTION_NEXT = "__NEXT__"
    ACTION_PREV = "__PREV__"

    while True:
        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)

        tags = []
        if m["vowel_mismatch"]:
            tags.append(f"{m['word_vowels']}v → {m['ipa_vowels']}v")
        if m["ng_mismatch"]:
            tags.append("ng ≠ ŋ")
        tag_str = ", ".join(tags)

        print(
            f"\n  📋  '{m['word']}' → {m['ipa']}"
            f"  [{tag_str}]"
            f"  [{start + 1}–{end} of {total}]\n"
            f"  {'─' * 64}"
        )

        for row_idx, tok_idx in examples[start:end]:
            row = pending_rows[row_idx]
            sentence = row["sentence"]
            phonemes = tokenize_phonemes(row["phoneme"])
            highlighted = [
                f"[{ph}]" if i == tok_idx else ph for i, ph in enumerate(phonemes)
            ]
            print(f"  sent: {sentence}")
            print(f"  phon: {' '.join(highlighted)}")
            print(f"  {'─' * 64}")

        nav = []
        if end < total:
            nav.append(questionary.Choice("▶  Next page", value=ACTION_NEXT))
        if page > 0:
            nav.append(questionary.Choice("◀  Previous page", value=ACTION_PREV))
        nav.append(
            questionary.Choice(
                f"✏️   Replace '{m['ipa']}' for all {m['count']} occurrence(s)",
                value=ACTION_REPLACE,
            )
        )
        nav.append(questionary.Choice("← Back to list", value=ACTION_BACK))

        action = questionary.select("", choices=nav, style=STYLE).ask()

        if action == ACTION_NEXT:
            page += 1
        elif action == ACTION_PREV:
            page -= 1
        elif action == ACTION_BACK or action is None:
            break
        elif action == ACTION_REPLACE:
            pending_rows = _do_replace(m, pending_rows, occurrence_map, csv_path)
            # If the replacement was applied, the pair is resolved — go back
            break

    return pending_rows


def _do_replace(
    m: dict,
    pending_rows: list[dict],
    occurrence_map: dict,
    csv_path: str,
) -> list[dict]:
    """Prompt for a new IPA, apply it, and offer to save."""
    new_ipa = questionary.text(
        f"Replace  {m['ipa']}  →  (enter new IPA, or leave blank to cancel):",
        style=STYLE,
        default=m["ipa"],
    ).ask()

    if not new_ipa or not new_ipa.strip():
        print("  Cancelled.\n")
        return pending_rows

    new_ipa = new_ipa.strip()

    pending_rows = apply_replacement(
        pending_rows, occurrence_map, m["word"], m["ipa"], new_ipa
    )

    # Update occurrence_map in-place so later operations stay consistent
    old_occurrences = occurrence_map.pop((m["word"], m["ipa"]), [])
    occurrence_map[(m["word"], new_ipa)].extend(old_occurrences)

    print(
        f"\n  ✅  Replaced {m['count']} occurrence(s): "
        f"'{m['word']}' / {m['ipa']} → {new_ipa}\n"
    )

    # Update the mismatch entry so the list reflects the change on re-render
    m["ipa"] = new_ipa
    m["ipa_vowels"] = count_vowels(new_ipa)
    m["ng_mismatch"] = has_ng_tail_mismatch(m["word"], new_ipa)
    m["vowel_mismatch"] = m["word_vowels"] != m["ipa_vowels"]

    action = questionary.select(
        "What would you like to do next?",
        choices=[
            "💾  Save & continue",
            "↩  Continue without saving",
        ],
        style=STYLE,
    ).ask()

    if action and action.startswith("💾"):
        out_path = write_csv(pending_rows, csv_path)
        print(f"\n  💾  Saved → {out_path}\n")

    return pending_rows


def _do_delete_below_threshold(
    mismatches: list[dict],
    pending_rows: list[dict],
    csv_path: str,
) -> list[dict]:
    """Collect all row indices touched by low-frequency mismatch pairs, confirm, delete."""
    threshold_str = questionary.text(
        "Delete sentences containing pairs with frequency strictly below:",
        style=STYLE,
    ).ask()

    if not threshold_str or not threshold_str.strip():
        print("  Cancelled.\n")
        return pending_rows

    try:
        threshold = int(threshold_str.strip())
    except ValueError:
        print("  Invalid number. Cancelled.\n")
        return pending_rows

    # Collect the set of row indices to drop
    rows_to_delete: set[int] = set()
    pairs_affected = 0
    for m in mismatches:
        if m["count"] < threshold:
            pairs_affected += 1
            for row_idx, _ in m["examples"]:
                rows_to_delete.add(row_idx)

    if not rows_to_delete:
        print(f"  No pairs below frequency {threshold}. Nothing to delete.\n")
        return pending_rows

    print(
        f"\n  ⚠️   This will delete {len(rows_to_delete)} sentence(s) "
        f"spanning {pairs_affected} mismatch pair(s) with frequency < {threshold}."
    )

    confirmed = questionary.confirm("  Proceed?", default=False, style=STYLE).ask()
    if not confirmed:
        print("  Cancelled.\n")
        return pending_rows

    pending_rows = [r for i, r in enumerate(pending_rows) if i not in rows_to_delete]
    print(f"\n  🗑️   Deleted {len(rows_to_delete)} sentence(s).\n")

    action = questionary.select(
        "What would you like to do next?",
        choices=[
            "💾  Save & continue",
            "↩  Continue without saving",
        ],
        style=STYLE,
    ).ask()

    if action and action.startswith("💾"):
        out_path = write_csv(pending_rows, csv_path)
        print(f"\n  💾  Saved → {out_path}\n")

    return pending_rows


# ── Top-level mismatch browser ────────────────────────────────────────────────


def browse_mismatches(
    mismatches: list[dict],
    rows: list[dict],
    occurrence_map: dict,
    csv_path: str,
) -> None:
    total = len(mismatches)
    pending_rows = deepcopy(rows)

    if total == 0:
        print("\n  ✅  No mismatches found in the dataset.\n")
        return

    page = 0

    while True:
        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)

        print(f"\n  🔤  Mismatches  [{start + 1}–{end} of {total}]\n" f"  {'─' * 64}")

        choices = []
        for i, m in enumerate(mismatches[start:end], start=start + 1):
            choices.append(
                questionary.Choice(
                    title=f"{i:>4}.  {mismatch_label(m)}",
                    value=m,
                )
            )

        if end < total:
            choices.append(questionary.Choice("▶  Next page", value="next"))
        if page > 0:
            choices.append(questionary.Choice("◀  Previous page", value="prev"))
        choices.append(
            questionary.Choice(
                "🗑️   Delete sentences below frequency threshold", value="delete_below"
            )
        )
        choices.append(questionary.Choice("✖  Quit", value="quit"))

        action = questionary.select(
            "Select a pair to inspect / fix, or navigate:",
            choices=choices,
            style=STYLE,
        ).ask()

        if action is None or action == "quit":
            break
        elif action == "next":
            page += 1
        elif action == "prev":
            page -= 1
        elif action == "delete_below":
            pending_rows = _do_delete_below_threshold(
                mismatches, pending_rows, csv_path
            )
        else:
            pending_rows = browse_pair(
                action, rows, pending_rows, occurrence_map, csv_path
            )


# ── Entry point ───────────────────────────────────────────────────────────────
from src.utils.dataset_files import CSV_PATHS


def main():
    # parser = argparse.ArgumentParser(
    #     description="Browse and fix word-IPA pairs with mismatched vowel counts or ng/ŋ tail mismatches."
    # )
    # parser.add_argument("csv_file", help="Path to input CSV file")
    # args = parser.parse_args()

    for csv_file in CSV_PATHS:
        print("-" * 80)
        print(f"\n📂  Loading {csv_file} …")
        rows = load_csv(csv_file)
        print(f"    {len(rows)} rows loaded.")

        print("    Building index …")
        _, occurrence_map = build_index(rows)

        print("    Scanning for mismatches …")
        mismatches = find_mismatches(rows)
        total_pairs = len(mismatches)
        total_instances = sum(m["count"] for m in mismatches)
        n_vowel = sum(1 for m in mismatches if m["vowel_mismatch"])
        n_ng = sum(1 for m in mismatches if m["ng_mismatch"])
        print(
            f"    {total_pairs} unique mismatched pair(s) "
            f"across {total_instances} total occurrence(s) "
            f"({n_vowel} vowel-count, {n_ng} ng/ŋ tail).\n"
        )

        browse_mismatches(mismatches, rows, occurrence_map, csv_file)
    print("\nGoodbye! 👋\n")


if __name__ == "__main__":
    main()
