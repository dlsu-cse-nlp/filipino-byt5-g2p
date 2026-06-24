#!/usr/bin/env python3
"""
fix_phonemes.py — Interactive Filipino G2P correction tool.

Usage:
    python fix_phonemes.py <input.csv> [--output-dir <dir>]

CSV must have columns: index, word, pronunciation, sentence, phoneme
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import questionary
from questionary import Style

# ── Styling ──────────────────────────────────────────────────────────────────

STYLE = Style(
    [
        ("qmark", "fg:#f5a623 bold"),
        ("question", "bold"),
        ("answer", "fg:#5bc4f5 bold"),
        ("pointer", "fg:#f5a623 bold"),
        ("highlighted", "fg:#f5a623 bold"),
        ("selected", "fg:#5bc4f5"),
        ("separator", "fg:#6c6c6c"),
        ("instruction", "fg:#6c6c6c"),
        ("text", ""),
        ("disabled", "fg:#858585 italic"),
    ]
)

# ── Text helpers ──────────────────────────────────────────────────────────────


def normalize_word(w: str) -> str:
    """Lowercase and strip punctuation for matching purposes."""
    return re.sub(r"[^\w']", "", w, flags=re.UNICODE).lower()


def tokenize_sentence(sentence: str) -> list[str]:
    """Split a sentence into word tokens, stripping punctuation."""
    tokens = re.split(r"\s+", sentence.strip())
    return [normalize_word(t) for t in tokens if normalize_word(t)]


def tokenize_phonemes(phoneme_str: str) -> list[str]:
    """Split IPA phoneme string on whitespace."""
    return phoneme_str.strip().split()


# ── CSV loading ───────────────────────────────────────────────────────────────


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"index", "word", "pronunciation", "sentence", "phoneme"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            sys.exit(f"[ERROR] CSV is missing required columns: {missing}")
        return list(reader)


# ── Index building ─────────────────────────────────────────────────────────────
#
# word_pron_map  : word → set of IPA strings seen for that word
# occurrence_map : (word, ipa) → list of (row_index, token_index) in the CSV rows
#
# We need occurrence_map so replacements are surgically precise — only
# nasaan→na'saʔan pairs are touched, not some other word that happens to
# share the same IPA string.


def build_index(rows: list[dict]):
    word_pron_map: dict[str, set[str]] = defaultdict(set)
    # (norm_word, ipa) -> [(row_idx, token_idx), ...]
    occurrence_map: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)

    for row_idx, row in enumerate(rows):
        words = tokenize_sentence(row["sentence"])
        ipas = tokenize_phonemes(row["phoneme"])

        if len(words) != len(ipas):
            # Silently skip misaligned rows (could warn if desired)
            continue

        for tok_idx, (w, ipa) in enumerate(zip(words, ipas)):
            word_pron_map[w].add(ipa)
            occurrence_map[(w, ipa)].append((row_idx, tok_idx))

    return word_pron_map, occurrence_map


# ── Replacement ───────────────────────────────────────────────────────────────


def apply_replacement(
    rows: list[dict],
    occurrence_map: dict[tuple[str, str], list[tuple[int, int]]],
    norm_word: str,
    old_ipa: str,
    new_ipa: str,
) -> list[dict]:
    """
    Replace old_ipa with new_ipa for every (norm_word, old_ipa) occurrence.
    Works by re-building the phoneme string of affected rows token-by-token.
    Other (word, ipa) pairs are untouched.
    """
    rows = deepcopy(rows)

    occurrences = occurrence_map.get((norm_word, old_ipa), [])
    if not occurrences:
        return rows

    # Group by row so we can fix each phoneme string once
    from collections import defaultdict as dd

    by_row: dict[int, list[int]] = dd(list)
    for row_idx, tok_idx in occurrences:
        by_row[row_idx].append(tok_idx)

    for row_idx, tok_indices in by_row.items():
        row = rows[row_idx]
        ipas = tokenize_phonemes(row["phoneme"])
        tok_set = set(tok_indices)
        ipas = [new_ipa if i in tok_set else ipa for i, ipa in enumerate(ipas)]
        rows[row_idx]["phoneme"] = " ".join(ipas)

    return rows


# ── CSV writing ───────────────────────────────────────────────────────────────


def write_csv(rows: list[dict], output_dir: str, input_path: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_path).stem
    out_path = os.path.join(output_dir, f"{stem}_fixed.csv")

    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# ── Main interaction loop ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Interactively fix Filipino G2P phoneme mappings."
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output CSV (default: ./output)",
    )
    args = parser.parse_args()

    print(f"\n📂  Loading {args.csv_file} …")
    rows = load_csv(args.csv_file)
    print(f"    {len(rows)} rows loaded.")

    word_pron_map, occurrence_map = build_index(rows)
    print(f"    {len(word_pron_map)} unique words indexed.\n")

    ACTION_QUIT = "⏻  Quit"
    ACTION_SAVE = "💾  Save & continue"
    ACTION_NOSAVE = "↩  Continue without saving"

    pending_rows = deepcopy(rows)  # accumulates all edits in this session

    while True:
        # ── Word input ────────────────────────────────────────────────────────
        raw_word = questionary.text(
            "Enter a word to inspect (or leave blank to quit):",
            style=STYLE,
        ).ask()

        if raw_word is None or raw_word.strip() == "":
            print("\nGoodbye! 👋\n")
            break

        norm = normalize_word(raw_word)

        if norm not in word_pron_map:
            print(f"  ⚠  '{norm}' not found in the dataset.\n")
            continue

        pronunciations = sorted(word_pron_map[norm])
        counts = {ipa: len(occurrence_map[(norm, ipa)]) for ipa in pronunciations}

        total_count = sum(counts.values())
        NUCLEAR = "__NUCLEAR__"

        choices = [
            questionary.Choice(
                title=f"{ipa}  ({counts[ipa]} occurrence{'s' if counts[ipa] != 1 else ''})",
                value=ipa,
            )
            for ipa in pronunciations
        ] + [
            questionary.Choice(
                title=f"☢  Replace ALL pronunciations  ({total_count} total occurrences)",
                value=NUCLEAR,
            ),
        ]

        selected_ipas = questionary.checkbox(
            f"Pronunciations for '{norm}'  (space to select, enter to confirm):",
            choices=choices,
            style=STYLE,
        ).ask()

        # None = Ctrl-C, empty list = enter with nothing checked
        if not selected_ipas:
            continue

        is_nuclear = NUCLEAR in selected_ipas
        targets = (
            pronunciations if is_nuclear else [s for s in selected_ipas if s != NUCLEAR]
        )
        target_count = sum(counts[ipa] for ipa in targets)

        if is_nuclear:
            print(
                f"\n  ☢  Nuclear mode — will replace ALL {len(pronunciations)} variant(s).\n"
            )
            prompt = (
                f"Replace ALL {len(pronunciations)} pronunciation(s) of '{norm}' →  "
                f"(enter new IPA, or leave blank to cancel):"
            )
        elif len(targets) == 1:
            prompt = (
                f"Replace  {targets[0]}  →  (enter new IPA, or leave blank to cancel):"
            )
        else:
            listed = ", ".join(targets)
            prompt = (
                f"Replace {len(targets)} selected pronunciation(s) [{listed}] →  "
                f"(enter new IPA, or leave blank to cancel):"
            )

        new_ipa = questionary.text(prompt, style=STYLE).ask()

        if new_ipa is None or new_ipa.strip() == "":
            print("  Cancelled.\n")
            continue

        new_ipa = new_ipa.strip()

        if is_nuclear or len(targets) > 1:
            label = "☢" if is_nuclear else "✅"
            confirmed = questionary.confirm(
                f"{label}  Replace {target_count} occurrence(s) across "
                f"{len(targets)} variant(s) of '{norm}' → {new_ipa}?",
                default=False,
                style=STYLE,
            ).ask()
            if not confirmed:
                print("  Cancelled.\n")
                continue

        # ── Apply all targeted replacements ───────────────────────────────────
        total_replaced = 0
        for old_ipa in targets:
            pending_rows = apply_replacement(
                pending_rows, occurrence_map, norm, old_ipa, new_ipa
            )
            old_occurrences = occurrence_map.pop((norm, old_ipa), [])
            occurrence_map[(norm, new_ipa)].extend(old_occurrences)
            total_replaced += len(old_occurrences)
            word_pron_map[norm].discard(old_ipa)

        word_pron_map[norm].add(new_ipa)

        label = "☢" if is_nuclear else "✅"
        variant_note = (
            f" across {len(targets)} variant(s)"
            if len(targets) > 1
            else f": '{norm}' / {targets[0]}"
        )
        print(
            f"\n  {label}  Replaced {total_replaced} occurrence{'s' if total_replaced != 1 else ''}"
            f"{variant_note} → {new_ipa}\n"
        )

        # ── Save prompt ───────────────────────────────────────────────────────
        action = questionary.select(
            "What would you like to do next?",
            choices=[ACTION_SAVE, ACTION_NOSAVE, ACTION_QUIT],
            style=STYLE,
        ).ask()

        if action == ACTION_SAVE or action == ACTION_QUIT:
            out_path = write_csv(pending_rows, args.output_dir, args.csv_file)
            print(f"\n  💾  Saved → {out_path}\n")

        if action == ACTION_QUIT:
            print("Goodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
