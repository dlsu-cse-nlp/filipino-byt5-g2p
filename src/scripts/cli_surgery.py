#!/usr/bin/env python3
"""
fix_phonemes.py — Interactive Filipino G2P correction tool.

Usage:
    python fix_phonemes.py <input.csv>

CSV must have columns: index, word, pronunciation, sentence, phoneme
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import questionary
from questionary import Style

from src.utils.phoneme_utils import (
    apply_replacement,
    build_index,
    normalize_word,
    tokenize_phonemes,
    tokenize_sentence,
    write_csv,
)
from src.utils.questionary_style import STYLE


def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"index", "word", "pronunciation", "sentence", "phoneme"}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            sys.exit(f"[ERROR] CSV is missing required columns: {missing}")
        return list(reader)


PAGE_SIZE = 5


def preview_occurrences(
    rows: list[dict],
    occurrence_map: dict[tuple[str, str], list[tuple[int, int]]],
    norm_word: str,
    ipas_to_preview: list[str],
) -> None:
    """
    Interactively page through every sentence that contains norm_word mapped
    to any of the given IPA strings, grouped by IPA variant.
    """
    entries: list[tuple[str, int, int]] = []
    for ipa in ipas_to_preview:
        for row_idx, tok_idx in occurrence_map.get((norm_word, ipa), []):
            entries.append((ipa, row_idx, tok_idx))

    if not entries:
        print("  ⚠  No occurrences found.\n")
        return

    entries.sort(key=lambda x: (x[0], x[1]))
    total = len(entries)
    page = 0

    while True:
        start = page * PAGE_SIZE
        end = min(start + PAGE_SIZE, total)
        print(
            f"\n  📋  Occurrences for '{norm_word}'  "
            f"[{start + 1}–{end} of {total}]\n"
            f"  {'─' * 60}"
        )

        for ipa, row_idx, tok_idx in entries[start:end]:
            row = rows[row_idx]
            sentence = row["sentence"]
            phonemes = tokenize_phonemes(row["phoneme"])

            highlighted = []
            for i, ph in enumerate(phonemes):
                highlighted.append(f"[{ph}]" if i == tok_idx else ph)
            phoneme_str = " ".join(highlighted)

            print(f"  IPA : {ipa}")
            print(f"  sent: {sentence}")
            print(f"  phon: {phoneme_str}")
            print(f"  {'─' * 60}")

        nav_choices = []
        if end < total:
            nav_choices.append(questionary.Choice("▶  Next page", value="next"))
        if page > 0:
            nav_choices.append(questionary.Choice("◀  Previous page", value="prev"))
        nav_choices.append(questionary.Choice("✖  Close preview", value="close"))

        action = questionary.select("", choices=nav_choices, style=STYLE).ask()

        if action == "next":
            page += 1
        elif action == "prev":
            page -= 1
        else:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Interactively fix Filipino G2P phoneme mappings."
    )
    parser.add_argument("csv_file", help="Path to input CSV file")
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
        PREVIEW = "__PREVIEW__"

        while True:
            choices = [
                questionary.Choice(
                    title=f"{ipa}  ({counts[ipa]} occurrence{'s' if counts[ipa] != 1 else ''})",
                    value=ipa,
                )
                for ipa in pronunciations
            ] + [
                questionary.Choice(
                    title="🔍  Preview occurrences  (select variants above first, or none for all)",
                    value=PREVIEW,
                ),
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
                break

            if PREVIEW in selected_ipas:
                # Preview the explicitly selected IPA variants, or all if none picked
                to_preview = [s for s in selected_ipas if s not in (PREVIEW, NUCLEAR)]
                if not to_preview:
                    to_preview = list(pronunciations)
                preview_occurrences(pending_rows, occurrence_map, norm, to_preview)
                continue  # loop back to the same checkbox screen

            break  # proceed to replacement

        if not selected_ipas:
            continue

        is_nuclear = NUCLEAR in selected_ipas
        targets = (
            pronunciations
            if is_nuclear
            else [s for s in selected_ipas if s not in (NUCLEAR, PREVIEW)]
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
            out_path = write_csv(pending_rows, args.csv_file)
            print(f"\n  💾  Saved → {out_path}\n")

        if action == ACTION_QUIT:
            print("Goodbye! 👋\n")
            break


if __name__ == "__main__":
    main()
