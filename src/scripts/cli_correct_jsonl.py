#!/usr/bin/env python3
"""
fix_pron_mismatches.py

Browse and fix pronunciation mismatches in a filled-homographs JSONL file.
Violations are entries where the stored "pronunciation" is not among the
valid pronunciations for that word according to wikipron_tl_df.

Usage:
    python fix_pron_mismatches.py <jsonl_path> <wikipron_csv_path>
"""

import argparse
import json
import sys
from pathlib import Path

import editdistance
import questionary
from questionary import Style

from src.utils.normalize_characters import normalize_characters

# ── colour scheme ────────────────────────────────────────────────────────────
STYLE = Style(
    [
        ("qmark", "fg:#ff9d00 bold"),
        ("question", "bold"),
        ("answer", "fg:#00d7af bold"),
        ("pointer", "fg:#ff9d00 bold"),
        ("highlighted", "fg:#ff9d00 bold"),
        ("selected", "fg:#00d7af"),
        ("separator", "fg:#6c6c6c"),
        ("instruction", "fg:#6c6c6c"),
        ("text", ""),
        ("disabled", "fg:#858585 italic"),
    ]
)


# ── helpers ───────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def save_jsonl(entries: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def find_violations(entries: list[dict], homographs: dict) -> list[int]:
    """Return indices (into entries) of rows that are pronunciation violations.

    The stored pronunciation is normalised before matching so that superficial
    character-encoding differences (e.g. different Unicode representations of
    the same IPA glyph) don't produce false positives.
    """
    bad = []
    for i, e in enumerate(entries):
        word = e.get("word", "")
        pron = e.get("pronunciation", "")
        valid = homographs.get(word)  # None if word not a homograph
        if valid is None:
            continue  # not a homograph – skip
        if normalize_characters(pron) not in valid:
            bad.append(i)
    return bad


def replace_in_sentence(sentence: str, old_pron: str, new_pron: str) -> str:
    """Replace every occurrence of old_pron with new_pron in sentence."""
    return sentence.replace(old_pron, new_pron)


def closest_by_edit_distance(stored: str, valid_prons: list[str]) -> str:
    """Return the valid pronunciation with the smallest edit distance to stored.
    Ties are broken by the order they appear in valid_prons."""
    return min(valid_prons, key=lambda p: editdistance.eval(stored, p))


# ── TUI ──────────────────────────────────────────────────────────────────────


def auto_fix_all_by_edit_distance(entries: list[dict], homographs: dict, groups) -> int:
    """Apply closest-edit-distance fix to every group without prompting.
    Returns the number of entries updated."""
    updated = 0
    for word, stored_pron, valid_prons, affected_indices in groups:
        new_pron = closest_by_edit_distance(stored_pron, valid_prons)
        for idx in affected_indices:
            e = entries[idx]
            e["filled_sentence"] = replace_in_sentence(
                e["filled_sentence"], stored_pron, new_pron
            )
            e["pronunciation"] = new_pron
        updated += len(affected_indices)
        print(
            f"  {word}:  {stored_pron}  →  {new_pron}  ({len(affected_indices)} entries)"
        )
    return updated


def group_violations(
    violation_indices: list[int], entries: list[dict], homographs: dict
):
    """Return an ordered list of (word, stored_pron, valid_prons, [entry_indices])
    with one entry per unique (word, stored_pron) pair, preserving first-seen order."""
    seen = {}  # (word, stored_pron) -> list of entry indices
    for idx in violation_indices:
        e = entries[idx]
        key = (e["word"], e["pronunciation"])
        seen.setdefault(key, []).append(idx)

    groups = []
    for (word, stored_pron), idxs in seen.items():
        valid_prons = homographs.get(word, [])
        groups.append((word, stored_pron, valid_prons, idxs))
    return groups


def browse_and_fix(entries: list[dict], homographs: dict) -> list[dict]:
    while True:
        violation_indices = find_violations(entries, homographs)

        if not violation_indices:
            print("\n✓ No more pronunciation violations found.\n")
            break

        groups = group_violations(violation_indices, entries, homographs)

        # Build label→group mapping; questionary always returns the label string
        AUTOFIX_LABEL = (
            f"⚡ Auto-fix ALL {len(groups)} group(s) by smallest edit distance"
        )
        DONE_LABEL = "── Done (save & quit) ──"

        label_to_group = {}
        choice_labels = []
        for word, stored_pron, valid_prons, idxs in groups:
            label = (
                f"{word}  |  "
                f"stored: {stored_pron}  |  "
                f"valid: {', '.join(valid_prons)}  "
                f"({len(idxs)} entries)"
            )
            label_to_group[label] = (word, stored_pron, valid_prons, idxs)
            choice_labels.append(label)
        choice_labels.append(AUTOFIX_LABEL)
        choice_labels.append(DONE_LABEL)

        selected = questionary.select(
            f"Violations remaining: {len(violation_indices)} across {len(groups)} group(s)  –  Pick a group to fix:",
            choices=choice_labels,
            style=STYLE,
        ).ask()

        if selected is None or selected == DONE_LABEL:
            break

        if selected == AUTOFIX_LABEL:
            # Show a preview of what will happen and ask for confirmation
            print()
            print("  Preview (stored → closest valid):")
            for word, stored_pron, valid_prons, idxs in groups:
                best = closest_by_edit_distance(stored_pron, valid_prons)
                dist = editdistance.eval(stored_pron, best)
                print(
                    f"    {word}:  {stored_pron}  →  {best}  (edit dist {dist}, {len(idxs)} entries)"
                )
            print()
            confirm = questionary.confirm(
                "Apply all of the above? (Ambiguous ties will take the first valid option.)",
                default=False,
                style=STYLE,
            ).ask()
            if confirm:
                n = auto_fix_all_by_edit_distance(entries, homographs, groups)
                print(f"\n  ✓  Auto-fixed {n} entries across {len(groups)} groups.\n")
            continue

        # Retrieve the matching group via the label
        sel_word, sel_stored_pron, valid_prons, affected_indices = label_to_group[
            selected
        ]

        # Show a sample of affected entries for context
        print()
        print(f"  Word            : {sel_word}")
        print(f"  Stored pron     : {sel_stored_pron}")
        print(f"  Valid options   : {', '.join(valid_prons)}")
        print(f"  Entries affected: {len(affected_indices)}")
        print()
        sample = affected_indices[:3]
        for idx in sample:
            e = entries[idx]
            print(f"  [{e['index']}] {e.get('original_sentence', '')}")
            print(f"       filled: {e['filled_sentence']}")
        if len(affected_indices) > 3:
            print(f"  … and {len(affected_indices) - 3} more.")
        print()

        SKIP_LABEL = "← Back (skip this group)"
        pron_choices = list(valid_prons) + [SKIP_LABEL]

        new_pron = questionary.select(
            f"Choose the correct pronunciation for all {len(affected_indices)} entries:",
            choices=pron_choices,
            style=STYLE,
        ).ask()

        if new_pron is None or new_pron == SKIP_LABEL:
            continue

        # Apply to every entry in the group
        for idx in affected_indices:
            e = entries[idx]
            e["filled_sentence"] = replace_in_sentence(
                e["filled_sentence"], sel_stored_pron, new_pron
            )
            e["pronunciation"] = new_pron

        print(
            f"\n  ✓  {sel_stored_pron}  →  {new_pron}  ({len(affected_indices)} entries updated)\n"
        )

    return entries


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fix pronunciation mismatches in a filled-homographs JSONL."
    )
    parser.add_argument("jsonl_path", help="Path to the input .jsonl file")
    parser.add_argument(
        "wikipron_path", help="Path to the WikiPron TSV/CSV used by wikipron_tl_df"
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    wikipron_path = Path(args.wikipron_path)

    if not jsonl_path.exists():
        sys.exit(f"Error: JSONL file not found: {jsonl_path}")
    if not wikipron_path.exists():
        sys.exit(f"Error: WikiPron file not found: {wikipron_path}")

    # ── import project module ─────────────────────────────────────────────────
    try:
        from src.datasets.wikipron_tl_df import wikipron_tl_df
    except ImportError as exc:
        sys.exit(
            f"Could not import wikipron_tl_df: {exc}\n"
            "Make sure you run this script from the project root."
        )

    FILE_PATH = str(wikipron_path)
    HOMOGRAPHS, NON_HOMOGRAPHS = wikipron_tl_df(FILE_PATH)

    # ── load data ─────────────────────────────────────────────────────────────
    entries = load_jsonl(jsonl_path)
    print(f"\nLoaded {len(entries)} entries from {jsonl_path}")

    initial_violations = find_violations(entries, HOMOGRAPHS)
    print(f"Found {len(initial_violations)} pronunciation violation(s) to review.\n")

    if not initial_violations:
        print("Nothing to fix – exiting.")
        return

    # ── interactive loop ──────────────────────────────────────────────────────
    entries = browse_and_fix(entries, HOMOGRAPHS)

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = jsonl_path.with_stem(jsonl_path.stem + "_corrected")
    save_jsonl(entries, out_path)
    print(f"Saved corrected file to: {out_path}\n")


if __name__ == "__main__":
    main()
