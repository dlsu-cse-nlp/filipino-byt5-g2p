#!/usr/bin/env python3
"""
fix_sentence_pron_mismatches.py

Browse and fix pronunciation mismatches in a filled-homographs JSONL file by
examining every word/phoneme token pair across the whole sentence, not just the
focal word stored in "word"/"pronunciation".

For each entry the script:
  1. Splits original_sentence into grapheme words and filled_sentence into IPA
     tokens (both on whitespace).
  2. Zips the two sequences together to get (word, pron) pairs.
  3. For each pair where word is a homograph, checks whether
     normalize_characters(pron) is a valid pronunciation.
  4. Collects all violations and groups them by (word, stored_pron) for review.

Fixing replaces the offending IPA token at its exact position in filled_sentence.

Usage:
    python fix_sentence_pron_mismatches.py <jsonl_path> <wikipron_csv_path>
"""

import argparse
import json
import sys
from collections import defaultdict
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


def replace_token_at(tokens: list[str], pos: int, new_tok: str) -> list[str]:
    """Return a new token list with tokens[pos] replaced by new_tok."""
    out = list(tokens)
    out[pos] = new_tok
    return out


def pair_sentence(entry: dict) -> list[tuple[str, str, int]]:
    """
    Zip grapheme words from original_sentence with IPA tokens from
    filled_sentence.  Returns [(word, pron_token, token_position), ...].

    If the two sequences differ in length (shouldn't happen in a well-formed
    file, but just in case) we zip up to the shorter one.
    """
    words = entry.get("original_sentence", "").split()
    tokens = entry.get("filled_sentence", "").split()
    return [(w, t, i) for i, (w, t) in enumerate(zip(words, tokens))]


# ── violation model ───────────────────────────────────────────────────────────
# A Violation is a (entry_idx, token_pos, word, stored_pron) namedtuple-like
# We keep it as a plain dict for simplicity.


def find_violations(entries: list[dict], homographs: dict) -> list[dict]:
    """
    Return one violation record per (entry, token_position) that fails the
    homograph pronunciation check.

    Each record:
        {
            "entry_idx":   int,   # index into entries list
            "token_pos":   int,   # position in filled_sentence token list
            "word":        str,
            "stored_pron": str,   # raw token from filled_sentence
        }
    """
    violations = []
    for entry_idx, entry in enumerate(entries):
        pairs = pair_sentence(entry)
        for word, pron_tok, tok_pos in pairs:
            valid = homographs.get(word)
            if valid is None:
                continue  # not a homograph
            if normalize_characters(pron_tok) not in valid:
                violations.append(
                    {
                        "entry_idx": entry_idx,
                        "token_pos": tok_pos,
                        "word": word,
                        "stored_pron": pron_tok,
                    }
                )
    return violations


def group_violations(violations: list[dict], homographs: dict):
    """
    Group violation records by (word, stored_pron).

    Returns list of:
        (word, stored_pron, valid_prons, [violation_records])
    preserving first-seen order.
    """
    seen: dict[tuple, list] = {}
    for v in violations:
        key = (v["word"], v["stored_pron"])
        seen.setdefault(key, []).append(v)

    groups = []
    for (word, stored_pron), recs in seen.items():
        valid_prons = homographs.get(word, [])
        groups.append((word, stored_pron, valid_prons, recs))
    return groups


def closest_by_edit_distance(stored: str, valid_prons: list[str]) -> str:
    return min(valid_prons, key=lambda p: editdistance.eval(stored, p))


# ── apply fix ────────────────────────────────────────────────────────────────


def apply_fix(entries: list[dict], recs: list[dict], new_pron: str) -> None:
    """
    For each violation record in recs, replace the token at token_pos in
    the entry's filled_sentence with new_pron.
    """
    for v in recs:
        e = entries[v["entry_idx"]]
        tokens = e["filled_sentence"].split()
        tokens = replace_token_at(tokens, v["token_pos"], new_pron)
        e["filled_sentence"] = " ".join(tokens)


def auto_fix_all_by_edit_distance(entries: list[dict], groups) -> int:
    updated = 0
    for word, stored_pron, valid_prons, recs in groups:
        new_pron = closest_by_edit_distance(stored_pron, valid_prons)
        apply_fix(entries, recs, new_pron)
        updated += len(recs)
        print(f"  {word}:  {stored_pron}  →  {new_pron}  ({len(recs)} token(s))")
    return updated


# ── TUI ──────────────────────────────────────────────────────────────────────


def browse_and_fix(entries: list[dict], homographs: dict) -> list[dict]:
    while True:
        violations = find_violations(entries, homographs)

        if not violations:
            print("\n✓ No more pronunciation violations found.\n")
            break

        groups = group_violations(violations, homographs)

        AUTOFIX_LABEL = (
            f"⚡ Auto-fix ALL {len(groups)} group(s) by smallest edit distance"
        )
        DONE_LABEL = "── Done (save & quit) ──"

        label_to_group = {}
        choice_labels = []
        for word, stored_pron, valid_prons, recs in groups:
            # Count distinct entries (one entry can contribute multiple tokens)
            n_entries = len({r["entry_idx"] for r in recs})
            label = (
                f"{word}  |  "
                f"stored: {stored_pron}  |  "
                f"valid: {', '.join(valid_prons)}  "
                f"({len(recs)} token(s) across {n_entries} sentence(s))"
            )
            label_to_group[label] = (word, stored_pron, valid_prons, recs)
            choice_labels.append(label)
        choice_labels.append(AUTOFIX_LABEL)
        choice_labels.append(DONE_LABEL)

        selected = questionary.select(
            f"Violations remaining: {len(violations)} token(s) in {len(groups)} group(s)  –  Pick a group to fix:",
            choices=choice_labels,
            style=STYLE,
        ).ask()

        if selected is None or selected == DONE_LABEL:
            break

        if selected == AUTOFIX_LABEL:
            print()
            print("  Preview (stored → closest valid):")
            for word, stored_pron, valid_prons, recs in groups:
                best = closest_by_edit_distance(stored_pron, valid_prons)
                dist = editdistance.eval(stored_pron, best)
                n_entries = len({r["entry_idx"] for r in recs})
                print(
                    f"    {word}:  {stored_pron}  →  {best}  "
                    f"(edit dist {dist}, {len(recs)} token(s) / {n_entries} sentence(s))"
                )
            print()
            confirm = questionary.confirm(
                "Apply all of the above? (Ties take the first valid option.)",
                default=False,
                style=STYLE,
            ).ask()
            if confirm:
                n = auto_fix_all_by_edit_distance(entries, groups)
                print(f"\n  ✓  Auto-fixed {n} token(s) across {len(groups)} groups.\n")
            continue

        sel_word, sel_stored_pron, valid_prons, recs = label_to_group[selected]

        # Show context: up to 3 sample sentences
        n_entries = len({r["entry_idx"] for r in recs})
        print()
        print(f"  Word            : {sel_word}")
        print(f"  Stored pron     : {sel_stored_pron}")
        print(f"  Valid options   : {', '.join(valid_prons)}")
        print(f"  Tokens affected : {len(recs)} across {n_entries} sentence(s)")
        print()
        seen_entries: set[int] = set()
        shown = 0
        for r in recs:
            if r["entry_idx"] in seen_entries:
                continue
            seen_entries.add(r["entry_idx"])
            e = entries[r["entry_idx"]]
            print(
                f"  [{e.get('index', r['entry_idx'])}] {e.get('original_sentence', '')}"
            )
            print(f"       filled : {e['filled_sentence']}")
            # Highlight the offending token
            tokens = e["filled_sentence"].split()
            tokens[r["token_pos"]] = f">>>{tokens[r['token_pos']]}<<<"
            print(f"       marked : {' '.join(tokens)}")
            shown += 1
            if shown >= 3:
                break
        if n_entries > 3:
            print(f"  … and {n_entries - 3} more sentence(s).")
        print()

        SKIP_LABEL = "← Back (skip this group)"
        pron_choices = list(valid_prons) + [SKIP_LABEL]

        new_pron = questionary.select(
            f"Choose the correct pronunciation for all {len(recs)} token(s):",
            choices=pron_choices,
            style=STYLE,
        ).ask()

        if new_pron is None or new_pron == SKIP_LABEL:
            continue

        apply_fix(entries, recs, new_pron)
        print(
            f"\n  ✓  {sel_stored_pron}  →  {new_pron}  ({len(recs)} token(s) updated)\n"
        )

    return entries


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Fix sentence-level pronunciation mismatches in a filled-homographs JSONL."
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

    try:
        from src.datasets.wikipron_tl_df import wikipron_tl_df
    except ImportError as exc:
        sys.exit(
            f"Could not import wikipron_tl_df: {exc}\n"
            "Make sure you run this script from the project root."
        )

    FILE_PATH = str(wikipron_path)
    HOMOGRAPHS, NON_HOMOGRAPHS = wikipron_tl_df(FILE_PATH)

    entries = load_jsonl(jsonl_path)
    print(f"\nLoaded {len(entries)} entries from {jsonl_path}")

    initial = find_violations(entries, HOMOGRAPHS)
    print(f"Found {len(initial)} token-level violation(s) across the sentences.\n")

    if not initial:
        print("Nothing to fix – exiting.")
        return

    entries = browse_and_fix(entries, HOMOGRAPHS)

    out_path = jsonl_path.with_stem(jsonl_path.stem + "_corrected")
    save_jsonl(entries, out_path)
    print(f"Saved corrected file to: {out_path}\n")


if __name__ == "__main__":
    main()
