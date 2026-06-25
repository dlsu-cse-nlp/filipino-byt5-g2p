import csv
import os
import re
from collections import defaultdict
from copy import deepcopy


def normalize_word(w):
    """Lowercase and strip punctuation for matching purposes."""
    return re.sub(r"[^\w']", "", w, flags=re.UNICODE).lower()


def tokenize_sentence(sentence):
    tokens = re.split(r"\s+", sentence.strip())
    return [normalize_word(t) for t in tokens if normalize_word(t)]


def tokenize_phonemes(phoneme_str):
    return phoneme_str.strip().split()


def build_index(rows):
    """Returns a set of IPA strings and their occurences"""
    word_pron_map: dict[str, set[str]] = defaultdict(set)
    occurrence_map: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)

    for row_idx, row in enumerate(rows):
        words = tokenize_sentence(row["sentence"])
        ipas = tokenize_phonemes(row["phoneme"])

        if len(words) != len(ipas):
            continue

        for tok_idx, (w, ipa) in enumerate(zip(words, ipas)):
            word_pron_map[w].add(ipa)
            occurrence_map[(w, ipa)].append((row_idx, tok_idx))

    return word_pron_map, occurrence_map


def apply_replacement(rows, occurrence_map, norm_word, old_ipa, new_ipa):
    """Replace old_ipa with new_ipa for every (norm_word, old_ipa) occurrence."""
    rows = deepcopy(rows)

    occurrences = occurrence_map.get((norm_word, old_ipa), [])
    if not occurrences:
        return rows

    by_row: dict[int, list[int]] = defaultdict(list)
    for row_idx, tok_idx in occurrences:
        by_row[row_idx].append(tok_idx)

    for row_idx, tok_indices in by_row.items():
        ipas = tokenize_phonemes(rows[row_idx]["phoneme"])
        tok_set = set(tok_indices)
        rows[row_idx]["phoneme"] = " ".join(
            new_ipa if i in tok_set else ipa for i, ipa in enumerate(ipas)
        )

    return rows


def write_csv(rows, input_path):
    """Atomically overwrite input_path in-place via a sibling .tmp file."""
    input_path = os.path.abspath(input_path)
    tmp_path = input_path + ".tmp"

    fieldnames = list(rows[0].keys()) if rows else []
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(tmp_path, input_path)
    return input_path
