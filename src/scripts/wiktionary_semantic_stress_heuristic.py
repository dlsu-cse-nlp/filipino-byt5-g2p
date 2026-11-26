"""
Uses stress classification as a heuristic to filter for words with
pronunciations that actually significantly change meaning
"""

import json
import re

from src.datasets.wikipron_tl_df import wikipron_tl_df
from src.utils.classify_stress import classify_stress
from src.utils.normalize_characters import normalize_characters

# TODO: Do an audit of this and other similar scripts

FILE_PATH = "data/wikipron/wikipron_tl.tsv"
OUTPUT_PATH = "data/wiktionary-scrape/wikipron_filtered.jsonl"


def fold_vowels(text):
    """Collapses 'e' into 'i', and 'o' into 'u'"""
    return text.replace("e", "i").replace("o", "u")


def strip_non_final_glottals(text):
    """Removes any glottal stop except terminal ones"""
    return re.sub(r"[ʔɁ](?!$)", "", text)


def write_dict_to_jsonl(data_dict, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for word, prons in data_dict.items():
            line = json.dumps({"word": word, "prons": prons}, ensure_ascii=False)
            f.write(line + "\n")


if __name__ == "__main__":
    HOMOGRAPHS, _ = wikipron_tl_df(FILE_PATH)

    filtered_homographs = {}

    for word, prons in HOMOGRAPHS.items():
        seen_classes = set()
        cleaned_prons = []

        for pron in prons:
            # Base normalization
            normalized = normalize_characters(pron)

            # WARNING: Decided against using these

            # Strip non-meaningful glottals
            # normalized = strip_non_final_glottals(normalized)

            # Heuristic 1: Collapse i/e and u/o differences
            # normalized = fold_vowels(normalized)

            stress_class = classify_stress(normalized)

            if stress_class != "None":
                seen_classes.add(stress_class)
                cleaned_prons.append(normalized)

        unique_prons = list(set(cleaned_prons))

        # Must have >1 variant remaining, that span at least 2 classes
        if len(unique_prons) > 1 and len(seen_classes) >= 2:
            filtered_homographs[word] = unique_prons

    print(f"Retained {len(filtered_homographs)} true heteronyms.")

    write_dict_to_jsonl(filtered_homographs, OUTPUT_PATH)
    print(f"Results saved to {OUTPUT_PATH}")
