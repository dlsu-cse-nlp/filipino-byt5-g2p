# TODO: This is an old script.

"""
This script parses the .jsonl output of the scripts/query_gemini_tatoeba.py to
produce a cleaner CSV file for use in training
"""

import argparse
import json
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from src.utils.homographs import fill_template, homographs
from src.utils.normalize_characters import normalize_characters
from src.utils.phoneme_inventory import PHONEME_INVENTORY

# TODO: Use argparse

DEFAULT_DATASET_PATH = "data/newsph-nli/results_gemini_2.5_lite.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
parser.add_argument("--output", type=str, default="output_from_jsonl.csv")
args = parser.parse_args()


def validate_characters(answers):
    """Checks if the string contains characters not in the phoneme inventory"""

    for word in answers:
        for char in word:
            if char not in PHONEME_INVENTORY:
                try:
                    char_name = unicodedata.name(char)
                except ValueError:
                    char_name = "Unknown"

                # Print exactly what caused the error
                print(f"Failed to process word: '{word}'")
                print(f"Error at '{char}' (U+{ord(char):04X}: {char_name})")

                return False

    return True


if __name__ == "__main__":
    # Load Tatoeba dataset
    print("Loading dataset...")

    with open("data/stress-minimal/single.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # dataset = load_dataset("tatoeba", "en-tl", lang1="en", lang2="tl")
    # sentences = [item["tl"] for item in dataset["train"]["translation"]]

    results_list = []
    error_count = 0
    invalid_responses = 0

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Processing JSONL", unit="line"):

            # TODO: This part needs cleaning up

            data = json.loads(line)
            idx = data.get("index")

            if idx is None or (idx - 1) >= len(sentences) or (idx - 1) < 0:
                continue

            sentence = sentences[idx - 1]

            if not sentence or not str(sentence).strip():
                continue

            try:
                _, _, output_template = homographs(sentence)
            except Exception:
                continue

            output_string = None
            status = None

            if data.get("success") is True:
                answers = data["content"]["answers"]
                answers = [normalize_characters(a) for a in answers]

                if validate_characters(answers):
                    try:
                        output_string = fill_template(output_template, answers)
                        status = "success"
                    except StopIteration:
                        error_count += 1
                        continue
                    except Exception:
                        error_count += 1
                        continue
            elif data.get("error") == "ERR_EMPTY_PROMPT":
                # Case when no ambiguous words were found
                output_string = " ".join(output_template)
                status = "empty_prompt_fallback"
            else:
                invalid_responses += 1
                continue

            if output_string and str(output_string).strip():
                results_list.append(
                    {
                        "index": idx,
                        "sentence": sentence,
                        "phoneme": output_string,
                        "status": status,
                    }
                )

    df = pd.DataFrame(results_list)
    df = df.drop(columns=["status"])
    df["phoneme"] = df["phoneme"].apply(normalize_characters)
    df.to_csv(args.output, index=False, encoding="utf-8")

    print("=" * 40)
    print("Summary")
    print("-" * 40)
    print(f"Total lines in JSONL: {total_lines}")
    print(f"Valid entries: {len(df)}")
    print(f"Template mismatches : {error_count}")
    print(f"Invalid API responses: {invalid_responses}")
    print("=" * 40)
