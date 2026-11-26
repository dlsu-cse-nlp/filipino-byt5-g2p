import json
import re
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# TODO: Do an audit of this and other similar scripts


def strip_pos(definition):
    """Remove leading '(POS)' tag"""
    return re.sub(r"^\([^)]+\)\s*", "", definition).strip()


def normalize_core(text):
    """Normalize text for comparison: lowercase, remove trailing 's' (plural), strip."""
    text = text.lower().strip()
    if text.endswith("s"):
        text = text[:-1]
    return text


def is_free_variation(definitions):
    """Return True if all definitions essentially mean the same thing."""
    if len(definitions) <= 1:
        return True
    cores = [normalize_core(strip_pos(d)) for d in definitions]
    return len(set(cores)) == 1


def get_definitions_for_word(word):
    """
    Returns (definitions, error_type)
        - success: (list_of_defs, None)
        - missing (page not found or no Tagalog defs): ([], 'missing')
        - transient (network/5xx): ([], 'transient')
    """
    url = f"https://en.wiktionary.org/api/rest_v1/page/definition/{word}"
    headers = {"User-Agent": "TagalogHeteronymProject/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException:
        # Network error, retry
        return [], "transient"

    if response.status_code == 404:
        # Page doesn't exist, no retry
        return [], "missing"

    if response.status_code != 200:
        # 5xx or some other error, retry
        return [], "transient"

    data = response.json()
    if "tl" not in data or not data["tl"]:
        # Page exists but no Tagalog definitions
        return [], "missing"

    definitions = []
    for pos_group in data["tl"]:
        pos = pos_group.get("partOfSpeech", "Unknown")
        for d in pos_group.get("definitions", []):
            clean_def = BeautifulSoup(d["definition"], "html.parser").get_text()
            definitions.append(f"({pos}) {clean_def}")

    if not definitions:
        return [], "missing"

    return definitions, None


def get_definitions_with_retry(word, max_retries=20, initial_delay=0.5):
    """
    Fetch definitions with retry on transient errors.
    Returns list of definitions, or empty list if permanently missing.
    """
    for attempt in range(max_retries):
        defs, err = get_definitions_for_word(word)
        if err is None:
            return defs  # success
        if err == "missing":
            tqdm.write(f"  ℹ️  No definitions for '{word}' (skipping)")
            return []  # abort without retry
        # err == 'transient'
        if attempt < max_retries - 1:
            wait = initial_delay * (2**attempt)
            tqdm.write(
                f"Transient error for '{word}', retry {attempt+1}/{max_retries} in {wait:.1f}s..."
            )
            time.sleep(wait)
    tqdm.write(
        f"Failed to fetch definitions for '{word}' after {max_retries} attempts."
    )
    return []


def main():
    input_file = "data/wiktionary-scrape/wikipron_filtered.jsonl"
    output_file = "data/wiktionary-scrape/initial_heteronyms.jsonl"
    skipped_file = "data/wiktionary-scrape/initial_free_variation_skipped.jsonl"

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} entries...")
    heteronyms = []
    free_variation = []

    for line in tqdm(lines, desc="Fetching definitions"):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        word = entry.get("word")
        prons = entry.get("prons", [])
        if not word:
            continue

        # Fetch with retry, but will skip immediately if missing
        definitions = get_definitions_with_retry(word)
        if not definitions:
            continue  # no definitions – skip this word

        entry["definitions"] = definitions

        # Print status
        tqdm.write(f"\nWord: {word}")
        tqdm.write(f"Prons: {prons}")
        tqdm.write(f"Definitions: {definitions}")

        if is_free_variation(definitions):
            tqdm.write("→ FILTERED (free variation)")
            free_variation.append(entry)
        else:
            tqdm.write("→ KEPT (heteronym)")
            heteronyms.append(entry)

        time.sleep(0.2)

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in heteronyms:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(skipped_file, "w", encoding="utf-8") as f:
        for entry in free_variation:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nKept {len(heteronyms)} heteronyms → {output_file}")
    print(f"Skipped {len(free_variation)} free‑variation entries → {skipped_file}")


if __name__ == "__main__":
    main()
