"""Matches Wiktionary IPA entries with definitions"""

import json
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

TARGET_LANGUAGE = "Tagalog"
INPUT_FILE = "data/wiktionary-scrape/wikipron_filtered.jsonl"
OUTPUT_FILE = "finalfinalfinal.jsonl"

# TODO: Do an audit of this and other similar scripts

from src.utils.normalize_characters import normalize_characters


def get_heading_info(node):
    if not node:
        return None, None
    if node.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        return int(node.name[1]), node.get_text().strip()
    if node.name == "div" and any(
        cls.startswith("mw-heading") for cls in node.get("class", [])
    ):
        inner_h = node.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if inner_h:
            return int(inner_h.name[1]), inner_h.get_text().strip()
    return None, None


def fetch_wiktionary_html(url, headers, max_retries=20, base_delay=1, max_delay=60):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 404:
                return None

            if response.status_code == 200 or (
                400 <= response.status_code < 500 and response.status_code != 429
            ):
                return response

            print(
                f"[Warning] Received status code {response.status_code} for {url}. Backing off..."
            )

        except (requests.ConnectionError, requests.Timeout) as e:
            print(
                f"[Warning] Connection failed ({type(e).__name__}) on attempt {attempt + 1}/{max_retries} for {url}"
            )
            if attempt == max_retries - 1:
                print(f"[Error] Max retries reached. Exiting block.", file=sys.stderr)
                return None
        except requests.RequestException as e:
            print(f"[Error] Critical request anomaly: {e}", file=sys.stderr)
            return None

        delay = min(base_delay * (2**attempt), max_delay)
        print(f"Sleeping for {delay} seconds before retrying...")
        time.sleep(delay)

    return None


def scrape_wiktionary_structure(word, target_lang):
    url = f"https://en.wiktionary.org/wiki/{word}"
    headers = {
        "User-Agent": "WiktionaryMatcherBot/1.0 (Contact: your_email@example.com)"
    }

    response = fetch_wiktionary_html(url, headers)
    if not response or response.status_code != 200:
        return None

    soup = BeautifulSoup(response.content, "html.parser")

    lang_heading = soup.find(["h2", "h3"], id=target_lang)
    if not lang_heading:
        lang_span = soup.find("span", class_="mw-headline", id=target_lang)
        if lang_span:
            lang_heading = lang_span.find_parent(["h2", "h3"])

    if not lang_heading:
        return None

    if lang_heading.parent and any(
        cls.startswith("mw-heading") for cls in lang_heading.parent.get("class", [])
    ):
        start_node = lang_heading.parent
    else:
        start_node = lang_heading

    etymology_blocks = []
    current_block = {"pronunciations": [], "definitions": []}
    current_pos = "Unknown"

    current_node = start_node.find_next_sibling()

    while current_node:
        level, heading_text = get_heading_info(current_node)

        if level is not None:
            if level <= 2:
                break

            clean_text = heading_text.replace("[edit]", "").strip()

            if "Etymology" in clean_text:
                if current_block["pronunciations"] or current_block["definitions"]:
                    etymology_blocks.append(current_block)
                current_block = {"pronunciations": [], "definitions": []}
            else:
                current_pos = clean_text

        else:
            ipa_spans_all = current_node.find_all("span", class_="IPA")
            if ipa_spans_all:
                found_structured = False
                for span in ipa_spans_all:
                    clean_ipa = span.get_text().strip("/[]")
                    if not clean_ipa:
                        continue

                    parent = span
                    pos_tags = []
                    glosses = []
                    while parent:
                        if parent.name in ("li", "dt"):
                            parent_text = parent.get_text().lower()
                            pos_tags = re.findall(
                                r"\b(noun|verb|adjective|adverb|pronoun|preposition|conjunction|interjection|particle)\b",
                                parent_text,
                            )
                            gloss_match = re.search(r"“([^”]+)”", parent.get_text())
                            if gloss_match:
                                gloss_raw = gloss_match.group(1).strip()
                                glosses = [
                                    g.strip() for g in gloss_raw.split(";") if g.strip()
                                ]
                            found_structured = True
                            break
                        parent = parent.parent
                        if parent == current_node or parent is None:
                            break

                    current_block["pronunciations"].append(
                        {"ipa": clean_ipa, "pos": pos_tags[:], "glosses": glosses}
                    )

                if not found_structured:
                    current_block["pronunciations"] = [
                        {"ipa": sp.get_text().strip("/[]"), "pos": [], "glosses": []}
                        for sp in ipa_spans_all
                        if sp.get_text().strip("/[]")
                    ]

            if current_node.name == "ol":
                for li in current_node.find_all("li", recursive=False):
                    for sub_list in li.find_all(["ul", "dl"]):
                        sub_list.decompose()

                    def_text = li.get_text().strip()
                    if def_text:
                        current_block["definitions"].append(
                            f"({current_pos}) {def_text}"
                        )

        current_node = current_node.find_next_sibling()

    if current_block["pronunciations"] or current_block["definitions"]:
        etymology_blocks.append(current_block)

    return etymology_blocks


def clean_text_for_gloss(text):
    """Remove parentheses and extra spaces to help fuzzy match glosses."""
    text = re.sub(r"\([^)]*\)", "", text)
    return " ".join(text.split()).strip().lower()


def match_jsonl_entry(entry, scraped_blocks):
    matched_results = []
    input_prons = entry.get("prons", [])

    for block in scraped_blocks:
        pair_list = []

        for sp in block["pronunciations"]:
            normalized_ipa = normalize_characters(sp["ipa"])
            matched_input_prons = [ip for ip in input_prons if ip == normalized_ipa]
            if not matched_input_prons:
                continue

            allowed_pos = set(sp["pos"])
            unrestricted = len(allowed_pos) == 0
            glosses = sp.get("glosses", [])
            has_glosses = bool(glosses)

            matched_defs = []
            for def_text in block.get("definitions", []):
                def_pos_match = re.match(r"^\((.*?)\)", def_text)
                if def_pos_match:
                    def_pos_text = def_pos_match.group(1).lower()
                    def_pos_tokens = [
                        token.strip() for token in def_pos_text.split(",")
                    ]
                    pos_ok = unrestricted or any(
                        token in allowed_pos for token in def_pos_tokens
                    )
                    if not pos_ok:
                        continue
                else:
                    pos_ok = True

                if has_glosses and def_pos_match:
                    sense_text = def_text[def_pos_match.end() :].strip().lower()
                    cleaned_sense = clean_text_for_gloss(sense_text)
                    if cleaned_sense:
                        if not any(
                            clean_text_for_gloss(g) in cleaned_sense for g in glosses
                        ):
                            continue

                matched_defs.append(def_text)

            if matched_defs:
                pair_list.append((list(matched_input_prons), matched_defs))

        defs_to_prons = {}
        for prons_list, defs in pair_list:
            defs_tuple = tuple(defs)
            if defs_tuple not in defs_to_prons:
                defs_to_prons[defs_tuple] = []
            # Extend the list, keeping the order of appearance
            for p in prons_list:
                if p not in defs_to_prons[defs_tuple]:
                    defs_to_prons[defs_tuple].append(p)

        for defs_tuple, ordered_prons in defs_to_prons.items():
            matched_results.append(
                {"prons": ordered_prons, "definitions": list(defs_tuple)}
            )

    # Remove completely duplicate matches
    unique_results = []
    seen = set()
    for res in matched_results:
        key = (tuple(res["prons"]), tuple(res["definitions"]))
        if key not in seen:
            seen.add(key)
            unique_results.append(res)

    return {"word": entry["word"], "matches": unique_results}


def process_jsonl(input_path, output_path):
    print(f"Processing lines from {input_path}...")

    with (
        open(input_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):

        for line_idx, line in enumerate(infile):
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                word = entry.get("word")
                print(f"Scraping and matching: '{word}'")

                scraped_data = scrape_wiktionary_structure(word, TARGET_LANGUAGE)

                if scraped_data:
                    output_data = match_jsonl_entry(entry, scraped_data)
                else:
                    output_data = {
                        "word": word,
                        "matches": [],
                        "status": "wiktionary_section_not_found",
                    }

                outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error on line {line_idx}: {e}", file=sys.stderr)

    print(f"Finished processing! Output saved to: {output_path}")


if __name__ == "__main__":
    process_jsonl(INPUT_FILE, OUTPUT_FILE)
