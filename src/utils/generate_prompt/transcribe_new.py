import csv
import json
import string
from collections import defaultdict
from string import Template

from src.utils.homographs import homographs

PHONEME_INVENTORY = [
    "'",
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "ŋ",
    "ɕ",
    "ə",
    "ɡ",
    "ɹ",  # NOTE: Only used by one word
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",  # NOTE: Technically a "duplicate"
    "ˌ",
    # NOTE: We handle the tie bar with a special instruction
    " ‍͡ (Tie bar: Unicode: U+0361)",
]

TEMPLATE_STR = f"""<sentence>
$sentence
</sentence>

<instructions>
Provide the most accurate IPA transcription for the listed words given the
context in the provided sentence, ensuring stress placement (') is correct based
on Tagalog grammar and context.
1. Selection criteria: If the correct pronunciation is in the provided options,
   select it exactly.
2. Root & inflection handling: For words marked [ROOT], infer the IPA for the
   FULL inflected word as it appears in the sentence, based on the provided
   root. Note that stress typically shifts when certain suffixes are added in
   Tagalog. Be careful not to accidentally merge repeated sounds. In the choices
   provided, note that the stress marker (') is included at the start of the
   stressed syllable.
3. Inference: For words marked [NO CHOICES], infer the IPA transcription based
   on standard Filipino pronunciation.
4. Formatting rules:
    - Only include answers for the list of words given.
    - Place stress markers (') at the BEGINNING of the stressed syllable.
    - Exclude prosodic markers (like tone) and syllable separators (dots).
    - Provide the response as a direct, comma-separated list of IPA strings in
    the exact order the words appear in the sentence.
    - You may use only the UTF-8 characters provided in the phoneme inventory.
</instructions>

<phoneme_inventory>
{PHONEME_INVENTORY}
</phoneme_inventory>

<word_list>
$words
</word_list>

<pron_choices>
$pronunciations
</pron_choices>
"""

TEMPLATE = Template(TEMPLATE_STR)

TRANSLATOR = str.maketrans("", "", string.punctuation)


def _format_word_data(data, definitions_map=None):
    word = data.get("word")
    if definitions_map and word in definitions_map:
        matches = definitions_map[word]
        lines = []
        for match in matches:
            prons = match.get("prons", [])
            defs = match.get("definitions", [])
            ipa = prons[0] if prons else ""
            def_str = json.dumps(defs, ensure_ascii=False)
            lines.append(f"- {ipa}: {def_str}")
        return f"{word}:\n" + "\n".join(lines)

    choices = data.get("choices")
    if choices:
        pron_str = ", ".join(choices)
    else:
        root = data.get("root", {})
        root_word = root.get("word", "")
        root_prons = ", ".join(root.get("choices", []))
        pron_str = f'[ROOT "{root_word}"] {root_prons}'
    return f"{word}: {pron_str}"


def generate_prompt(sentence, pronunciations, words, definitions_map=None):
    seen = set()
    unique_pronunciations = []
    for data in pronunciations:
        w = data.get("word")
        if w not in seen:
            seen.add(w)
            unique_pronunciations.append(data)

    formatted_words = ", ".join([f'"{word}"' for word in words])
    formatted_pronunciations = "\n".join(
        _format_word_data(data, definitions_map) for data in unique_pronunciations
    )
    formatted_sentence = f'"{sentence}"'

    return TEMPLATE.safe_substitute(
        words=formatted_words,
        sentence=formatted_sentence,
        pronunciations=formatted_pronunciations,
    )


# --- CSV-driven batched prompt generation ----------------------------------


def _rows_from_csv(csv_file):
    if hasattr(csv_file, "read"):
        return list(csv.DictReader(csv_file))
    with open(csv_file, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_context_hints(row, all_rows_by_sentence):
    """
    Return (pronunciations, definitions_map) for non-target words
    co-occurring with `row` in its sentence.
    """
    target_word = row["word"]
    pronunciations = []
    definitions_map = defaultdict(list)
    for other in all_rows_by_sentence[row["sentence"]]:
        if other is row:
            continue
        w = other["word"]
        pron = other.get("pronunciation", "")
        definition = other.get("definition", "")
        pronunciations.append({"word": w, "choices": [pron] if pron else []})
        definitions_map[w].append(
            {
                "prons": [pron] if pron else [],
                "definitions": [definition] if definition else [],
            }
        )
    return pronunciations, definitions_map or None


def generate_prompts_from_csv(csv_file):
    """
    Build one prompt per unique target word. All rows sharing the same
    'word' are batched into a single prompt. Each sentence block lists
    only the words that are actually blanked (homographs other than the
    target), along with how many answers are needed for that sentence.

    Returns a list of:
        {
            "word":       <target word>,
            "rows":       [<row dict>, ...],      # same order as answer slices
            "templates":  [<output_template>, ...],
            "slot_counts": [<int>, ...],
            "prompt":     <str>,
        }

    total answers = sum(slot_counts); slice by slot_counts to recover
    per-row answer lists.
    """
    rows = _rows_from_csv(csv_file)

    by_sentence = defaultdict(list)
    for row in rows:
        by_sentence[row["sentence"]].append(row)

    by_word = defaultdict(list)
    for row in rows:
        by_word[row["word"]].append(row)

    results = []
    for word, group in by_word.items():
        sentence_blocks = []
        templates = []
        slot_counts = []
        total_slots = 0

        for i, row in enumerate(group, start=1):
            _, _, output_template = homographs(
                row["sentence"], row["word"], row["pronunciation"]
            )
            templates.append(output_template)

            token_words = row["sentence"].lower().translate(TRANSLATOR).split()
            blank_words = {
                token_words[j] for j, t in enumerate(output_template) if t == "_"
            }
            n_slots = len(blank_words)  # unique blanked words for hint display
            n_slots = output_template.count("_")
            slot_counts.append(n_slots)
            total_slots += n_slots

            pronunciations, definitions_map = _build_context_hints(row, by_sentence)
            hints = [d for d in pronunciations if d["word"] in blank_words]

            block_lines = [
                f'[{i}] "{row["sentence"]}" ({n_slots} answer{"s" if n_slots != 1 else ""})'
            ]
            if hints:
                formatted = "\n".join(
                    "    " + _format_word_data(d, definitions_map) for d in hints
                )
                block_lines.append(f"    choices:\n{formatted}")
            sentence_blocks.append("\n".join(block_lines))

        word_list_section = "\n\n".join(sentence_blocks)
        all_sentences = "\n".join(
            f'[{i}] "{r["sentence"]}"' for i, r in enumerate(group, start=1)
        )

        prompt = TEMPLATE.safe_substitute(
            words=f'"{word}" (target — already resolved, do not transcribe)',
            sentence=all_sentences,
            pronunciations=word_list_section,
        )

        results.append(
            {
                "word": word,
                "rows": group,
                "templates": templates,
                "slot_counts": slot_counts,
                "prompt": prompt,
            }
        )

    return results
