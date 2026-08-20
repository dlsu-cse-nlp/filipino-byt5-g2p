from string import Template

from src.utils.phoneme_inventory import PHONEME_INVENTORY

# TODO: This is the current working prompt template...
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
    - You may use only the UTF-8 characters provided in the pphoneme inventory.
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

import json
from string import Template


def _format_word_data(data, definitions_map=None):
    """
    Format one word's pronunciation choices with definition annotations.
    If definitions_map is provided and the word is found, list each
    pronunciation together with its definitions.
    """
    word = data.get("word")
    if definitions_map and word in definitions_map:
        # Build the enriched list
        matches = definitions_map[word]
        lines = []
        for match in matches:
            prons = match.get("prons", [])
            defs = match.get("definitions", [])
            # Use the first pronunciation (or join if multiple, but usually one per match)
            ipa = prons[0] if prons else ""
            # Format the definitions as a JSON list for easy parsing by the LLM
            def_str = json.dumps(defs, ensure_ascii=False)
            lines.append(f"- {ipa}: {def_str}")
        return f"{word}:\n" + "\n".join(lines)

    # Fallback: old style (choices or root)
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
    # Remove duplicates (keep first occurrence of each word)
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
