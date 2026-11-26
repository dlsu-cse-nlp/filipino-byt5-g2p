import re


def normalize_characters(text):
    """Normalizes the characters in a string to the phoneme inventory above"""

    if not text:
        return ""

    # TODO: Can look into the following mappings:
    # "ɛ": "e", "ɪ": "i", "ʊ": "u", "ɔ": "o", "ɑ": "a",
    replacements = {
        "g": "ɡ",
        "r": "ɾ",
        "ɹ": "ɾ",
        ",": "",
        "ˌ": "",
        "Ɂ": "ʔ",
        ".": "",
        "ˈ": "'",
        "‍": "",
        "ɛ": "e",
        "ɪ": "i",
        "ʊ": "u",
        "ɔ": "o",
        "ɑ": "a",
        "æ": "a",
        ":": "",
        "꞉": "",
        "ː": "",
        "\u200b": "",  # Zero-width space
        "ɐ": "a",
        "á": "a",
        "ʌ": "a",
        "ɭ": "l",
        "ʤ": "dʒ",
        "ɕ": "ʃ",
        # ---------
        "y": "j",
        "_": "",
        "-": "",
        "η": "ŋ",
        # Diacritics that can be dropped (combining marks, no base change needed)
        "\u032a": "",  # COMBINING BRIDGE BELOW — dental diacritic, drop it
        "\u032f": "",  # COMBINING INVERTED BREVE BELOW — non-syllabic diacritic, drop it
        # Vowels with diacritics → strip to base vowel
        "ú": "u",  # u with acute → u (acute likely redundant stress mark)
        "ā": "a",  # a with macron → a
        "ã": "a",  # a with tilde → a
        "ı": "i",  # dotless i → i
        "ó": "o",  # o with acute → o
        "ɒ": "o",  # turned alpha (open back rounded) → closest is o
        # "ɚ": "ə",  # schwa with hook (r-colored schwa) → ə
        # "ɜ": "ə",  # reversed open e → ə (closest in inventory)
        "ø": "o",  # o with stroke (front rounded) → o
        "ʉ": "u",  # u bar (central rounded) → u
        # Consonants → nearest inventory equivalent
        "ɲ": "n",  # palatal nasal → n
        "ɴ": "n",  # uvular nasal → n
        "ɱ": "m",  # labiodental nasal → m
        # "θ": "s",  # voiceless dental fricative → s (closest sibilant-ish)
        # "ð": "d",  # voiced dental fricative → d
        # "β": "v",  # voiced bilabial fricative → v
        # "ɸ": "f",  # voiceless bilabial fricative → f
        # "ɰ": "w",  # voiced velar approximant → w
        # "ɯ": "u",  # close back unrounded vowel → u
        # "c": "k",  # voiceless palatal stop → k
        # "ɟ": "j",  # voiced palatal stop → j
        # "ʑ": "z",  # voiced alveolo-palatal fricative → z (closer than ʒ)
        # "ʝ": "j",  # voiced palatal fricative → j
        # "x": "h",  # voiceless velar fricative → h (closest available)
        # "q": "k",  # voiceless uvular stop → k
        # Uppercase → lowercase equivalents in inventory
        # Apostrophe-like
        "`": "'",  # grave accent used as glottal/stress mark → '
        "ŉ": "n",  # n preceded by apostrophe → n (apostrophe dropped; marginal)
    }

    text = text.replace(".", "")  # Remove syllable markers

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()
