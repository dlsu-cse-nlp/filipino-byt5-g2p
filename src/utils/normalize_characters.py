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
        "‍": "",  # Zero-width joiner that Gemini hallucinates sometimes
        # For Gemini 2.5-Flash-Lite output
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
    }

    text = text.replace(".", "")  # Remove syllable markers

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()
