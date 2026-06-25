import pandas as pd

from src.scripts.read_jsonl import normalize_characters

PHONEME_INVENTORY = [
    " ",
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
    "z",
    "ŋ",
    # "ɕ",
    # "ə",
    "ɡ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",
    "ˌ",
    "\u0361",
]


def process_csv(file_path):
    # 1. Read everything into memory immediately
    df_raw = pd.read_csv(file_path)

    # 2. Determine columns based on path
    if file_path == "data/wiktionary-scrape/transcribed/homographs_flattened.csv":
        cols = [
            "index",
            "word",
            "pronunciation",
            "sentence",
            "phoneme",
            "answers",
            "definition",
        ]
        df = df_raw[cols].fillna("").copy()
        df["pronunciation"] = (
            df["pronunciation"].astype(str).apply(normalize_characters)
        )
    else:
        cols = ["index", "sentence", "phoneme"]
        df = df_raw[cols].fillna("").copy()

    df["phoneme"] = df["phoneme"].astype(str).apply(normalize_characters)

    # 3. Drop rows whose phoneme field contains any out-of-inventory character
    inventory_set = set(PHONEME_INVENTORY)

    def is_valid_phoneme(s):
        return all(c in inventory_set for c in s)

    before = len(df)
    mask = df["phoneme"].apply(is_valid_phoneme)
    df = df[mask].copy()
    after = len(df)
    print(
        f"{file_path}: dropped {before - after} / {before} rows with out-of-inventory characters"
    )

    # 4. Write back to the exact same path safely
    df.to_csv(file_path, index=False)


csv_paths = [
    "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split.csv",
    "data/stress-minimal/stress-minimal_single_split.csv",
    "data/wiktionary-scrape/transcribed/homographs_flattened.csv",
]

for x in csv_paths:
    process_csv(x)  # Removed the redundant output_path=x
