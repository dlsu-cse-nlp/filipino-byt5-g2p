"""
This script is used to write Tatoeba corpus data into a .txt file.
"""

import os
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/tatoeba")
OUTPUT_FILENAME = "tatoeba.txt"

print("Loading Tatoeba dataset...")
dataset = load_dataset("tatoeba", "en-tl", lang1="en", lang2="tl")
sentences = [item["tl"] for item in dataset["train"]["translation"]]

sentences = [
    item["tl"] for item in dataset["train"]["translation"] if item["tl"].strip()
]

print(f"Number of sentences: {len(sentences)}")
