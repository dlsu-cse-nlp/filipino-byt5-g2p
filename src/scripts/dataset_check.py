# TODO: We don't use all these imports but whatever
import argparse
from datetime import datetime

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from datasets import concatenate_datasets
from src.utils.dataset_from_csv import dataset_from_csv, dataset_from_csv_list
from src.utils.preview_callback import PreviewCallback

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = "data/combined.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

dataset = [
    "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split.csv",
    "data/stress-minimal/stress-minimal_single_split.csv",
]

split_dataset = dataset_from_csv_list(dataset, tokenizer)

train_set = split_dataset["train"]
manual_set = dataset_from_csv("data/manual_set_old.csv", tokenizer)
manual_set = concatenate_datasets(list(manual_set.values()))

train_sentences = set(train_set["sentence"])
manual_sentences = set(manual_set["sentence"])

leakage = train_sentences.intersection(manual_sentences)

# 1. Flatten sentences into sets of unique words (lowercased for better matching)
train_words = {word for sent in train_sentences for word in sent.lower().split()}
manual_words = {word for sent in manual_sentences for word in sent.lower().split()}

# 2. Find the intersection (seen words)
seen_words = manual_words.intersection(train_words)
unseen_words = manual_words - train_words

print(f"Unique words in training: {len(train_words)}")
print(f"Unique words in manual set: {len(manual_words)}")
print(f"Seen words: {len(seen_words)} ({len(seen_words)/len(manual_words):.2%})")
print(f"Unseen words: {len(unseen_words)} ({len(unseen_words)/len(manual_words):.2%})")

if len(unseen_words) > 0:
    print(f"Sample OOV words: {list(unseen_words)[:10]}")

if len(leakage) > 0:
    print(f"Found {len(leakage)} leaking examples")
    for i, sentence in enumerate(list(leakage)[:5]):
        print(f"{i+1}. {sentence}")

    train_set_cleaned = manual_set.filter(
        lambda example: example["sentence"] not in leakage
    )
    cols_to_remove = [
        col
        for col in train_set_cleaned.column_names
        if col not in ["sentence", "phoneme"]
    ]
    train_set_final = train_set_cleaned.remove_columns(cols_to_remove)

    train_set_final.to_csv("data/manual_set_noleak.csv", index=False)
else:
    print("No leakage detected")
