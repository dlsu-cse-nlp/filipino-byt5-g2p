"""
This script evaluates a model checkpoint on a given test set by calculating PER
and PFER.
"""

# TODO: This needs work.

import argparse

import panphon
import panphon.distance
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.engines.roberta_fused_byt5 import RoBERTaFusedByT5
from src.utils.dataset_from_csv import dataset_from_csv_list

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"

DEFAULT_CHECKPOINT = (
    # "models/checkpoints/2026-03-16_22-45_baseline_combined/checkpoint-3080"
    "models/checkpoints/2026-03-16_13-52_baseline_tatoeba/checkpoint-455"
    # "models/checkpoints/2026-03-16_14-27_baseline_combined/checkpoint-2695"
)
DEFAULT_DATASET_PATH = "data/combined.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
parser.add_argument(
    "--dataset", default="tatoeba", choices=["tatoeba", "newsph-nli", "combined"]
)
args = parser.parse_args()

# Set up dataset CSV paths
if args.dataset == "tatoeba":
    dataset = ["data/tatoeba/phonetic_tatoeba_gemini_3.csv"]
elif args.dataset == "newsph-nli":
    dataset = ["data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv"]
elif args.dataset == "combined":
    dataset = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    ]

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# TODO: Add to CLI
model = RoBERTaFusedByT5.from_pretrained(args.checkpoint_path)
model.to(device)
model.eval()

# Use Panphon's built in tools, less of a headache this way
ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

# Get the test split
split_dataset = dataset_from_csv_list(dataset, tokenizer)
test_set = split_dataset["test"]

# Running sums
total_per_dist = 0
total_pfer_dist = 0
total_cer_dist = 0
total_phonemes = 0
total_chars = 0

print(f"Evaluating {len(test_set)} samples from {dataset}")

with torch.no_grad():
    for item in tqdm(test_set):
        target_text = item["phoneme"]
        target_segs = ft.ipa_segs(target_text)

        if not target_segs:
            continue

        inputs = tokenizer(item["sentence"], return_tensors="pt").to(device)

        # """
        outputs = model.generate(
            **inputs,
            max_length=512,
            repetition_penalty=1.3,
            sentence=[item["sentence"] * 8],
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_segs = ft.ipa_segs(pred_text)

        """
        encoder_outputs = model.encoder(**inputs)
        hidden = encoder_outputs.last_hidden_state
        fused = model._fuse_via_cross_attention(
            [item["sentence"]], hidden, hidden.device
        )
        encoder_outputs.last_hidden_state = fused
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs[
                "attention_mask"
            ],  # only pass attention_mask, not input_ids
            encoder_outputs=encoder_outputs,
            max_length=512,
            repetition_penalty=1.3,
        )

        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_segs = ft.ipa_segs(pred_text)
        # """

        print(
            f"Input: {item["sentence"]}\nTarget: {target_text}\nPred:   {pred_text}\n-----"
        )

        # Calculate PER
        per_dist = dst.levenshtein_distance(pred_segs, target_segs)

        # Calculate PFER (we need a try-catch in case Panphon alignment dies)
        try:
            pfer_dist = dst.feature_edit_distance(pred_text, target_text)
        except ValueError:
            pfer_dist = len(target_segs)

        # Calculate CER (Levenshtein on raw characters)
        cer_dist = dst.levenshtein_distance(pred_text, target_text)

        # Add to the summation
        total_per_dist += per_dist
        total_pfer_dist += pfer_dist
        total_cer_dist += cer_dist

        total_phonemes += len(target_segs)
        total_chars += len(target_text)

        print(
            f"Running PER: {total_per_dist / total_phonemes if total_phonemes > 0 else 0}"
        )

final_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
final_pfer = total_pfer_dist / total_phonemes if total_phonemes > 0 else 0
final_cer = total_cer_dist / total_chars if total_chars > 0 else 0

print(f"Character Error Rate (CER):         {final_cer:.4f}")
print(f"Phoneme Error Rate (PER):           {final_per:.4f}")
print(f"Phonetic Feature Error Rate (PFER): {final_pfer:.4f}")
print(f"Total reference phonemes:           {total_phonemes}")
