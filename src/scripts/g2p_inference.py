"""
This script performs a single inference of a given sentence on a trained model
checkpoint.
"""

import argparse

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.sliding_window import predict_sliding_window

DEFAULT_SENTENCE = "Puno ng tubig ang baso."
DEFAULT_CHECKPOINT_PATH = (
    "models/checkpoints/2026-03-16_22-45_baseline_combined/checkpoint-3080"
    # "models/checkpoints/2026-03-16_14-27_baseline_combined/checkpoint-2695"
    # "models/checkpoints/2026-03-16_13-52_baseline_tatoeba/checkpoint-455"
)

parser = argparse.ArgumentParser()
parser.add_argument("sentence", nargs="?", default=DEFAULT_SENTENCE)
parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path).to(device)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=5,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


output_phonemes = predict_sliding_window(args.sentence, model, tokenizer, 11)
print("\nOriginal sentence:")
print(args.sentence)
print("\nNo sliding window:")
print(predict(args.sentence))
print("\nSliding window:")
print(output_phonemes)
