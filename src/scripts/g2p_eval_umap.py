# TODO: vibecoded lol
"""
This script evaluates a model checkpoint on a given test set by calculating PER
and PFER.
"""

import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panphon
import panphon.distance
import plotly.express as px
import torch
import umap
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from datasets import concatenate_datasets
from src.utils.classify_stress import classify_stress
from src.utils.dataset_from_csv import dataset_from_csv, dataset_from_csv_list

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = "data/combined.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--checkpoint-path", default="")
parser.add_argument(
    "--dataset",
    default="tatoeba",
    choices=["tatoeba", "newsph-nli", "combined", "manual"],
)
args = parser.parse_args()

os.makedirs("results", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

if args.dataset == "tatoeba":
    dataset = ["data/tatoeba/phonetic_tatoeba_gemini_3.csv"]
    split_dataset = dataset_from_csv_list(dataset, tokenizer)
elif args.dataset == "newsph-nli":
    dataset = ["data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv"]
    split_dataset = dataset_from_csv_list(dataset, tokenizer)
elif args.dataset == "combined":
    dataset = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    ]
    split_dataset = dataset_from_csv_list(dataset, tokenizer)
elif args.dataset == "manual":
    dataset = "data/manual_set.csv"
    split_dataset = dataset_from_csv(dataset, tokenizer)
    split_dataset["test"] = concatenate_datasets(list(split_dataset.values()))

device = "cuda" if torch.cuda.is_available() else "cpu"

model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
model.to(device)
model.eval()

ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

test_set = split_dataset["test"]

total_per_dist = 0
total_pfer_dist = 0
total_cer_dist = 0
total_phonemes = 0
total_chars = 0

output = []

word_embeddings = []  # list of 1-D numpy arrays (one per word)
word_labels = []  # stress class for each word

print(f"Evaluating {len(test_set)} samples from {dataset}")

with torch.no_grad():
    for item in tqdm(test_set):
        target_text = item["phoneme"]
        target_segs = ft.ipa_segs(target_text)

        if not target_segs:
            continue

        inputs = tokenizer(item["sentence"], return_tensors="pt").to(device)

        encoder_outputs = model.encoder(**inputs, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states[8][0]

        outputs = model.generate(
            **inputs,
            max_length=256,
            # num_beams=5,
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_segs = ft.ipa_segs(pred_text)

        print("-" * 80)
        print(f"Target:  {target_text}\nPredict: {pred_text}")

        per_dist = dst.levenshtein_distance(pred_segs, target_segs)

        try:
            pfer_dist = dst.feature_edit_distance(pred_text, target_text)
        except ValueError:
            pfer_dist = len(target_segs)

        cer_dist = dst.levenshtein_distance(pred_text, target_text)

        total_per_dist += per_dist
        total_pfer_dist += pfer_dist
        total_cer_dist += cer_dist
        total_phonemes += len(target_segs)
        total_chars += len(target_text)

        running_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
        print(f"Running PER: {running_per}")

        output.append(
            {
                "sentence": item["sentence"],
                "target": target_text,
                "predicted": pred_text,
                "per": per_dist / len(target_segs),
                "cer": cer_dist / len(target_text) if len(target_text) > 0 else 0,
                "pfer": pfer_dist / len(target_segs),
            }
        )

        # ByT5 encodes bytes; we recover word spans via character offsets.
        sentence = item["sentence"]
        ipa_words = target_text.split()
        ortho_words = sentence.split()

        for ortho_word, ipa_word in zip(ortho_words, ipa_words):
            word_enc = tokenizer(
                ortho_word,
                return_tensors="pt",
                return_offsets_mapping=False,
            )
            n_word_tokens = word_enc["input_ids"].shape[1] - 1  # drop EOS

            word_ids = word_enc["input_ids"][0, :-1].to(device)  # drop EOS
            full_ids = inputs["input_ids"][0]
            span_start = None
            for i in range(len(full_ids) - len(word_ids) + 1):
                if torch.equal(full_ids[i : i + len(word_ids)], word_ids):
                    span_start = i
                    break

            if span_start is None:
                continue  # word not found (tokenization edge case)

            span_end = span_start + len(word_ids)
            word_vec = hidden_states[span_start:span_end].mean(dim=0).cpu().numpy()

            stress_class = classify_stress(ipa_word)
            word_embeddings.append(word_vec)
            word_labels.append(stress_class)

final_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
final_pfer = total_pfer_dist / total_phonemes if total_phonemes > 0 else 0
final_cer = total_cer_dist / total_chars if total_chars > 0 else 0

print("=" * 40)
print("Pooled error statistics")
print("-" * 40)
print(f"Character Error Rate (CER):         {final_cer:.4f}")
print(f"Phoneme Error Rate (PER):           {final_per:.4f}")
print(f"Phonetic Feature Error Rate (PFER): {final_pfer:.4f}")
print(f"Total reference phonemes:           {total_phonemes}")

df = pd.DataFrame(output)

print("-" * 40)
print("Per-sample error statistics")
print("-" * 40)
print(df[["per", "cer", "pfer"]].describe())

# df.to_pickle(
#    f"results/output_{Path(args.checkpoint_path).parts[-2]}_{Path(args.checkpoint_path).parts[-1]}_{args.dataset}.pkl"
# )

print("=" * 40)
print("Top 20 worst PER")
print("-" * 40)

for _, row in df.nlargest(20, "per").iterrows():
    print(f"PER:      {row['per']:.4f}")
    print(f"Sentence: {row['sentence']}")
    print(f"Target:   {row['target']}")
    print(f"Predict:  {row['predicted']}")
    print("-" * 40)

if word_embeddings:
    print("=" * 40)
    print("Computing UMAP projection...")

    CLASS_COLORS = {
        "Mabilis": "#E24B4A",
        "Maragsa": "#EF9F27",
        "Malumi": "#1D9E75",
        "Malumay": "#378ADD",
        "None": "#888780",
        "Nonstandard": "#7F77DD",
    }

    X = np.stack(word_embeddings)
    reducer = umap.UMAP(n_components=2, random_state=765)
    X_2d = reducer.fit_transform(X)

    # Create a DataFrame for easy plotting
    plot_df = pd.DataFrame(
        {"UMAP-1": X_2d[:, 0], "UMAP-2": X_2d[:, 1], "Stress class": word_labels}
    )

    # Generate interactive scatter plot
    fig = px.scatter(
        plot_df,
        x="UMAP-1",
        y="UMAP-2",
        color="Stress class",
        color_discrete_map=CLASS_COLORS,
        title="Interactive UMAP of ByT5 Encoder Word Embeddings",
        template="plotly_white",
        render_mode="webgl",  # Faster rendering for many points
    )

    # Save as interactive HTML instead of static PNG
    plot_path = f"results/pruned_umap_{Path(args.checkpoint_path).parts[-2]}_{Path(args.checkpoint_path).parts[-1]}_{args.dataset}.png"
    html_path = plot_path.replace(".png", ".html")
    fig.write_html(html_path)
    print(f"Interactive UMAP saved to {html_path}")

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls, color in CLASS_COLORS.items():
        mask = np.array(word_labels) == cls
        if mask.any():
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=color,
                label=cls,
                s=12,
                alpha=0.7,
                linewidths=0,
            )

    legend_handles = [
        mpatches.Patch(color=c, label=l)
        for l, c in CLASS_COLORS.items()
        if l in word_labels
    ]
    ax.legend(
        handles=legend_handles, title="Stress class", fontsize=9, title_fontsize=9
    )
    ax.set_title("UMAP of ByT5 encoder word embeddings (mean-pooled by word span)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"UMAP plot saved to {plot_path}")
