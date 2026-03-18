import subprocess
import sys
from pathlib import Path

import questionary

# Define the available datasets for finetuning
DATASETS = ["tatoeba", "newsph-nli", "combined"]
TRAINERS = ["Seq2SeqTrainer", "StressWeightedSeq2SeqTrainer"]

# Prompt user to select a training set
selected_dataset = questionary.select(
    "Select a dataset for finetuning:", choices=DATASETS, use_indicator=True
).ask()

if not selected_dataset:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select a model trainer
selected_trainer = questionary.select(
    "Select a model trainer:", choices=TRAINERS, use_indicator=True
).ask()

if not selected_trainer:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user for a run description
description = questionary.text("Enter a description for this run:").ask()

if description is None:
    print("Cancelled by user.")
    sys.exit(0)

# NOTE: We run the Python script with uv!
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.g2p_finetune",
    "--dataset",
    selected_dataset,
    "--trainer",
    selected_trainer,
    "--description",
    description,
]

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
