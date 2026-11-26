import subprocess
import sys
from pathlib import Path

import questionary

# Define the available datasets for finetuning
DATASETS = ["tatoeba", "newsph-nli", "combined", "stress", "combined-stress", "all"]
DEFAULT_CHECKPOINTS_PATH = "models/checkpoints"


def get_checkpoints(base_dir=DEFAULT_CHECKPOINTS_PATH):
    path = Path(base_dir)
    if not path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        sys.exit(1)

    # Recursively check all checkpoint subdirectories
    checkpoints = [
        str(p) for p in path.glob("*/*") if p.is_dir() and "checkpoint" in p.name
    ]

    return sorted(checkpoints)


# Prompt user to select a training set
selected_dataset = questionary.select(
    "Select a dataset for finetuning:", choices=DATASETS, use_indicator=True
).ask()

if not selected_dataset:
    print("Cancelled by user.")
    sys.exit(0)

# Ask if the user wants to load (resume/finetune) from a checkpoint
resume_path = None
finetune_path = None
if questionary.confirm("Load from a checkpoint?", default=False).ask():
    available_checkpoints = get_checkpoints()
    if available_checkpoints:
        checkpoint_path = questionary.select(
            "Select checkpoint:", choices=available_checkpoints
        ).ask()

        # Determine if we are resuming or finetuning
        mode = questionary.select(
            "Checkpoint mode:",
            choices=["Resume (Keep optimizer/LR)", "Finetune (New optimizer/LR)"],
        ).ask()

        if "Resume" in mode:
            resume_path = checkpoint_path
        else:
            finetune_path = checkpoint_path
    else:
        print("No checkpoints found. Starting a fresh run.")

# Prompt for learning rate
lr = questionary.text("Enter learning rate:", default="3e-4").ask()

# Prompt for layer freezing
freeze = questionary.confirm("Freeze the first 4 layers?", default=False).ask()

# Prompt to include word-level data
include_words = questionary.confirm("Include word-level data?", default=True).ask()

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
    "--description",
    description,
    "--learning-rate",
    lr,
]

# Add resume OR finetune argument based on selection
if resume_path:
    cmd.extend(["--resume-from", resume_path])
elif finetune_path:
    cmd.extend(["--finetune-from", finetune_path])

if include_words:
    cmd.extend(["--include-word-level"])

# Add freeze flag if selected
if freeze:
    cmd.append("--freeze")

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
