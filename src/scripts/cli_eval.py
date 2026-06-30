import subprocess
import sys
from pathlib import Path

import questionary

DEFAULT_CHECKPOINTS_PATH = "models/checkpoints"

DATASETS = ["tatoeba", "newsph-nli", "combined", "manual", "synthetic2"]


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


# List all available checkpoints
checkpoints = get_checkpoints()
checkpoints = ["BASE MODEL (Charsiu byT5)", "Epitran"] + checkpoints

if not checkpoints:
    print("No checkpoints found.")
    sys.exit(1)

# Prompt user to select a checkpoint
selected_checkpoint = questionary.select(
    "Choose a checkpoint to evaluate:", choices=checkpoints, use_indicator=True
).ask()

if not selected_checkpoint:
    print("Cancelled by user.")
    sys.exit(0)

# Prompt user to select a test set
selected_dataset = questionary.select(
    "Choose a testing dataset:", choices=DATASETS, use_indicator=True
).ask()

if not selected_dataset:
    print("Cancelled by user.")
    sys.exit(0)

umap = False
if questionary.confirm("Get UMAP plot?", default=False).ask():
    umap = True

# NOTE: We run the Python script with uv!
cmd = [
    "uv",
    "run",
    "python3",
    "-m",
    "src.scripts.g2p_eval",
    "--checkpoint-path",
    selected_checkpoint,
    "--dataset",
    selected_dataset,
]

if selected_checkpoint == "BASE MODEL (Charsiu byT5)":
    cmd.append("--base-model")
elif selected_checkpoint == "Epitran":
    cmd.append("--epitran")
else:
    cmd.extend(["--checkpoint-path", selected_checkpoint])

if umap:
    cmd.append("--umap")

print(f"\nRunning command: {' '.join(cmd)}\n")

subprocess.run(cmd, check=True)
