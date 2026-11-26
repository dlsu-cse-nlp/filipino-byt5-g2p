import subprocess
import sys

import questionary

DATASETS = ["tatoeba", "newsph-nli", "combined", "manual", "synthetic2", "validation"]


def run_epitran_cli():
    selected_dataset = questionary.select(
        "Choose a testing dataset to evaluate with Epitran:",
        choices=DATASETS,
        use_indicator=True,
    ).ask()

    if not selected_dataset:
        print("Cancelled by user.")
        sys.exit(0)

    cmd = [
        "uv",
        "run",
        "python3",
        "-m",
        "src.scripts.epitran_eval",
        "--dataset",
        selected_dataset,
    ]

    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_epitran_cli()
