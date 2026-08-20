<div align="center">

<h1>ByT5-Based Stress-Aware Sentence-Level Filipino G2P</h1>

[![HuggingFace Model](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg)](https://huggingface.co/lowestofthelow/filipino-byt5-g2p)

</div>

This repository is for a sentence-level Filipino G2P project using a ByT5-based model, pre-trained on multilingual word-level G2P data (See [lingjzhu/charsiug2p](https://github.com/lingjzhu/charsiug2p)) and on three sentence-level G2P datasets annotated with an LLM-assisted pipeline guided by Wiktionary data.

> [!WARNING]
> For more information, an arXiv link to the camera-ready manuscript will be included here in the near future.

## News

- `[2026-08-07]` Paper accepted to [NLPIR 2026](https://www.nlpir.net/index.html).

## Project structure

```
├── data                       # Datasets
│   ├── newsph-nli
│   ├── stress-minimal         # "Naive synthetic" dataset
│   ├── tatoeba
│   ├── wikipron
│   └── wiktionary-scrape      # "Wiktionary-guided" dataset
│       ├── generated
│       └── transcribed
├── models
│   └── checkpoints
├── results
└── src
    ├── datasets
    ├── scripts
    └── utils
        └── generate_prompt
```

## Setting up

### Installing dependencies

This project uses [`uv`](https://docs.astral.sh/uv/) to manage packages.

1. Create a Python 3.12 virtual environment with `uv venv --python 3.12`.
2. Run `uv sync` to install dependencies.
3. Activate the virtual environment with `source .venv/bin/activate`.

### Set up `pre-commit`

1. `pre-commit` should have been installed as a development dependency. Check
   with `pre-commit --version`.
2. Install the hook scripts with `pre-commit install`.
3. Run `pre-commit run --all-files` to run the pre-commit hooks on all files.

---

> [!WARNING]
> The following sections are WIP. Check back later.

## Inference quickstart

## Reproduction

---

## Datasets

All datasets are currently found in the `data/` directory.
