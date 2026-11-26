<div align="center">

<h1>ByT5-Based Stress-Aware Sentence-Level Filipino G2P</h1>

[![HuggingFace Model](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg)](https://huggingface.co/lowestofthelow/filipino-byt5-g2p)
[![HuggingFace Space](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/lowestofthelow/filipino-byt5-g2p-demo)
[![arXiv](https://img.shields.io/badge/arXiv-2609.09974-b31b1b.svg)](https://arxiv.org/abs/2609.09974)

</div>

This repository is for a sentence-level Filipino G2P project using a ByT5-based model, pre-trained on multilingual word-level G2P data (See [lingjzhu/charsiug2p](https://github.com/lingjzhu/charsiug2p)) and on three sentence-level G2P datasets annotated with an LLM-assisted pipeline guided by Wiktionary data.

## News

- `[2026-09-09]` Paper uploaded to [arXiv](https://arxiv.org/abs/2609.09974).
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

## Inference quickstart

You can load the fine-tuned model directly from HuggingFace (`lowestofthelow/filipino-byt5-g2p`):

```python
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

model_id = "lowestofthelow/filipino-byt5-g2p"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id).to(device)
model.eval()

sentence = "Bukas ay bukas pa ang tindahan."
inputs = tokenizer(sentence, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=256, num_beams=5)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

Or run single-sentence inference via the CLI script:

```bash
python -m src.scripts.g2p_inference "Bukas ay bukas pa ang tindahan." --checkpoint-path lowestofthelow/filipino-byt5-g2p
```

(If you only want a quick test, a live demo is available on [HuggingFace Spaces](https://huggingface.co/spaces/lowestofthelow/filipino-byt5-g2p-demo))

## Reproduction

### Fine-tuning

Run `src/scripts/g2p_finetune.py` (or the interactive wrapper `python -m src.scripts.cli_finetune`):

```bash
python -m src.scripts.g2p_finetune --dataset all
```

`--dataset` accepts `tatoeba`, `newsph-nli`, `combined`, `stress`, `combined-stress`, and `all` (the full configuration combining Tatoeba, NewsPH-NLI, naive synthetic, and Wiktionary-guided synthetic datasets; see paper).

### Evaluation

Run `src/scripts/g2p_eval.py` (or `python -m src.scripts.cli_eval`) against the manually-corrected test set (`--dataset manual`):

```bash
python -m src.scripts.g2p_eval --checkpoint-path <path_to_checkpoint> --dataset manual
```

## Datasets

All datasets are stored under `data/`:

| Path | Description |
| --- | --- |
| `data/tatoeba/` | Filipino sentences from Tatoeba transcribed into IPA with Gemini 3 Flash (`phonetic_tatoeba_gemini_3.csv`). |
| `data/newsph-nli/` | NewsPH-NLI sentences transcribed into IPA with Gemini 2.5 Flash-Lite (`phonetic_newsph-nli_gemini_2.5_lite.csv`). |
| `data/stress-minimal/` | "Naive synthetic" dataset targeting stress contrasts (`stress-minimal_ambiguous_split.csv`, `stress-minimal_single_split.csv`). |
| `data/wiktionary-scrape/` | "Wiktionary-guided" synthetic homograph dataset generated and transcribed using Wiktionary pronunciations (`transcribed/homographs_flattened.csv`). |
| `data/manual_set_1.csv`, `data/manual_set_2.csv` | Manually-corrected evaluation set used for the paper's benchmark (`--dataset manual`). |
