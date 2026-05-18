<div align="center">

CEARTIN Machine Project

<h1>ByT5-Based Fine-Tuned Stress-Aware Filipino G2P Model</h1>

</div>

This serves as the main Git repository for a Filipino NLP project with ByT5.

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

### Fine-tuning

1. Run `src/scripts/g2p_finetune.py` with the `--dataset` argument, which
   accepts `"tatoeba"`, `"newsph-nli"`, and `"combined"`

```
python3 -m src.scripts.g2p_finetune --dataset "tatoeba"
```

> [!NOTE] A note on the `manual_set` files in `data/`
> TLDR: Use `manual_set.csv`. The others aren't actually manually fixed, so
  ignore them for now.
