"""
This script begins a finetune of the charsiu/g2p_multilingual_byT5_small_100
model using the HuggingFace API.
"""

# TODO: Resolve the following warning (?):
# "The tied weights mapping and config for this model specifies to tie
# shared.weight to decoder.embed_tokens.weight, but both are present in the
# checkpoints, so we will NOT tie them. You should update the config with
# `tie_word_embeddings=False` to silence this warning."

import argparse
from datetime import datetime

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from src.engines.roberta_fused_byt5 import RoBERTaFusedByT5
from src.utils.dataset_from_csv import dataset_from_csv_list
from src.utils.enable_fusion_callback import EnableFusionCallback
from src.utils.preview_callback import PreviewCallback

RANDOM_STATE = 765  # ナムコプロ最強

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = ["data/tatoeba/phonetic_tatoeba_gemini_3.csv"]

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument(
    "--dataset", default="tatoeba", choices=["tatoeba", "newsph-nli", "combined"]
)
parser.add_argument("--description", default="sample_finetune")
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

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
split_dataset = dataset_from_csv_list(dataset, tokenizer)


class RoBERTaFusedDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, **kwargs):
        sentences = [f.pop("sentence") for f in features]
        # Remove any other non-tensor columns
        for f in features:
            f.pop("phoneme", None)
            f.pop("index", None)
        batch = super().__call__(features, **kwargs)
        batch["sentence"] = sentences
        return batch


# Set up model and DataCollator
# TODO: Add to CLI
model = RoBERTaFusedByT5.from_pretrained(args.model_id)
data_collator = RoBERTaFusedDataCollator(tokenizer=tokenizer, model=model)

# Create a new directory for this run...
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
custom_path = f"./models/checkpoints/{run_id}_{args.description}"

# Optimizer and trainer configuration

training_args = Seq2SeqTrainingArguments(
    output_dir=custom_path,
    bf16=True,
    optim="adafactor",
    # --------------------------------------------
    # Effective batch size: 16 * 4 = 64
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    # --------------------------------------------
    # Use standard ByT5 learning rates...
    learning_rate=3e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=100 if args.dataset == "tatoeba" else 300,
    # --------------------------------------------
    num_train_epochs=15,  # TODO: Is 10 good?
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=10,
    # --------------------------------------------
    # Max generated length of 512 for sentences
    predict_with_generate=True,
    generation_max_length=512,
    # --------------------------------------------
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["validation"],  # type: ignore
    data_collator=data_collator,
    callbacks=[
        EnableFusionCallback,
        PreviewCallback(split_dataset["validation"], data_collator, tokenizer),
    ],
)

trainer.train()
