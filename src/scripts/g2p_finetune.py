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

from src.utils.dataset_from_csv import dataset_from_csv_list
from src.utils.preview_callback import PreviewCallback

RANDOM_STATE = 765  # ナムコプロ最強

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = ["data/tatoeba/phonetic_tatoeba_gemini_3.csv"]

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument(
    "--dataset",
    default="tatoeba",
    choices=["tatoeba", "newsph-nli", "combined", "stress", "combined-stress"],
)
parser.add_argument("--description", default="sample_finetune")
parser.add_argument("--resume-from", default=None)
parser.add_argument("--finetune-from", default=None)
parser.add_argument("--freeze", action="store_true")
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--include-word-level", action="store_true")
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
elif args.dataset == "stress":
    dataset = [
        "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        "data/stress-minimal/stress-minimal_single_split.csv",
    ]
elif args.dataset == "combined-stress":
    dataset = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        "data/stress-minimal/stress-minimal_single_split.csv",
    ]
elif args.dataset == "all":
    dataset = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        "data/stress-minimal/stress-minimal_single_split.csv",
        "data/wiktionary-scrape/transcribed/homographs_flattened.csv",
    ]

if args.include_word_level:
    dataset.append("data/multitask/multitask_ds.csv")

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
split_dataset = dataset_from_csv_list(dataset, tokenizer)

# Set up model and DataCollator
if args.finetune_from:
    model = T5ForConditionalGeneration.from_pretrained(args.finetune_from)
else:
    model = T5ForConditionalGeneration.from_pretrained(args.model_id)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Create a new directory for this run...
run_id = datetime.now().strftime("%Y-%m-%d_%H-%M")
custom_path = f"./models/checkpoints/{run_id}_{args.description}"

if args.freeze:
    for i, block in enumerate(model.encoder.block):
        if i < 4:
            for param in block.parameters():
                param.requires_grad = False
        else:
            break

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Optimizer and trainer configuration

training_args = Seq2SeqTrainingArguments(
    output_dir=custom_path,
    bf16=True,
    optim="adafactor",
    # --------------------------------------------
    # Effective batch size: 16 * 4 = 64
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,
    # --------------------------------------------
    # Use standard ByT5 learning rates...
    learning_rate=args.learning_rate,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=(
        100 if args.dataset == "tatoeba" else (600 if args.dataset == "all" else 300)
    ),
    # --------------------------------------------
    num_train_epochs=10,  # TODO: Is 10 good?
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=10,
    # --------------------------------------------
    # Max generated length of 512 for sentences
    predict_with_generate=True,
    generation_max_length=256,  # 512,
    # --------------------------------------------
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["validation"],  # type: ignore
    data_collator=data_collator,
    callbacks=[PreviewCallback(split_dataset["validation"], data_collator, tokenizer)],
)

if args.resume_from:
    trainer.train(resume_from_checkpoint=args.resume_from)
else:
    trainer.train()
