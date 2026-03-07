import grapheme
import torch
from transformers import DataCollatorForSeq2Seq  # Corrected to Seq2Seq collator
from transformers import TrainerCallback  # Needed for the custom generation callback
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.optimization import Adafactor

from datasets import DatasetDict, load_dataset

PHONEME_INVENTORY = [
    "|",
    "'",
    "ŋ",
    "ɕ",
    "ə",
    "ɡ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",
    "ˌ",
    # NOTE: We allow both normal and joined tie bars
    "\u0361",
    " ‍͡ ",
    " ",
]


model_name = "jcblaise/roberta-tagalog-base"
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

num_added_toks = tokenizer.add_tokens(PHONEME_INVENTORY)
new_vocab_size = len(tokenizer)

model.encoder.resize_token_embeddings(new_vocab_size)
model.decoder.resize_token_embeddings(new_vocab_size)

model.config.vocab_size = new_vocab_size
model.config.decoder.vocab_size = new_vocab_size

dataset = load_dataset("csv", data_files="data/phonetic_tatoeba_gemini_3.csv")["train"]

train_test = dataset.train_test_split(test_size=0.2, seed=42)
eval_test = train_test["test"].train_test_split(test_size=0.5, seed=42)

split_dataset = DatasetDict(
    {
        "train": train_test["train"],
        "validation": eval_test["train"],
        "test": eval_test["test"],
    }
)


def format_g(text):
    spaced = " ".join(grapheme.graphemes(text))
    return spaced.replace("   ", " | ")


def preprocess_function(examples):
    # TODO: Space the inputs too?
    # inputs = [format_g(x) for x in examples["original_sentence"]]
    inputs = examples["original_sentence"]
    targets = [format_g(x) for x in examples["processed_output"]]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=256,
        truncation=True,
        padding="max_length",
    )

    # TODO: Negative padding
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in model_inputs["labels"]
    ]

    return model_inputs


tokenized_dataset = split_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


class SampleGenerationCallback(TrainerCallback):
    """Generates a sample prediction at the end of each epoch."""

    def __init__(self, tokenizer, sample_data):
        self.tokenizer = tokenizer
        self.sample_data = sample_data

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()

        input_text = self.sample_data["original_sentence"]
        target_text = self.sample_data["processed_output"]

        inputs = self.tokenizer(input_text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_start_token_id=model.config.decoder_start_token_id,
                min_length=15,
                max_length=128,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print("-" * 40)
        print(f"Input:      {input_text}")
        print(f"Target:     {target_text}")
        print(f"Prediction: {prediction}")
        print("-" * 40)


sample_for_callback = split_dataset["validation"][0]

# ------------------------------------------------------------------------------

# We define the Adafactor optimizer based on the following paper:
# https://arxiv.org/abs/2111.06053
# TODO: Check if I got everything right

optimizer = Adafactor(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.1,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/checkpoints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    weight_decay=0.1,
    fp16=True,
    logging_steps=100,
    eval_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,  # Added this: standard for Seq2Seq tasks
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    optimizers=(optimizer, None),
    callbacks=[
        SampleGenerationCallback(tokenizer, sample_for_callback)
    ],  # Inject custom callback here
)

trainer.train()
