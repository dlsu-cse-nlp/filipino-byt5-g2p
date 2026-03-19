from transformers import TrainerCallback


class PreviewCallback(TrainerCallback):
    """Shows a sample evaluation at the end of a training epoch"""

    def __init__(self, val_dataset, data_collator, tokenizer):
        super().__init__()
        self.val_dataset = val_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        # Take 3 samples from the validation set
        samples = self.val_dataset.select(range(3))

        keys_to_keep = ["input_ids", "attention_mask", "labels", "sentence"]
        batch_list = [
            {k: v for k, v in samples[i].items() if k in keys_to_keep}
            for i in range(len(samples))
        ]

        inputs = self.data_collator(batch_list)

        model_inputs = {
            "input_ids": inputs["input_ids"].to(args.device),
            "attention_mask": inputs["attention_mask"].to(args.device),
            "sentence": inputs["sentence"],
        }

        generated_tokens = kwargs["model"].generate(**model_inputs, max_length=128)

        preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        labels = self.tokenizer.batch_decode(inputs["labels"], skip_special_tokens=True)

        print("=" * 40)
        print(f"Epoch: {state.epoch}")
        print("-" * 40)
        for p, l in zip(preds, labels):
            print(f"Target:  {l}")
            print(f"Predict: {p}")
        print("=" * 40)
