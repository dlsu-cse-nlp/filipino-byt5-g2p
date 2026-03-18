import torch
from transformers import Seq2SeqTrainer

STRESS_BYTE = 0x27 + 3


class StressWeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, *args, stress_weight=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        weights = torch.ones(self.model.config.vocab_size)
        weights[STRESS_BYTE] = stress_weight
        self.stress_weights = weights

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=self.stress_weights.to(model.device), ignore_index=-100
        )(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
