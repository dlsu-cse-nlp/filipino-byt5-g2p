import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration


class RoBERTaFusedByT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        # Add a learnable linear layer that maps from RoBERTa's hidden state to
        # ByT5's hidden state
        self.proj = nn.Linear(768, config.d_model)
        nn.init.normal_(self.proj.weight, std=0.01)  # small init
        nn.init.zeros_(self.proj.bias)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        roberta_name="jcblaise/roberta-tagalog-base",
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load the Tagalog RoBERTa model from HuggingFace
        model.roberta = AutoModel.from_pretrained(roberta_name)
        model.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_name)

        # Remove RoBERTa parameters from state dict
        for param in model.roberta.parameters():
            param.requires_grad = False

        return model

    def _get_roberta_word_reprs(self, sentence, device):
        encoded = self.roberta_tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            roberta_out = self.roberta(**encoded).last_hidden_state

        return self.proj(roberta_out)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        sentence=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

            # Extract the hidden state from the RoBERTa model
            hidden = encoder_outputs.last_hidden_state

            # TODO: This is a crude "alignment"
            # We need to align the subword embeddings from RoBERTa with the
            # byte-level embeddings from ByT5, but I'm too lazy right now so we
            # just pad with zeroes so they're the same length...
            if sentence is not None:
                roberta_reprs = self._get_roberta_word_reprs(sentence, hidden.device)

                R, T = roberta_reprs.size(1), hidden.size(1)

                # Case when subwords > bytes (pretty much never happening)
                if R > T:
                    roberta_reprs = roberta_reprs[:, :T, :]
                # Case when bytes > subwords (most cases)
                elif R < T:
                    pad = torch.zeros(
                        hidden.size(0), T - R, hidden.size(2), device=hidden.device
                    )
                    roberta_reprs = torch.cat([roberta_reprs, pad], dim=1)

                # TODO: We just do addition as a form of fusion. Need to
                # investigate more expressive approaches...
                hidden = hidden + roberta_reprs
                encoder_outputs.last_hidden_state = hidden

        return super().forward(
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
