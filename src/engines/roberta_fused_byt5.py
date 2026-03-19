import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration


class RoBERTaFusedByT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        roberta_name="jcblaise/roberta-tagalog-base",
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.fusion_enabled = False

        model.roberta = AutoModel.from_pretrained(roberta_name)
        model.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_name)

        for param in model.roberta.parameters():
            param.requires_grad = False

        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device

        model.proj = nn.Linear(768, model.config.d_model).to(dtype=dtype, device=device)
        nn.init.normal_(model.proj.weight, std=0.01)
        nn.init.zeros_(model.proj.bias)

        return model

    def _fuse_roberta_into_bytes(self, sentences, hidden, device):
        roberta_reprs = torch.zeros_like(hidden)  # (B, T, d_model)

        # Tokenize on CPU first so word_ids() works, THEN move to device
        encoded = self.roberta_tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )

        # Extract before moving to device (these are CPU-side metadata)
        offset_mapping = encoded.pop("offset_mapping")  # (B, R, 2)
        word_ids_per_batch = [
            encoded.word_ids(batch_index=b) for b in range(len(sentences))
        ]

        encoded = encoded.to(device)

        with torch.no_grad():
            out = self.roberta(**encoded).last_hidden_state  # (B, R, 768)

        for b, sentence in enumerate(sentences):
            word_ids = word_ids_per_batch[b]
            offsets = offset_mapping[b].tolist()

            valid_wids = [wid for wid in word_ids if wid is not None]
            if not valid_wids:
                continue
            num_words = max(valid_wids) + 1

            # Build per-word mean representations
            word_reprs = []
            for w in range(num_words):
                token_indices = [i for i, wid in enumerate(word_ids) if wid == w]
                if not token_indices:
                    continue
                word_reprs.append(out[b, token_indices].mean(dim=0))

            if not word_reprs:
                continue

            word_reprs = self.proj(torch.stack(word_reprs))  # (num_words, d_model)

            # Scatter word reprs into byte positions
            T = hidden.size(1)
            for w in range(len(word_reprs)):
                token_indices = [i for i, wid in enumerate(word_ids) if wid == w]
                if not token_indices:
                    continue
                char_start = offsets[token_indices[0]][0]
                char_end = offsets[token_indices[-1]][1]

                byte_start = len(sentence[:char_start].encode("utf-8"))
                byte_end = len(sentence[:char_end].encode("utf-8"))

                if byte_start >= T:
                    continue
                byte_end = min(byte_end, T)

                span_len = byte_end - byte_start
                if span_len > 0:
                    roberta_reprs[b, byte_start:byte_end] = (
                        word_reprs[w].unsqueeze(0).expand(span_len, -1)
                    )

        return hidden + roberta_reprs

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
            hidden = encoder_outputs.last_hidden_state

            if sentence is not None and self.fusion_enabled:
                hidden = self._fuse_roberta_into_bytes(sentence, hidden, hidden.device)
                encoder_outputs.last_hidden_state = hidden

        return super().forward(
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
