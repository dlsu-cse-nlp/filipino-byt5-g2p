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

        # Multi-head cross-attention
        # Queries from ByT5, keys/values from RoBERTa
        model.cross_attn = nn.MultiheadAttention(
            embed_dim=model.config.d_model,
            num_heads=8,
            kdim=768,
            vdim=768,
            batch_first=True,
        ).to(dtype=dtype, device=device)

        # Small init on output projection to start near-zero
        nn.init.normal_(model.cross_attn.out_proj.weight, std=0.01)
        nn.init.zeros_(model.cross_attn.out_proj.bias)

        return model

    def _fuse_via_cross_attention(self, sentences, hidden, device):
        encoded = self.roberta_tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Invert attention_mask for MultiheadAttention
        roberta_key_padding_mask = (encoded["attention_mask"] == 0).to(device)

        encoded = encoded.to(device)

        with torch.no_grad():
            roberta_out = self.roberta(
                **encoded, output_hidden_states=True
            ).hidden_states[
                8
            ]  # (B, R, 768)

        fused, _ = self.cross_attn(
            query=hidden,  # (B, T, d_model)
            key=roberta_out,  # (B, R, 768)
            value=roberta_out,  # (B, R, 768)
            key_padding_mask=roberta_key_padding_mask,  # (B, R)
        )

        return hidden + fused

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
                hidden = self._fuse_via_cross_attention(sentence, hidden, hidden.device)
                encoder_outputs.last_hidden_state = hidden

        return super().forward(
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        sentence = kwargs.pop("sentence", None)

        if self.fusion_enabled and sentence is not None:
            original_forward = self.encoder.forward

            def fused_forward(*args, **encoder_kwargs):
                outputs = original_forward(*args, **encoder_kwargs)
                hidden = outputs.last_hidden_state
                print(f"[DEBUG] fused_forward called, hidden shape: {hidden.shape}")
                fused = self._fuse_via_cross_attention(sentence, hidden, hidden.device)
                print(f"[DEBUG] fusion applied")
                print(f"[DEBUG] fused norm: {fused.norm().item()}")
                print(f"[DEBUG] hidden norm: {hidden.norm().item()}")
                print(f"[DEBUG] diff norm: {(fused - hidden).norm().item()}")
                print(f"[DEBUG] fused nan: {torch.isnan(fused).any()}")
                outputs.last_hidden_state = fused
                return outputs

            self.encoder.forward = fused_forward
            result = super().generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
            self.encoder.forward = original_forward
            return result

        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
