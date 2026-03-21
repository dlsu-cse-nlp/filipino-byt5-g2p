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
        roberta_injection_layer=8,
        encoder_injection_layer=8,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.fusion_enabled = True
        model.roberta_injection_layer = roberta_injection_layer
        model.encoder_injection_layer = encoder_injection_layer

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

    def _get_roberta_hidden(self, sentences, device):
        """Extract RoBERTa hidden states at the configured injection layer."""
        encoded = self.roberta_tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        roberta_key_padding_mask = (encoded["attention_mask"] == 0).to(device)
        encoded = encoded.to(device)

        with torch.no_grad():
            roberta_out = self.roberta(
                **encoded, output_hidden_states=True
            ).hidden_states[
                self.roberta_injection_layer
            ]  # (B, R, 768)

        return roberta_out, roberta_key_padding_mask

    def _install_fusion_hook(self, sentences, device):
        """
        Installs a forward hook on encoder block[encoder_injection_layer]
        that injects cross-attention after that block runs, allowing
        upper encoder layers to re-contextualize the fused representations.

        Returns the hook handle — caller must call handle.remove() when done.
        """
        roberta_out, roberta_key_padding_mask = self._get_roberta_hidden(
            sentences, device
        )

        def hook(module, input, output):
            # T5Block output is a tuple; first element is hidden states
            hidden = output[0]
            fused, attn_weights = self.cross_attn(
                query=hidden,
                key=roberta_out,
                value=roberta_out,
                key_padding_mask=roberta_key_padding_mask,
            )
            print(
                f"attn_weights entropy: {-(attn_weights.clamp(min=1e-9) * attn_weights.clamp(min=1e-9).log()).sum(-1).mean()}"
            )
            return (hidden + fused,) + output[1:]

        handle = self.encoder.block[self.encoder_injection_layer].register_forward_hook(
            hook
        )
        return handle

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
            if sentence is not None and self.fusion_enabled:
                device = (
                    input_ids.device if input_ids is not None else attention_mask.device
                )
                handle = self._install_fusion_hook(sentence, device)
                try:
                    encoder_outputs = self.encoder(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                finally:
                    handle.remove()
            else:
                encoder_outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )

        return super().forward(
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        sentence = kwargs.pop("sentence", None)

        if self.fusion_enabled and sentence is not None:
            device = (
                input_ids.device if input_ids is not None else attention_mask.device
            )
            handle = self._install_fusion_hook(sentence, device)
            try:
                result = super().generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs,
                )
            finally:
                handle.remove()
            return result

        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
