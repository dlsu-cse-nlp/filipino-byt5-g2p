from transformers import TrainerCallback


class EnableFusionCallback(TrainerCallback):
    def __init__(self, enable_at_epoch=3):
        self.enable_at_epoch = enable_at_epoch

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch == self.enable_at_epoch:
            kwargs["model"].fusion_enabled = True
            print(f"[INFO] RoBERTa fusion enabled at epoch {state.epoch}")
