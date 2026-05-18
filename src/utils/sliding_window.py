import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def predict_sliding_window(sentence, model, tokenizer, window_size=11):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    words = sentence.split()
    N = len(words)

    # if sentence is less than window size
    if N <= window_size:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    half_window = window_size // 2

    all_windows = []
    target_indices = []

    # windows
    for i in range(N):
        start = i - half_window
        end = start + window_size

        if start < 0:
            start = 0
            end = window_size

        if end > N:
            end = N
            start = N - window_size

        window_words = words[start:end]
        # start of windows capitalized
        window_text = " ".join(window_words).capitalize()

        all_windows.append(window_text)
        target_indices.append(i - start)

    inputs = tokenizer(all_windows, padding=True, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
        )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    final_phonemes = []
    for window_output, target_idx in zip(decoded_outputs, target_indices):
        phoneme_words = window_output.split()

        if phoneme_words:
            safe_idx = min(target_idx, len(phoneme_words) - 1)
            final_phonemes.append(phoneme_words[safe_idx])
        else:
            final_phonemes.append("")

    return " ".join(final_phonemes)
