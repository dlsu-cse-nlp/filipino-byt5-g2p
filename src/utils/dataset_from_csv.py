import panphon
import torch
from scipy import stats

from datasets import DatasetDict, concatenate_datasets, load_dataset

RANDOM_STATE = 765  # ナムコプロ最強
ft = panphon.FeatureTable()


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        text_target=examples["phoneme"],
        max_length=256,  # TODO: Confirm if we're gonna use this limit
        padding="max_length",  # Forces everything to have the same length
        truncation=True,  # This truncates when it generates > max_length
    )


def capitalize_sentence(item):
    if item["sentence"] and item["sentence"].strip():
        item["sentence"] = item["sentence"].strip()
        item["sentence"] = item["sentence"][0].upper() + item["sentence"][1:]
    return item


def is_valid(item):
    if not item["sentence"] or not item["sentence"].strip():
        return False
    if any(c.isdigit() for c in item["sentence"]):
        return False
    if len(item["sentence"].encode("utf-8")) > 512:
        return False
    return True


def preprocess_dataset(dataset):
    print(f"Dataset length before filtering: {len(dataset)}")
    dataset = dataset.filter(is_valid, load_from_cache_file=False)
    print(f"Dataset length after filtering: {len(dataset)}")

    dataset = dataset.map(capitalize_sentence, load_from_cache_file=False)

    # Print sentence lengths
    lengths = [len(item["sentence"].encode("utf-8")) for item in dataset]
    avg = sum(lengths) / len(lengths)
    std = (sum((l - avg) ** 2 for l in lengths) / len(lengths)) ** 0.5

    print(
        f"Sentence lengths (bytes): min={min(lengths)}, max={max(lengths)}, avg={avg:.1f}, std={std:.1f}"
    )

    if len(lengths) >= 8:
        k2_stat, p_value = stats.normaltest(lengths)
        # If p-value > 0.05, we fail to reject the null hypothesis (i.e. normal distribution)
        is_normal = p_value > 0.05
        normality_msg = f"Yes" if is_normal else f"No"
        normality_msg += f" (p-value={p_value:.3e})"
    else:
        normality_msg = "Not enough data to test (n < 8)"

    print(normality_msg)

    return dataset


def dataset_from_csv(csv_path, tokenizer):
    dataset = load_dataset("csv", data_files=csv_path)["train"]

    # Filter out duplicate sentences
    sentences = dataset["sentence"]
    unique_indices = []
    seen = set()

    for i, s in enumerate(sentences):
        if s not in seen:
            unique_indices.append(i)
            seen.add(s)

    dataset = dataset.select(unique_indices)

    dataset = dataset.filter(
        lambda x: x["phoneme"] is not None and str(x["phoneme"]).strip() != ""
    )

    # Preprocessing
    dataset = preprocess_dataset(dataset)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    # Perform an 80%/10%/10% train-test-validation split
    train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=RANDOM_STATE)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_STATE)

    split_dataset = DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    split_dataset["train"] = split_dataset["train"].shuffle(seed=RANDOM_STATE)
    split_dataset["test"] = split_dataset["test"].filter(
        lambda x: len(x["input_ids"]) <= 256
    )

    return split_dataset


def save_splits_for_csv(csv_path, tokenizer):
    ds_dict = dataset_from_csv(csv_path, tokenizer)
    base = csv_path.removesuffix(".csv")
    for split_name, dataset in ds_dict.items():
        out_path = f"{base}_{split_name}.csv"
        dataset.select_columns(["index", "sentence", "phoneme"]).to_csv(
            out_path, index=False
        )
        print(f"Wrote {len(dataset)} rows to {out_path}")


def dataset_from_csv_list(csv_paths, tokenizer):
    grouped_splits = {}

    for path in csv_paths:
        ds_dict = dataset_from_csv(path, tokenizer)

        # Group splits by "train", "test", and "validation"
        for split_name, dataset in ds_dict.items():
            if split_name not in grouped_splits:
                grouped_splits[split_name] = []
            grouped_splits[split_name].append(dataset)

    # Concatenate the lists of datasets for each split
    concatenated_splits = {}
    for split_name, dataset_list in grouped_splits.items():
        concatenated_splits[split_name] = concatenate_datasets(dataset_list)

    concatenated_dict = DatasetDict(concatenated_splits)

    print(
        sorted(
            set("".join(["".join(ds["phoneme"]) for ds in concatenated_dict.values()]))
        )
    )

    print(
        sorted(
            set(
                phoneme
                for ds in concatenated_dict.values()
                for text in ds["phoneme"]
                for phoneme in ft.ipa_segs(text)
            )
        )
    )

    return concatenated_dict


if __name__ == "__main__":
    from transformers import AutoTokenizer

    csv_paths = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
        "data/stress-minimal/stress-minimal_ambiguous_split.csv",
        "data/stress-minimal/stress-minimal_single_split.csv",
    ]

    tokenizer = AutoTokenizer.from_pretrained("charsiu/g2p_multilingual_byT5_small_100")

    for csv_path in csv_paths:
        print(f"\nProcessing {csv_path}")
        save_splits_for_csv(csv_path, tokenizer)
