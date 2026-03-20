from datasets import DatasetDict, concatenate_datasets, load_dataset

RANDOM_STATE = 765  # ナムコプロ最強


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        text_target=examples["phoneme"],
        max_length=128,  # TODO: Confirm if we're gonna use this limit
        padding="max_length",  # Forces everything to have the same length
        truncation=True,  # This truncates when it generates > max_length
    )


def is_within_max_length(item, tokenizer, max_length=128):
    # Filter out digits
    if any(c.isdigit() for c in item["sentence"]):
        return False

    # Filter out sentences that start with a lowercase letter
    if item["sentence"][0].islower():
        return False

    inputs = tokenizer(item["sentence"], return_tensors="pt")
    return inputs["input_ids"].shape[1] < max_length


def dataset_from_csv(csv_path, tokenizer):
    dataset = load_dataset("csv", data_files=csv_path)["train"]
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

    # Perform an 80%/10%/10% train-test-validation split
    train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=RANDOM_STATE)
    val_test = train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_STATE)

    val_test["test"] = val_test["test"].filter(
        lambda item: is_within_max_length(item, tokenizer),
        load_from_cache_file=False,
    )

    split_dataset = DatasetDict(
        {
            "train": train_test["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        }
    )

    return split_dataset


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

    return DatasetDict(concatenated_splits)
