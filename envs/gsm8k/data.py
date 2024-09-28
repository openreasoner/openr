from datasets import load_dataset


def get_train_test_dataset(*args, **kwargs):
    num_train_data = kwargs.get("num_train_data", None)
    if num_train_data:
        train_dataset = load_dataset("gsm8k", "main", split=f"train[:{num_train_data}]")
    else:
        train_dataset = load_dataset("gsm8k", "main", split=f"train")

    test_dataset = load_dataset("gsm8k", "main")["test"]
    return train_dataset, test_dataset
