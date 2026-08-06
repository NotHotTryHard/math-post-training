"""Load and normalize mathematical datasets."""

from datasets import load_dataset

from math_post_training.data.sources import gsm8k, open_math_instruct_2

NORMALIZERS = {
    "gsm8k": gsm8k.normalize,
    "open_math_instruct_2": open_math_instruct_2.normalize,
}


def load_math_dataset(config):
    """Load one dataset described by the experiment's ``dataset`` section."""

    normalizer = NORMALIZERS[config["adapter"]]
    streaming = config.get("streaming", False)

    dataset = load_dataset(
        config["path"],
        name=config.get("subset"),
        revision=config["revision"],
        split=config["split"],
        streaming=streaming,
    )

    shuffle_seed = config.get("shuffle_seed")
    if shuffle_seed is not None:
        if streaming:
            dataset = dataset.shuffle(
                seed=shuffle_seed,
                buffer_size=config["shuffle_buffer_size"],
            )
        else:
            dataset = dataset.shuffle(seed=shuffle_seed)

    limit = config.get("limit")
    if limit is not None:
        if limit < 1:
            raise ValueError("dataset.limit must be positive")
        if streaming:
            dataset = dataset.take(limit)
        else:
            dataset = dataset.select(range(min(limit, len(dataset))))

    original_columns = dataset.column_names
    if original_columns is None:
        raise ValueError("Dataset does not expose its column names")

    def normalize(row):
        return vars(normalizer(row))

    return dataset.map(normalize, remove_columns=original_columns)
