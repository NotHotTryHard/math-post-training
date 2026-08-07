"""Load and normalize mathematical datasets."""

from datasets import interleave_datasets, load_dataset

from math_post_training.data.sources import (
    gsm1k,
    gsm8k,
    hendrycks_math,
    open_math_instruct_2,
)

NORMALIZERS = {
    "gsm1k": gsm1k.normalize,
    "gsm8k": gsm8k.normalize,
    "hendrycks_math": hendrycks_math.normalize,
    "open_math_instruct_2": open_math_instruct_2.normalize,
}


def load_math_dataset(config):
    """Load and optionally mix the experiment's dataset sources."""

    sources = config["sources"]
    datasets = [load_math_source(source) for source in sources]

    if len(datasets) == 1:
        return datasets[0]

    probabilities = [source["probability"] for source in sources]

    return interleave_datasets(
        datasets,
        probabilities=probabilities,
        seed=config["seed"],
        stopping_strategy=config["stopping_strategy"],
    )


def load_math_source(config):
    """Load and normalize one source before it is used alone or in a mixture."""

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
