"""Load and normalize mathematical datasets."""

from datasets import Features, IterableDataset, Value, interleave_datasets, load_dataset

from math_post_training.data.sources import (
    gsm1k,
    gsm8k,
    hendrycks_math,
    mmlu,
    open_math_instruct_2,
)

NORMALIZERS = {
    "gsm1k": gsm1k.normalize,
    "gsm8k": gsm8k.normalize,
    "hendrycks_math": hendrycks_math.normalize,
    "mmlu": mmlu.normalize,
    "open_math_instruct_2": open_math_instruct_2.normalize,
}

NORMALIZED_FEATURES = Features(
    {
        "problem": Value("string"),
        "solution": Value("string"),
        "answer": Value("string"),
        "source": Value("string"),
    }
)


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


def split_train_validation(dataset, config):
    """Remove a deterministic validation holdout from a training dataset."""

    size = config["size"]
    if size < 1:
        raise ValueError("dataset.validation.size must be positive")

    seed = config.get("seed", 42)
    if isinstance(dataset, IterableDataset):
        shuffled = dataset.shuffle(
            seed=seed,
            buffer_size=config.get("shuffle_buffer_size", 10_000),
        )
        train_dataset = shuffled.skip(size)
        validation_dataset = shuffled.take(size)
        return _make_replayable(train_dataset), _make_replayable(validation_dataset)

    if size >= len(dataset):
        raise ValueError(
            "dataset.validation.size must be smaller than the training dataset"
        )

    split = dataset.train_test_split(test_size=size, seed=seed)
    return split["train"], split["test"]


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

    return dataset.map(
        normalize,
        remove_columns=original_columns,
        features=NORMALIZED_FEATURES,
    )


def _make_replayable(dataset):
    """Replay one fixed iterable sequence without reshuffling its data sources."""

    return IterableDataset.from_generator(
        lambda: iter(dataset),
        features=dataset.features,
    )
