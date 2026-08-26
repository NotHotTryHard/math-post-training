"""Save training outputs in one stable policy checkpoint layout."""

from pathlib import Path

from math_post_training.model import prepare_math_policy_tokenizer

ROOT_ADAPTER_FILENAMES = (
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
)
TOKENIZER_TEMPLATE_FILENAMES = ("chat_template.jinja",)


def save_policy_artifacts(trainer, tokenizer, output_dir):
    """Save a PEFT adapter and a clean merged checkpoint for inference."""

    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    prepare_math_policy_tokenizer(tokenizer)

    trainer.save_model(adapter_dir)
    _remove_files(adapter_dir, TOKENIZER_TEMPLATE_FILENAMES)
    tokenizer.save_pretrained(adapter_dir)

    _remove_files(output_dir, (*ROOT_ADAPTER_FILENAMES, *TOKENIZER_TEMPLATE_FILENAMES))
    model = trainer.accelerator.unwrap_model(trainer.model)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _remove_files(output_dir, (*ROOT_ADAPTER_FILENAMES, *TOKENIZER_TEMPLATE_FILENAMES))

    return adapter_dir


def _remove_files(directory, filenames):
    for filename in filenames:
        (directory / filename).unlink(missing_ok=True)
