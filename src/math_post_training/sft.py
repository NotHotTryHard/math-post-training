"""Supervised fine-tuning with TRL."""

from pathlib import Path

from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from math_post_training.data.loaders import load_math_dataset
from math_post_training.data.preprocessing import to_sft_example

QWEN_CHAT_EOS_TOKEN = "<|im_end|>"


def train_sft(config, *, resume_from_checkpoint=None):
    """Train and save the model described by an experiment config."""

    model_config = config["model"]
    model_name = model_config["name_or_path"]
    training_config = dict(config["sft"])
    output_dir = Path(training_config["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    if QWEN_CHAT_EOS_TOKEN in tokenizer.get_vocab():
        tokenizer.eos_token = QWEN_CHAT_EOS_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_math_dataset(config["dataset"])
    original_columns = dataset.column_names
    if original_columns is None:
        raise ValueError("Training dataset does not expose its column names")
    dataset = dataset.map(
        to_sft_example,
        remove_columns=original_columns,
    )

    training_config.setdefault("run_name", config["experiment"]["name"])
    training_config["output_dir"] = str(output_dir)
    training_config["model_init_kwargs"] = {
        "dtype": model_config.get("dtype", "auto"),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "attn_implementation": "sdpa",
    }

    trainer = SFTTrainer(
        model=model_name,
        args=SFTConfig(**training_config),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return output_dir
