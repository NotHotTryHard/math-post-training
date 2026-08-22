"""Reinforcement learning with verifiable rewards through TRL's GRPO trainer."""

from pathlib import Path

from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from math_post_training.data.loaders import load_math_dataset, split_train_validation
from math_post_training.data.preprocessing import to_grpo_example
from math_post_training.model import QWEN_CHAT_EOS_TOKEN, load_tokenizer
from math_post_training.rewards import accuracy_reward, boxed_format_reward

REWARD_FUNCTIONS = [accuracy_reward, boxed_format_reward]


def train_grpo(config, *, resume_from_checkpoint=None):
    """Train a LoRA policy with online generations and verifiable rewards."""

    model_config = config["model"]
    model_name = model_config["name_or_path"]
    training_config = dict(config["grpo"])
    output_dir = Path(training_config["output_dir"])

    tokenizer = load_tokenizer(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
        eos_token=QWEN_CHAT_EOS_TOKEN,
    )
    train_dataset = load_math_dataset(config["dataset"])
    eval_dataset = None
    validation_config = config["dataset"].get("validation")
    if validation_config is not None:
        train_dataset, eval_dataset = split_train_validation(
            train_dataset,
            validation_config,
        )

    train_dataset = _prepare_dataset(train_dataset, name="Training")
    if eval_dataset is not None:
        eval_dataset = _prepare_dataset(eval_dataset, name="Validation")

    training_config.setdefault("run_name", config["experiment"]["name"])
    training_config["output_dir"] = str(output_dir)
    training_config["model_init_kwargs"] = {
        "dtype": model_config.get("dtype", "auto"),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "attn_implementation": "sdpa",
    }

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=REWARD_FUNCTIONS,
        args=GRPOConfig(**training_config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(**config["lora"]),
    )
    trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    _save_model(trainer, tokenizer, output_dir)
    return output_dir


def _prepare_dataset(dataset, *, name):
    """Convert one normalized math split into TRL's GRPO prompt format."""

    original_columns = dataset.column_names
    if original_columns is None:
        raise ValueError(f"{name} dataset does not expose its column names")
    return dataset.map(
        to_grpo_example,
        remove_columns=original_columns,
    )


def _save_model(trainer, tokenizer, output_dir):
    """Save both the lightweight adapter and a merged inference checkpoint."""

    adapter_dir = output_dir / "adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model = trainer.accelerator.unwrap_model(trainer.model)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
