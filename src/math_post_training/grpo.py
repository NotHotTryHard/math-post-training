"""Reinforcement learning with verifiable rewards through TRL's GRPO trainer."""

from pathlib import Path

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from math_post_training.data.loaders import load_math_dataset, split_train_validation
from math_post_training.data.preprocessing import to_grpo_example
from math_post_training.model import load_tokenizer, prepare_math_policy_tokenizer
from math_post_training.rewards import (
    accuracy_reward,
    boxed_format_reward,
    strict_boxed_reward,
)

REWARD_FUNCTIONS = [accuracy_reward, boxed_format_reward]
REWARD_PROFILES = {
    "accuracy_and_format": REWARD_FUNCTIONS,
    "strict_boxed": [strict_boxed_reward],
}


def train_grpo(config, *, resume_from_checkpoint=None):
    """Train a LoRA policy with online generations and verifiable rewards."""

    model_config = config["model"]
    model_name = model_config["name_or_path"]
    adapter_name = model_config.get("adapter_name_or_path")
    training_config = dict(config["grpo"])
    output_dir = Path(training_config["output_dir"])
    reward_funcs = _reward_functions(config.get("rewards"))

    tokenizer = load_tokenizer(
        adapter_name or model_name,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    prepare_math_policy_tokenizer(tokenizer)
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
    model_init_kwargs = _model_init_kwargs(model_config)
    if adapter_name is None:
        policy = model_name
        training_config["model_init_kwargs"] = model_init_kwargs
    else:
        policy = _merge_adapter(model_name, adapter_name, model_init_kwargs)

    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=reward_funcs,
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


def _model_init_kwargs(model_config):
    return {
        "dtype": model_config.get("dtype", "auto"),
        "trust_remote_code": model_config.get("trust_remote_code", False),
        "attn_implementation": "sdpa",
    }


def _merge_adapter(model_name, adapter_name, model_init_kwargs):
    """Merge an SFT adapter before attaching a fresh trainable RL adapter."""

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_init_kwargs,
    )
    sft_model = PeftModel.from_pretrained(base_model, adapter_name)
    return sft_model.merge_and_unload()


def _prepare_dataset(dataset, *, name):
    """Convert one normalized math split into TRL's GRPO prompt format."""

    original_columns = dataset.column_names
    if original_columns is None:
        raise ValueError(f"{name} dataset does not expose its column names")
    return dataset.map(
        to_grpo_example,
        remove_columns=original_columns,
    )


def _reward_functions(config):
    """Resolve an explicit reward profile while preserving the legacy default."""

    profile = "accuracy_and_format" if config is None else config["profile"]
    try:
        return REWARD_PROFILES[profile]
    except KeyError as error:
        supported = ", ".join(sorted(REWARD_PROFILES))
        raise ValueError(
            f"Unknown rewards.profile {profile!r}; expected one of: {supported}"
        ) from error


def _save_model(trainer, tokenizer, output_dir):
    """Save both the lightweight adapter and a merged inference checkpoint."""

    adapter_dir = output_dir / "adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    model = trainer.accelerator.unwrap_model(trainer.model)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
