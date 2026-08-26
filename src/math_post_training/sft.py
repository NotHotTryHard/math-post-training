"""Supervised fine-tuning with TRL."""

from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from math_post_training.artifacts import save_policy_artifacts
from math_post_training.data.loaders import (
    load_math_dataset,
    split_train_validation,
)
from math_post_training.data.preprocessing import to_sft_example
from math_post_training.model import load_tokenizer, prepare_math_policy_tokenizer


def train_sft(config, *, resume_from_checkpoint=None):
    """Train and save the model described by an experiment config."""

    model_config = config["model"]
    model_name = model_config["name_or_path"]
    training_config = dict(config["sft"])
    eos_loss_weight = training_config.pop("eos_loss_weight", 1.0)
    if eos_loss_weight != 1.0:
        loss_type = training_config.get("loss_type", "nll")
        if loss_type != "nll":
            raise ValueError("sft.eos_loss_weight requires sft.loss_type: nll")
        training_config["loss_type"] = "nll"
    output_dir = Path(training_config["output_dir"])
    lora_config = LoraConfig(**config["lora"])

    tokenizer = load_tokenizer(
        model_name,
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

    train_dataset = _prepare_dataset(
        train_dataset,
        name="Training",
        eos_token=tokenizer.eos_token,
    )
    if eval_dataset is not None:
        eval_dataset = _prepare_dataset(
            eval_dataset,
            name="Validation",
            eos_token=tokenizer.eos_token,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        compute_loss_func=_weighted_eos_loss(tokenizer.eos_token_id, eos_loss_weight),
    )
    trainer.train(
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        )
    )
    save_policy_artifacts(trainer, tokenizer, output_dir)

    return output_dir


def _weighted_eos_loss(eos_token_id, eos_loss_weight):
    """Return token NLL with additional weight on the assistant EOS label."""

    if eos_loss_weight < 1.0:
        raise ValueError("sft.eos_loss_weight must be at least 1.0")
    if eos_loss_weight == 1.0:
        return None

    def compute_loss(outputs, labels, num_items_in_batch=None):
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        flat_labels = shift_labels.view(-1)
        weights = torch.ones_like(token_loss)
        weights[flat_labels == eos_token_id] = eos_loss_weight
        weighted_loss = (token_loss * weights).sum()
        denominator = (
            num_items_in_batch if num_items_in_batch is not None else (flat_labels != -100).sum()
        )
        return weighted_loss / denominator.clamp_min(1)

    return compute_loss


def _prepare_dataset(dataset, *, name, eos_token):
    """Convert one normalized math split into the native-EOS SFT format."""

    original_columns = dataset.column_names
    if original_columns is None:
        raise ValueError(f"{name} dataset does not expose its column names")
    return dataset.map(
        to_sft_example,
        fn_kwargs={"eos_token": eos_token},
        remove_columns=original_columns,
    )
