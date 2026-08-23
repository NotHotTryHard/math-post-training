"""Shared loading of causal language models and their tokenizers."""

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_BASE_EOS_TOKEN = "<|endoftext|>"


def load_tokenizer(name_or_path, *, trust_remote_code=False):
    """Load a tokenizer and make it safe to use for batched causal generation."""

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def require_qwen_base_eos(tokenizer):
    """Fail before training if a ChatML/Instruct tokenizer slipped into the policy."""

    if tokenizer.eos_token != QWEN_BASE_EOS_TOKEN or tokenizer.eos_token_id is None:
        raise ValueError(
            "Math policy training requires a Qwen Base tokenizer with "
            f"eos_token={QWEN_BASE_EOS_TOKEN!r}; got {tokenizer.eos_token!r}"
        )


def prepare_math_policy_tokenizer(tokenizer):
    """Validate native EOS and prevent saved policy checkpoints from exposing ChatML."""

    require_qwen_base_eos(tokenizer)
    tokenizer.chat_template = None
    getattr(tokenizer, "init_kwargs", {}).pop("chat_template", None)
    return tokenizer


def load_model_and_tokenizer(
    name_or_path,
    *,
    dtype="auto",
    trust_remote_code=False,
    device=None,
):
    """Load a Hugging Face model ID or a local ``save_pretrained`` checkpoint."""

    tokenizer = load_tokenizer(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if device is not None:
        model.to(device)

    return model, tokenizer
