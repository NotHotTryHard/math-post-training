"""Shared loading of causal language models and their tokenizers."""

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_CHAT_EOS_TOKEN = "<|im_end|>"


def load_tokenizer(name_or_path, *, trust_remote_code=False, eos_token=None):
    """Load a tokenizer and make it safe to use for batched causal generation."""

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if eos_token is not None and eos_token in tokenizer.get_vocab():
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
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
