"""Shared loading of causal language models and their tokenizers."""

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    name_or_path,
    *,
    dtype="auto",
    trust_remote_code=False,
    device=None,
):
    """Load a Hugging Face model ID or a local ``save_pretrained`` checkpoint."""

    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    if device is not None:
        model.to(device)

    return model, tokenizer
