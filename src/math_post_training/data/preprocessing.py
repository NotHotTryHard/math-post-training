"""Create stage-specific rows from canonical mathematical examples."""

from math_post_training.model import QWEN_BASE_EOS_TOKEN
from math_post_training.prompts.training import build_math_prompt

CHATML_CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>")


def to_sft_example(example, *, eos_token=QWEN_BASE_EOS_TOKEN):
    """Build a plain prompt-completion row ending in exactly one native EOS."""

    if not example["solution"]:
        raise ValueError("SFT example has no solution")
    if eos_token != QWEN_BASE_EOS_TOKEN:
        raise ValueError(f"SFT requires native EOS {QWEN_BASE_EOS_TOKEN!r}")

    prompt = build_math_prompt(example["problem"])
    completion = example["solution"].rstrip()
    _reject_chatml(prompt, field="prompt")
    _reject_chatml(completion, field="completion")
    while completion.endswith(eos_token):
        completion = completion.removesuffix(eos_token).rstrip()
    if eos_token in completion:
        raise ValueError("completion contains native EOS before its end")
    completion += eos_token
    return {
        "prompt": prompt,
        "completion": completion,
    }


def to_grpo_example(example):
    """Build a prompt row while keeping metadata used by GRPO rewards."""

    if not example["answer"]:
        raise ValueError("GRPO example has no reference answer")

    prompt = build_math_prompt(example["problem"])
    _reject_chatml(prompt, field="prompt")
    return {
        "prompt": prompt,
        "answer": example["answer"],
        "source": example["source"],
    }


def _reject_chatml(text, *, field):
    """Prevent old ChatML control tokens from leaking into the new protocol."""

    token = next((token for token in CHATML_CONTROL_TOKENS if token in text), None)
    if token is not None:
        raise ValueError(f"{field} contains forbidden ChatML token {token!r}")
