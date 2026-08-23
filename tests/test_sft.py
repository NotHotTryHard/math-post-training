from types import SimpleNamespace

import pytest
import torch

from math_post_training.model import QWEN_BASE_EOS_TOKEN, require_qwen_base_eos
from math_post_training.sft import _weighted_eos_loss


def test_weighted_eos_loss_increases_only_eos_contribution():
    logits = torch.tensor(
        [[[0.0, 0.0], [2.0, -2.0], [0.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, 1]])
    outputs = SimpleNamespace(logits=logits)

    plain = _weighted_eos_loss(eos_token_id=1, eos_loss_weight=1.0)
    weighted = _weighted_eos_loss(eos_token_id=1, eos_loss_weight=20.0)

    assert plain is None
    assert weighted(outputs, labels) > 10


def test_weighted_eos_loss_rejects_downweighting():
    with pytest.raises(ValueError, match="at least 1.0"):
        _weighted_eos_loss(eos_token_id=1, eos_loss_weight=0.5)


def test_policy_training_requires_qwen_base_native_eos():
    tokenizer = SimpleNamespace(eos_token=QWEN_BASE_EOS_TOKEN, eos_token_id=151643)
    require_qwen_base_eos(tokenizer)

    tokenizer.eos_token = "<|im_end|>"
    with pytest.raises(ValueError, match="Qwen Base tokenizer"):
        require_qwen_base_eos(tokenizer)
