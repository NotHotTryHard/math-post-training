from pathlib import Path
from types import SimpleNamespace

import pytest

from math_post_training import grpo
from math_post_training.config import load_yaml_config
from math_post_training.prompts.training import MATH_SYSTEM_PROMPT
from math_post_training.rewards import accuracy_reward, boxed_format_reward

GRPO_CONFIGS = [
    Path("configs/config.example.yaml"),
    *sorted(Path("configs/grpo").glob("*.yaml")),
]


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def map(self, function, remove_columns):
        assert remove_columns == self.column_names
        return FakeDataset([function(row) for row in self.rows])


class FakeTokenizer:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(Path(path))


class FakeMergedModel:
    def __init__(self):
        self.saved = None

    def save_pretrained(self, path, *, safe_serialization):
        self.saved = (Path(path), safe_serialization)


class FakeModel:
    def __init__(self):
        self.merged = FakeMergedModel()

    def merge_and_unload(self):
        return self.merged


class FakeSftModel:
    def __init__(self, merged):
        self.merged = merged

    def merge_and_unload(self):
        return self.merged


class FakeTrainer:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = FakeModel()
        self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        self.resume_from_checkpoint = None
        self.saved_adapter = None
        FakeTrainer.instance = self

    def train(self, *, resume_from_checkpoint):
        self.resume_from_checkpoint = resume_from_checkpoint

    def save_model(self, path):
        self.saved_adapter = Path(path)


class RecordingConfig:
    def __init__(self, **kwargs):
        self.values = kwargs


class RecordingLoraConfig:
    def __init__(self, **kwargs):
        self.values = kwargs


def test_train_grpo_prepares_data_and_saves_adapter_and_merged_model(monkeypatch, tmp_path):
    rows = [
        {
            "problem": "What is 2 + 2?",
            "solution": "2 + 2 = 4",
            "answer": "4",
            "source": "fixture",
        },
        {
            "problem": "What is 3 + 3?",
            "solution": "3 + 3 = 6",
            "answer": "6",
            "source": "fixture",
        },
    ]
    tokenizer = FakeTokenizer()
    train_dataset = FakeDataset(rows)
    eval_dataset = FakeDataset(rows[:1])
    output_dir = tmp_path / "grpo"
    config = {
        "experiment": {"name": "test-grpo"},
        "model": {
            "name_or_path": "sft-checkpoint",
            "dtype": "bfloat16",
            "trust_remote_code": False,
        },
        "lora": {"r": 32, "lora_alpha": 64},
        "dataset": {"sources": [{}], "validation": {"size": 1}},
        "grpo": {
            "output_dir": str(output_dir),
            "max_steps": 2,
            "use_vllm": True,
        },
    }

    monkeypatch.setattr(grpo, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(grpo, "load_math_dataset", lambda dataset_config: train_dataset)
    monkeypatch.setattr(
        grpo,
        "split_train_validation",
        lambda dataset, validation_config: (dataset, eval_dataset),
    )
    monkeypatch.setattr(grpo, "GRPOConfig", RecordingConfig)
    monkeypatch.setattr(grpo, "LoraConfig", RecordingLoraConfig)
    monkeypatch.setattr(grpo, "GRPOTrainer", FakeTrainer)

    result = grpo.train_grpo(config, resume_from_checkpoint=tmp_path / "checkpoint-1")

    trainer = FakeTrainer.instance
    assert result == output_dir
    assert trainer.kwargs["model"] == "sft-checkpoint"
    assert trainer.kwargs["reward_funcs"] == [accuracy_reward, boxed_format_reward]
    assert trainer.kwargs["train_dataset"].rows[0] == {
        "prompt": [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "answer": "4",
        "source": "fixture",
    }
    assert trainer.kwargs["eval_dataset"].rows[0]["answer"] == "4"
    assert trainer.kwargs["peft_config"].values == {"r": 32, "lora_alpha": 64}
    assert trainer.kwargs["args"].values == {
        "output_dir": str(output_dir),
        "max_steps": 2,
        "use_vllm": True,
        "run_name": "test-grpo",
        "model_init_kwargs": {
            "dtype": "bfloat16",
            "trust_remote_code": False,
            "attn_implementation": "sdpa",
        },
    }
    assert trainer.resume_from_checkpoint == str(tmp_path / "checkpoint-1")
    assert trainer.saved_adapter == output_dir / "adapter"
    assert trainer.model.merged.saved == (output_dir, True)
    assert tokenizer.saved_paths == [output_dir / "adapter", output_dir]


def test_prepare_dataset_requires_visible_columns():
    dataset = FakeDataset([{"problem": "problem"}])
    dataset.column_names = None

    with pytest.raises(
        ValueError,
        match="Training dataset does not expose its column names",
    ):
        grpo._prepare_dataset(dataset, name="Training")


def test_merge_adapter_loads_base_weights_then_merges_sft_lora(monkeypatch):
    base_model = object()
    merged_model = object()
    calls = {}

    def load_base(model_name, **kwargs):
        calls["base"] = (model_name, kwargs)
        return base_model

    def load_adapter(model, adapter_name):
        calls["adapter"] = (model, adapter_name)
        return FakeSftModel(merged_model)

    monkeypatch.setattr(grpo.AutoModelForCausalLM, "from_pretrained", load_base)
    monkeypatch.setattr(grpo.PeftModel, "from_pretrained", load_adapter)

    result = grpo._merge_adapter(
        "Qwen/Qwen2.5-1.5B",
        "owner/sft-adapter",
        {"dtype": "bfloat16", "attn_implementation": "sdpa"},
    )

    assert result is merged_model
    assert calls == {
        "base": (
            "Qwen/Qwen2.5-1.5B",
            {"dtype": "bfloat16", "attn_implementation": "sdpa"},
        ),
        "adapter": (base_model, "owner/sft-adapter"),
    }


@pytest.mark.parametrize("path", GRPO_CONFIGS)
def test_grpo_configs_are_accepted_by_trl(path):
    training_config = dict(load_yaml_config(path)["grpo"])
    training_config.update(bf16=False, tf32=False)

    args = grpo.GRPOConfig(**training_config)

    assert args.generation_batch_size % args.num_generations == 0
    assert args.reward_weights == [1.0, 0.1]
