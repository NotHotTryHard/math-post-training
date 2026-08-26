from types import SimpleNamespace

from math_post_training.artifacts import ROOT_ADAPTER_FILENAMES, save_policy_artifacts
from math_post_training.model import QWEN_BASE_EOS_TOKEN


class FakeTokenizer:
    eos_token = QWEN_BASE_EOS_TOKEN
    eos_token_id = 151643

    def __init__(self):
        self.chat_template = "stale template"
        self.init_kwargs = {"chat_template": "stale template"}
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(path)


class FakeMergedModel:
    def __init__(self):
        self.saved = None

    def save_pretrained(self, path, *, safe_serialization):
        self.saved = (path, safe_serialization)


class FakeModel:
    def __init__(self):
        self.merged = FakeMergedModel()

    def merge_and_unload(self):
        return self.merged


class FakeTrainer:
    def __init__(self):
        self.model = FakeModel()
        self.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
        self.saved_adapter = None

    def save_model(self, path):
        self.saved_adapter = path


def test_save_policy_artifacts_keeps_adapter_and_merged_layout_separate(tmp_path):
    output_dir = tmp_path / "policy"
    output_dir.mkdir()
    for filename in (*ROOT_ADAPTER_FILENAMES, "chat_template.jinja"):
        (output_dir / filename).write_text("stale")
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "chat_template.jinja").write_text("stale")
    trainer = FakeTrainer()
    tokenizer = FakeTokenizer()

    result = save_policy_artifacts(trainer, tokenizer, output_dir)

    assert result == adapter_dir
    assert trainer.saved_adapter == adapter_dir
    assert trainer.model.merged.saved == (output_dir, True)
    assert tokenizer.saved_paths == [adapter_dir, output_dir]
    assert tokenizer.chat_template is None
    assert "chat_template" not in tokenizer.init_kwargs
    assert not any((output_dir / filename).exists() for filename in ROOT_ADAPTER_FILENAMES)
    assert not (output_dir / "chat_template.jinja").exists()
    assert not (adapter_dir / "chat_template.jinja").exists()
