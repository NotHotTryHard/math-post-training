import sys
from types import SimpleNamespace

from math_post_training import cli
from math_post_training.model import QWEN_BASE_EOS_TOKEN
from math_post_training.prompts.training import build_math_prompt


class RecordingTokenizer:
    eos_token = QWEN_BASE_EOS_TOKEN
    eos_token_id = 151643


class FakeBackend:
    def __init__(self, model, tokenizer, device):
        pass

    def generate(self, prompts, config):
        assert prompts == [build_math_prompt("hello")]
        return [["completion"]]


def test_generate_can_show_exact_rendered_prompt(monkeypatch, capsys):
    config = {
        "model": {"name_or_path": "fake-model"},
        "inference": {"backend": "transformers"},
    }
    monkeypatch.setattr(cli, "load_yaml_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "load_model_and_tokenizer",
        lambda **kwargs: (object(), RecordingTokenizer()),
    )
    monkeypatch.setattr(cli, "TransformersBackend", FakeBackend)

    assert cli.generate_main(["hello", "--show-prompt"]) == 0
    assert capsys.readouterr().out == (
        f"=== prompt sent to model ===\n{build_math_prompt('hello')}"
        "\n=== completion ===\ncompletion\n"
    )


def test_grpo_command_starts_training_and_prints_output(monkeypatch, capsys, tmp_path):
    config_path = tmp_path / "experiment.yaml"
    checkpoint = tmp_path / "checkpoint-10"
    output_dir = tmp_path / "trained-model"
    config = {"experiment": {"name": "test-grpo"}}
    call = {}

    monkeypatch.setattr(cli, "load_yaml_config", lambda path: config)

    def train(config_arg, *, resume_from_checkpoint, stop_after_step):
        call["config"] = config_arg
        call["resume"] = resume_from_checkpoint
        call["stop_after_step"] = stop_after_step
        return output_dir

    monkeypatch.setitem(
        sys.modules,
        "math_post_training.grpo",
        SimpleNamespace(train_grpo=train),
    )

    assert (
        cli.grpo_main(
            [
                "--config",
                str(config_path),
                "--resume-from-checkpoint",
                str(checkpoint),
            ]
        )
        == 0
    )
    assert call == {"config": config, "resume": checkpoint, "stop_after_step": None}
    assert capsys.readouterr().out == f"Model: {output_dir}\n"
