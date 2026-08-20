from types import SimpleNamespace

from math_post_training.generation import vllm
from math_post_training.generation.base import GenerationConfig


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.tokenizer = object()
        self.generate_calls = []

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params, use_tqdm):
        self.generate_calls.append((prompts, sampling_params, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(index=1, text=" second "),
                    SimpleNamespace(index=0, text=" first "),
                ]
            )
            for _ in prompts
        ]


def _backend(monkeypatch, **engine_args):
    monkeypatch.setattr(vllm, "_load_vllm", lambda: (FakeLLM, FakeSamplingParams))
    return vllm.VLLMBackend("fake-model", **engine_args)


def test_vllm_backend_initializes_engine_and_exposes_tokenizer(monkeypatch):
    backend = _backend(
        monkeypatch,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
    )

    assert backend.llm.init_kwargs == {
        "model": "fake-model",
        "dtype": "bfloat16",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.85,
        "generation_config": "vllm",
    }
    assert backend.tokenizer is backend.llm.tokenizer


def test_vllm_backend_maps_sampling_options_and_preserves_prompt_order(monkeypatch):
    backend = _backend(monkeypatch)
    config = GenerationConfig(
        max_new_tokens=128,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        seed=17,
        stop_strings=["Question:"],
    )

    completions = backend.generate(["one", "two"], config)

    assert completions == [["first", "second"], ["first", "second"]]
    prompts, sampling_params, use_tqdm = backend.llm.generate_calls[0]
    assert prompts == ["one", "two"]
    assert use_tqdm is False
    assert sampling_params.kwargs == {
        "n": 2,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "seed": 17,
        "stop": ["Question:"],
    }


def test_vllm_backend_uses_greedy_sampling_and_handles_empty_input(monkeypatch):
    backend = _backend(monkeypatch)
    config = GenerationConfig(do_sample=False, temperature=0.8, top_p=0.95)

    assert backend.generate([], config) == []
    backend.generate(["prompt"], config)

    _, sampling_params, _ = backend.llm.generate_calls[0]
    assert sampling_params.kwargs["temperature"] == 0.0
    assert sampling_params.kwargs["top_p"] == 1.0
