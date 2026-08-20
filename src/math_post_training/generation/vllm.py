"""Text generation through the in-process vLLM engine."""


def _load_vllm():
    """Import vLLM lazily so non-Linux development environments still work."""

    try:
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise RuntimeError(
            "The vLLM backend requires the Linux-only `vllm` dependency group"
        ) from error
    return LLM, SamplingParams


class VLLMBackend:
    """A thin adapter around vLLM's offline batched inference API."""

    def __init__(self, name_or_path, **engine_args):
        llm_class, self.sampling_params_class = _load_vllm()
        engine_args.setdefault("generation_config", "vllm")
        self.llm = llm_class(model=name_or_path, **engine_args)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts, config):
        if not prompts:
            return []

        sampling_kwargs = {
            "n": config.num_return_sequences,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.do_sample else 0.0,
            "top_p": config.top_p if config.do_sample else 1.0,
            "seed": config.seed,
            "stop": config.stop_strings,
        }
        if config.do_sample and config.top_k is not None:
            sampling_kwargs["top_k"] = config.top_k

        sampling_params = self.sampling_params_class(
            **sampling_kwargs,
        )
        outputs = self.llm.generate(list(prompts), sampling_params, use_tqdm=False)
        return [
            [
                completion.text.strip()
                for completion in sorted(output.outputs, key=lambda item: item.index)
            ]
            for output in outputs
        ]
