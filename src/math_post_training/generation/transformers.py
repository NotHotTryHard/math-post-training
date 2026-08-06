"""Text generation through Hugging Face Transformers."""

import torch
from transformers import set_seed


class TransformersBackend:
    """A thin adapter around ``model.generate``."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.tokenizer.padding_side = "left"
        self.model.eval()

    def generate(self, prompts, config):
        if not prompts:
            return []

        if config.seed is not None:
            set_seed(config.seed)

        inputs = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(self.device)
        prompt_width = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            if config.do_sample:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    num_return_sequences=config.num_return_sequences,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    num_return_sequences=config.num_return_sequences,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        flat_completions = self.tokenizer.batch_decode(
            output_ids[:, prompt_width:],
            skip_special_tokens=True,
        )
        count = config.num_return_sequences
        return [
            [completion.strip() for completion in flat_completions[start : start + count]]
            for start in range(0, len(flat_completions), count)
        ]
