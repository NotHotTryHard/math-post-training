# Reproducibility and usage

The repository README is the experiment report. This document contains the operational reference:
environment setup, Docker/RunPod execution, training, evaluation, checkpoint recovery, and the code
layout.

## Requirements

- Python `3.12`;
- Linux + NVIDIA GPU for vLLM training/evaluation;
- a Hugging Face read token for public inputs and write access for private checkpoint uploads;
- optional Weights & Biases credentials;
- Docker with the NVIDIA Container Toolkit for the most reproducible path.

## Environment

Copy the example and fill in only the services you use:

```bash
cp .env.example .env
```

```dotenv
HF_HOME=/workspace/.cache/huggingface
HF_TOKEN=...
WANDB_DIR=/workspace/wandb
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=...
GHCR_TOKEN=...
```

`.env` is ignored by Git. Never place access tokens in YAML configs or logs.

## Local installation

Development and CPU-only tests:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .
```

GPU evaluation/training dependencies:

```bash
uv sync --group dev --group train --group eval --group vllm
```

## Docker

Build the pinned image:

```bash
docker build -t math-post-training .
```

Run it with GPU access and enough shared memory for vLLM:

```bash
docker run \
  --gpus all \
  --ipc=host \
  --env-file .env \
  -v "$PWD/outputs:/workspace/outputs" \
  -v "$PWD/.cache:/workspace/.cache" \
  -it math-post-training bash
```

The CI workflow also publishes immutable `sha-<commit>` tags and a moving `dev` tag to
[`ghcr.io/nothottryhard/math-post-training`](https://github.com/NotHotTryHard/math-post-training/pkgs/container/math-post-training).
For an exact reproduction, prefer the SHA tag recorded for the experiment over `dev`.

## RunPod

1. Create a GPU pod using the GHCR image.
2. Mount persistent storage at `/workspace`.
3. Add the `.env` values as RunPod secrets/environment variables.
4. Use `--ipc=host` or an equivalent large shared-memory setting.
5. Connect by SSH and run commands from `/app`.

The measured production setting was one A100 80 GB. Hardware changes do not require config changes
for correctness, but training microbatch, vLLM memory utilization, and evaluation batch size should
be re-benchmarked.

## Configuration model

Every experiment is a self-contained YAML file. It pins:

- the model or adapter;
- dataset paths, revisions, splits, and optional sampling probabilities;
- the SFT/GRPO optimizer and LoRA geometry;
- generation and vLLM settings;
- benchmark revisions and evaluation protocol;
- output directories and private Hub repository IDs.

`configs/config.example.yaml` is the complete schema-oriented example. Frozen experiments live in
`configs/sft`, `configs/grpo`, and `configs/eval`.

Dataset adapters normalize source-specific columns into one math schema. DeepMath retains
`difficulty`, `topic`, `final_answer`, and the original solution fields. Sampling/filtering belongs
in YAML or a versioned curation script rather than an unrecorded notebook cell.

## Native-EOS policy contract

SFT, GRPO, post-training evaluation, and manual generation use one plain-text prompt:

```text
Solve the following math problem step by step. Put your final answer within \boxed{}.

Problem: <problem>

Solution:
```

SFT appends exactly one `<|endoftext|>` to each completion. The loaders reject ChatML control
tokens, an Instruct tokenizer EOS, or a saved chat template. This makes a merged checkpoint usable
by vLLM without silently changing the policy interface.

## Supervised fine-tuning

Run the selected native-EOS experiment:

```bash
model-sft \
  --config configs/sft/qwen2_5_1_5b_base_openmath_296k_1epoch_lora.yaml
```

Resume an interrupted Trainer checkpoint:

```bash
model-sft \
  --config configs/sft/qwen2_5_1_5b_base_openmath_296k_1epoch_lora.yaml \
  --resume-from-checkpoint /workspace/outputs/sft/<experiment>/checkpoint-4000
```

The LoRA adapter is stored under `<output_dir>/adapter`; merged vLLM-compatible weights are stored
at the output root. Trainer checkpoints include optimizer, scheduler, RNG, and trainer state.

## GRPO / DAPO

Run the final full-GSM8K configuration directly:

```bash
model-grpo \
  --config configs/grpo/qwen2_5_1_5b_openmath_gsm8k_full_dapo_strict_kl1e3_long.yaml
```

Resume exactly from a complete checkpoint:

```bash
model-grpo \
  --config configs/grpo/qwen2_5_1_5b_openmath_gsm8k_full_dapo_strict_kl1e3_long.yaml \
  --resume-from-checkpoint /workspace/outputs/grpo/<experiment>/checkpoint-2500
```

The built-in TRL backend handles colocated vLLM generation, synchronization of updated policy
weights, importance ratios, group-scaled advantages, clipping, and the optional PEFT reference
policy KL term.

### Integrated train/eval sweep

The production loop trains to the next configured boundary, exits to free GPU memory, performs a
full external evaluation, then resumes with optimizer/scheduler state intact:

```bash
model-grpo-eval-loop \
  --config configs/grpo/qwen2_5_1_5b_openmath_gsm8k_full_dapo_strict_kl1e3_long.yaml
```

The config controls checkpoint steps, benchmark config, evaluation batch size, and optional early
stopping. `patience: null` records the curve without stopping. A numeric patience should only be
used with a real development set, not the final test benchmarks.

## Evaluation

Full greedy evaluation of the native-EOS SFT model:

```bash
model-eval \
  --config configs/eval/qwen2_5_1_5b_openmath_native_eos_greedy.yaml
```

Override the model without duplicating a benchmark config:

```bash
model-eval \
  --config configs/eval/qwen2_5_1_5b_openmath_native_eos_greedy.yaml \
  --model /workspace/outputs/grpo/<experiment>/checkpoint-2500
```

Fast diagnostic run:

```bash
WANDB_MODE=offline model-eval \
  --config configs/eval/qwen2_5_1_5b_openmath_native_eos_greedy.yaml \
  --limit 10
```

`--limit` uses deterministic seeded sampling; MATH and MMLU-STEM are balanced round-robin across
subsets rather than taking the first rows.

### Self-consistency@8

The SC@8 config samples eight responses per example and selects the plurality answer after
normalizing the extracted final answers:

```bash
model-eval \
  --config configs/eval/qwen2_5_1_5b_openmath_kl_dapo_sc8.yaml \
  --model /workspace/outputs/grpo/<experiment>/checkpoint-3000
```

Ties use the first generated candidate deterministically. Predictions preserve all eight raw
rollouts, individual accuracy, observed pass@8, vote ties, format/parse rates, truncation, and
completion lengths.

### Evaluate every saved checkpoint

```bash
model-eval-checkpoints \
  --config configs/eval/qwen2_5_1_5b_openmath_native_eos_greedy.yaml \
  --checkpoint-root /workspace/outputs/grpo/<experiment> \
  --output-root /workspace/outputs/eval/<experiment> \
  --work-dir /workspace/work/<experiment> \
  --batch-size 768
```

The command is sequential and resumable. Completed summaries are reused; it never runs multiple GPU
evaluation stages concurrently.

## Manual generation

```bash
model-generate "If x + 3 = 7, what is x?"
```

Inspect the exact rendered prompt:

```bash
model-generate --show-prompt "If x + 3 = 7, what is x?"
```

Send a raw prompt without the math template:

```bash
model-generate \
  --model Qwen/Qwen2.5-1.5B \
  --raw "The answer to 2 + 2 is"
```

## Durable outputs

Each evaluation creates:

- `summary.json` — protocol, timings, accuracy, parse/format, truncation, and length metrics;
- `predictions.jsonl.gz` — compressed per-example prompts, outputs, extracted answers, and scores;
- `sweep-summary.json` for checkpoint sweeps;
- optional W&B scalar histories and prediction tables.

Each training output contains:

- `checkpoint-*` Trainer state for exact resume;
- `adapter/` for the final PEFT adapter;
- merged root weights for vLLM;
- tokenizer metadata with EOS `<|endoftext|>` and no chat template.

## Repository map

```text
configs/
├── sft/                    # frozen SFT experiments
├── grpo/                   # frozen GRPO/DAPO experiments
├── eval/                   # full benchmark and SC@8 protocols
├── config.example.yaml     # complete configuration example
└── current.yaml            # local scratch experiment

src/math_post_training/
├── artifacts.py            # adapter and merged-checkpoint layout
├── checkpoint_eval.py      # sequential resumable checkpoint sweeps
├── cli.py                  # generate/eval/SFT/GRPO entry points
├── config.py               # YAML loading
├── curation.py             # rollout-dataset filtering
├── grpo.py                 # TRL GRPOTrainer and verifiable rewards
├── grpo_eval_loop.py       # segmented training + full external eval
├── model.py                # model/tokenizer loading and policy checks
├── rewards.py              # correctness and format rewards
├── rollouts.py             # resumable vLLM dataset rollouts
├── sft.py                  # supervised fine-tuning
├── data/                   # schemas, loaders, preprocessing, adapters
├── evaluation/             # runner, scoring, metrics, reporting
├── generation/             # Transformers and vLLM backends
├── prompts/                # explicit training/inference/eval protocols
└── verifiers/              # extraction, math equivalence, choices

scripts/                    # one-off but versioned audit/report utilities
docs/                       # experiment reports and normalized CSV/SVG artifacts
tests/                      # policy, data, verifier, trainer, and CLI contracts
```

## Validation before a long run

```bash
uv run ruff check .
uv run pytest
```

Then run 10–20 real GPU optimizer steps. Initialization and the first step are not representative.
Keep rollout count and effective completion batch fixed while tuning the microbatch split. Inspect
step time, GPU utilization, peak VRAM, reward standard deviation, zero-variance groups, entropy,
gradient norm, importance ratios, clipping, truncation, and NaN/Inf before committing to a long run.
