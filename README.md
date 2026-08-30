<div align="center">

# Math Post-Training on a Qwen-2.5-1.5B-Base Model

**A reproducible BASE -> SFT -> DAPO Qwen post-training cycle with GSM8K as the primary target and MATH as the harder generalization benchmark. Compared with math-specialized [`Qwen2.5-Math`](https://arxiv.org/abs/2409.12122) checkpoint as a stronger-model reference for capability estimation.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TRL](https://img.shields.io/badge/TRL-GRPO-FFD21E)](https://huggingface.co/docs/trl/index)
[![vLLM](https://img.shields.io/badge/vLLM-0.25.1-4B8BBE)](https://vllm.ai/)
[![Docker](https://img.shields.io/badge/Docker-reproducible-2496ED?logo=docker&logoColor=white)](https://github.com/NotHotTryHard/math-post-training/pkgs/container/math-post-training)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Starting from **`Qwen/Qwen2.5-1.5B-base`**, the final pipeline reaches
**79.38% GSM8K / 50.10% MATH with greedy decoding** and
**85.29% GSM8K / 57.78% MATH with self-consistency@8** versus
**84.80% GSM8K / 75.81% MATH with greedy decoding of `Qwen/Qwen2.5-Math-1.5B-Instruct`**
.

[Results](#results) · [What mattered](#what-actually-mattered) · [Ablations](#ablation-led-development) · [Reproduce](#reproduce-it) · [Detailed docs](docs/reproducibility.md)

</div>

---

## TL;DR

| Stage | GSM8K | MATH | What changed |
| --- | ---: | ---: | --- |
| Local Base baseline | 64.14 | 32.08 | Qwen Base few-shot CoT protocol |
| Native-EOS SFT | 75.66 | 45.64 | 296K OpenMathInstruct-2 examples, plain policy format |
| Full-GSM8K KL-DAPO, greedy | **79.38** | **50.10** | strict verifiable reward, full GSM8K, `beta=1e-3` |
| Full-GSM8K KL-DAPO, SC@8 | **85.29** | **57.78** | checkpoint 3000, majority vote over 8 samples |

The result was not produced by copying a single recipe. The useful configuration emerged from
ablations over prompt/EOS format, SFT data scale, LoRA scaling, effective batch size, reward
strictness, rollout group size, filtered versus full GSM8K, KL regularization, DeepMath selection,
and GPU scheduling.

The most important early correction was to drop the attempts to teach a Base model answeing in ChatML format (we need much more data + full finetune, not the LoRA). Instead I replaced the inherited ChatML-style policy interface with one plain prompt and the
native Base tokenizer EOS, `<|endoftext|>`, which fixed the training/inference contract before more compute
was spent on SFT or RL.

## Results

![GSM8K accuracy across the Qwen2.5-1.5B post-training pipeline](docs/readme-gsm8k-pipeline.svg)

All project results below were measured locally with the repository evaluator. Greedy runs use `max_new_tokens=2048`. SC@8 uses 8 samples at `temperature=0.8`, `top_p=0.95`, then selects the most frequent normalized final answer.

| Model / checkpoint | Decoding | GSM8K | GSM1K | MATH | MMLU-STEM |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-1.5B Base, local baseline | greedy | 64.14 | 58.17 | 32.08 | 42.31 |
| ChatML SFT, 98K | greedy | 69.60 | 64.23 | 33.74 | **52.17** |
| Native-EOS SFT, 296K | greedy | 75.66 | 69.88 | 45.64 | 41.93 |
| Strict filtered-GSM8K DAPO | greedy | 77.26 | 72.20 | 49.84 | **44.59** |
| Full-GSM8K KL-DAPO, step 2500 | greedy | **79.38** | 71.70 | **50.10** | 43.42 |
| Strict filtered-GSM8K DAPO | SC@8 | 83.02 | **79.50** | **58.40** | 49.35 |
| Full-GSM8K KL-DAPO, step 3000 | SC@8 | **85.29** | 78.34 | 57.78 | **49.38** |

**MATH was the better canary.** Earlier permissive DAPO barely moved GSM8K while MATH still grew. A GSM8K plateau therefore did not imply that RL had stopped improving mathematical reasoning.

### Full-GSM8K KL-DAPO checkpoint sweep

![Greedy validation during full-GSM8K KL-DAPO](docs/readme-kl-dapo-checkpoint-sweep.svg)

GSM8K is noisy after step 1000; MATH rises much more smoothly until step 2500. This is exactly why
the integrated sweep evaluates more than the training-domain benchmark.

> [!NOTE]
> These checkpoint evaluations are retrospective test analysis. Best model selection used
> a held-out GSM8K development split and an independent math validation set to avoid test leakage.

## What actually mattered

### 1. Match the policy format to the Base model

The project starts from [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B), not an
Instruct checkpoint. Teaching it a complete ChatML interaction contract was not just a formatting
change: the model had to learn new control tokens, turn boundaries, and stopping behavior. The
98K-example SFT run did not provide enough data, and LoRA did not provide the degrees of freedom of
a full fine-tune, to learn that contract reliably.

The final pipeline instead uses one plain prompt from SFT through RL and inference, and terminates
every training completion with the Base tokenizer's native `<|endoftext|>`. This removed the
avoidable interface-learning problem and made stopping behavior testable. The `69.60 -> 75.66`
GSM8K jump is not a pure format ablation because the native-EOS run also used more data, but the
format and stopping failures disappeared immediately.

### 2. MATH exposed failures that GSM8K hid

GSM8K is short and forgiving; MATH is more sensitive to broken stopping, extraction, and
truncation. During native-EOS SFT, MATH rose from `42.66` to `45.64`, parse rate improved from
`83.76%` to `88.62%`, and truncations fell from `796` to `547`. It remained the useful canary when
GSM8K was noisy or temporarily flat.

### 3. Strict reward beat a permissive proxy

The first pilots used `accuracy + 0.1 × boxed_format` with a last-number fallback. That proxy
could improve while answer format and stopping degraded. The final reward has one unambiguous
contract:

| Completion | Reward |
| --- | ---: |
| correct final `\boxed{...}` | `+1.0` |
| wrong final `\boxed{...}` | `0.0` |
| missing or incomplete box | `-0.5` |

No separate format bonus or fallback is used. Sixteen rollouts provide the within-problem contrast;
DAPO applies token-level loss normalization and asymmetric clipping `[0.8, 1.28]`.

### 4. KL regularization was important 

The decisive final change was `beta=1e-3`. It deliberately slowed the immediate policy update, but
also constrained drift from the SFT policy, allowing useful training to continue for longer. The
best full-GSM8K `beta=0` checkpoint reached `77.94` GSM8K / `49.22` MATH; the KL run continued to
`79.38` / `50.10` at step 2500 and produced the best primary `mean(GSM8K, MATH)` score.

The cool detail is that with `TRL+PEFT`, reference log-probabilities were obtained by temporarily
disabling the LoRA adapter, so this did not require a second resident copy of the 1.5B model.
