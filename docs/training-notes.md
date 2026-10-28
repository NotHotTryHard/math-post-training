# Training notes

[Back to the README](../README.md)

These notes collect the SFT, RL, data, and runtime experiments behind the final recipe.
The linked dated reports contain the original measurements.

## Prompt format, reward, and validation

### 1. Match the policy format to the Base model

The project starts from [`Qwen/Qwen2.5-1.5B`](https://huggingface.co/Qwen/Qwen2.5-1.5B), not an
Instruct checkpoint. Teaching it a complete ChatML interaction contract was not just a formatting
change: the model had to learn new control tokens, turn boundaries, and stopping behavior. The
98K-example LoRA run did not learn that contract reliably. Limited data and adapter capacity were
possible explanations; these experiments do not establish that ChatML requires full fine-tuning.

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

The final run used `beta=1e-3` to constrain drift from the SFT policy. The best full-GSM8K
beta-zero checkpoint by mean(GSM8K, MATH) reached `77.94 / 50.92`; the KL checkpoint reached
`79.38 / 50.10`. The joint score improved by about 0.31 percentage points, with a GSM8K gain
and a MATH decrease. This supports choosing the KL checkpoint for the joint metric, but does not
isolate KL as the cause: the schedule also changed and these are single-seed runs.

With TRL + PEFT, reference log-probabilities were obtained by temporarily disabling the LoRA
adapter, avoiding a second resident reference model. See the
[full KL report](full-gsm8k-kl-dapo-2026-08-29.md) for the sweep and SC@8 results.

## Ablation-led development

### SFT: LoRA scaling and effective batch

This controlled 98K-example ablation predates the native-EOS correction, so its absolute scores are
not final-model scores. It is still the clean comparison used to choose the LoRA geometry.

| SFT setting | GSM8K | MATH | Observation |
| --- | ---: | ---: | --- |
| `r32 / a64 / batch 64` | **69.60** | **33.74** | selected |
| `r32 / a128 / batch 64` | 68.69 | 32.88 | larger alpha hurt all four benchmarks |
| `r32 / a64 / batch 32` | 51.78 | 19.50 | unstable long completions and truncation |

The winner remained ordinary LoRA `r32/a64`, not rsLoRA and not the larger `a128` used by an
adjacent public GSM8K recipe.

### RL: reward, group batch, data, and KL

| RL variant | GSM8K | MATH | Main lesson |
| --- | ---: | ---: | --- |
| native-EOS SFT | 75.66 | 45.64 | starting policy |
| permissive DAPO, 32 completions/update | 74.53 | 46.54 | extra noisy updates did not help |
| permissive DAPO, 64 completions/update | 75.82 | 46.82 | lower gradient noise was better |
| strict DAPO, rollout-filtered GSM8K | 77.26 | 49.84 | strict verifiable reward mattered |
| strict DAPO, full GSM8K, `beta=0`, step 2500 | 77.94 | **50.92** | full data retained coverage; best beta-0 primary checkpoint |
| strict DAPO, full GSM8K, `beta=1e-3`, step 2500 | **79.38** | 50.10 | best greedy GSM8K+MATH mean |

Keeping easy `16/16` groups in the dataset does not itself prevent forgetting: a zero-variance
group has zero group-relative advantage. Full GSM8K was nevertheless worth testing because policy
drift can turn an easy task into a mixed group later, and because filtering is a model-relative,
noisy decision. These experiments did not isolate which factors preserved previously solved tasks; monitoring
measures changes but does not itself prevent forgetting.

### RL data: rollout filtering helped; DeepMath mixing did not

Before RL, the SFT policy generated multiple answers for each training problem. This made it
possible to separate tasks it solved consistently from tasks that still produced both successes
and failures. Training first on the rollout-filtered GSM8K subset concentrated updates on problems
with useful reward variation and gave a clear improvement over the SFT checkpoint. The later
full-GSM8K experiment showed the trade-off: filtering improves signal efficiency, while retaining
the full dataset preserves broader coverage as the policy changes.

DeepMath was the natural next candidate: it is newer, larger, and substantially more varied than
GSM8K, so I expected it to improve general mathematical reasoning. Before training, a stratified
rollout audit across its difficulty levels confirmed that the problems were not simply out of
reach—both the native-EOS SFT policy and Qwen2.5-Math-1.5B-Instruct could solve meaningful portions
of the sample.

I then selected 1,743 DeepMath problems on which the SFT policy produced mixed outcomes and trained
on a `50/50` mixture with full GSM8K. The checkpoint sweep showed no convincing advantage over the
GSM8K-only runs, so the experiment was stopped after step 1400. The useful negative result was
simple: adding a newer and harder dataset did not automatically improve transfer; in this setup,
cleaner reward, full GSM8K coverage, and KL regularization mattered more than broader data.

The sampling audit and selection details remain in
[`docs/deepmath-level-audit-2026-08-28.md`](deepmath-level-audit-2026-08-28.md).

### Systems: 3.54× faster without changing the objective

On one A100 80 GB, the same 64-completion optimizer update was benchmarked under several training
microbatch splits:

| Microbatch / accumulation | vLLM sleep | Mean step | Speedup |
| --- | :---: | ---: | ---: |
| `1 / 64` | on | 23.13 s | 1.00× |
| `4 / 16` | on | 9.49 s | 2.44× |
| `8 / 8` | on | 8.38 s | 2.76× |
| `16 / 4` | on | 8.24 s | 2.81× |
| `8 / 8` | off | **6.53 s** | **3.54×** |

The selected setup kept the training and colocated vLLM copies resident, reached mostly 80–100%
GPU utilization, and reduced a projected ~19-hour strict run to roughly 5.4 hours. The effective
batch, number of rollouts, reward, and number of optimizer updates were unchanged.


## Compute accounting

The final SFT run took about 7.5 hours, and the KL training run about 6.9 hours, excluding evaluation.
The earlier README listed 124.4 billed A100 hours and a per-stage allocation of 16 hours for
evaluation, 45 for SFT, 11 for filtering, and 52.4 for RL. That allocation was manually estimated,
not measured per stage; it should not be used as a runtime benchmark. The 124.4-hour figure was
itself inferred from billing, not read from a billed-hours export. A precise breakdown requires
reconciling pod billing records with run timestamps. W&B process runtime and billed pod uptime
measure different things.

## References and related work

- [Qwen2.5-Math Technical Report](https://arxiv.org/abs/2409.12122) — Qwen's SFT, reward-model,
  GRPO, and inference-scaling pipeline.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — introduces GRPO for mathematical reasoning.
- [DAPO](https://arxiv.org/abs/2503.14476) — decoupled clipping, dynamic sampling, token-level loss,
  and overlong-reward shaping at scale.
- [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) — the SFT source.
- [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) — the broader mathematics
  dataset evaluated in the mixed RL experiment.
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) — the supervised
  fine-tuning interface and dataset-format reference.
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/index) and [vLLM](https://vllm.ai/) — the training
  and generation stack used here.
- [DrEternity/gsm8k-post-training](https://github.com/DrEternity/gsm8k-post-training) — an adjacent
  open GSM8K SFT/GRPO project and a useful point of comparison for experiment structure.
- [codex-cli](https://github.com/openai/codex) — the terminal engineering assistant used
  during development.
