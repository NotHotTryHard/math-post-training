# Strict DAPO and DeepMath pilot — 2026-08-24

This note freezes the intermediate results before selecting the next DeepMath
training slice. Percentages are absolute accuracies unless stated otherwise.

## Strict GSM8K DAPO

The run starts from the native-EOS SFT checkpoint and uses 3 passes over 3,868
rollout-filtered GSM8K train problems, 16 completions per prompt, strict boxed
reward, LoRA `r32/a64`, effective completion batch 64, LR `1e-5` cosine and no
reference-policy KL penalty.

Training diagnostics over all 2,901 optimizer steps:

| Metric | Value |
| --- | ---: |
| Sum of measured step times | 19,467.1 s (5 h 24 m 27 s) |
| Mean step time | 6.710 s |
| Mean strict reward | 0.7621 |
| Mean within-group reward std | 0.3921 |
| Mean zero-std prompt fraction | 45.10% |
| Mean completion length | 194.1 tokens |
| Mean clipped-completion fraction | 2.66% |
| Mean gradient norm | 0.1405 |

### Full greedy eval

| Model | GSM8K | GSM1K | MATH | MMLU-STEM |
| --- | ---: | ---: | ---: | ---: |
| Native-EOS SFT | 75.6634% | 69.8755% | 45.6400% | 41.9283% |
| Strict DAPO | **77.2555%** | **72.1992%** | **49.8400%** | **44.5925%** |
| Delta | +1.5921 pp | +2.3237 pp | +4.2000 pp | +2.6642 pp |

Greedy generation diagnostics are `parse rate / truncated / mean completion tokens`:

| Benchmark | Diagnostics |
| --- | ---: |
| GSM8K | 96.74% / 43 / 226.5 |
| GSM1K | 97.59% / 30 / 248.3 |
| MATH | 89.94% / 485 / 541.9 |
| MMLU-STEM | 82.21% / 248 / 354.6 |

### Sampled majority vote at 8

Protocol: 8 independent samples, temperature `0.8`, top-p `0.95`, maximum
2,048 new tokens; the most frequent parsed answer wins.

| Benchmark | Majority@8 | Individual rollout | Observed pass@8 | Tie rate |
| --- | ---: | ---: | ---: | ---: |
| GSM8K | 83.0174% | 78.1653% | 91.4329% | 5.61% |
| GSM1K | 79.5021% | 73.1535% | 88.9627% | 6.56% |
| MATH | 58.4000% | 48.6875% | 72.7400% | 18.82% |
| MMLU-STEM | 49.3498% | 42.8758% | 71.0435% | 7.04% |

All eight completions remain in the RunPod prediction artifacts. Majority@8
improves over strict-DAPO greedy by `+5.76/+7.30/+8.56/+4.76` percentage points.

## DeepMath SFT-policy pilot

The full 103K pass was intentionally stopped after a complete, valid prefix of
14,592 examples (`index=0..14591`). The partial artifact is resumable but must
not be treated as a random sample of the complete dataset.

Protocol: native-EOS SFT policy, 3 rollouts per task, temperature `0.8`, top-p
`0.95`, maximum 4,096 new tokens and model length 8,192.

| Metric | Value |
| --- | ---: |
| Prompts / rollouts | 14,592 / 43,776 |
| Rollout accuracy | 25.1690% |
| Parse rate | 90.8923% |
| Boxed-format rate | 97.1948% |
| Truncation rate | 2.7526% |
| Mean completion length | 673.2 tokens |
| Pass histogram `0/1/2/3` | 8,486 / 2,618 / 2,064 / 1,424 |
| Mixed groups (`1/3` or `2/3`) | 4,682 (32.0861%) |

### Results by reported difficulty

The dataset paper defines larger scores as harder. The policy-relative pilot
shows the opposite monotonic trend, so the metadata score is not a safe proxy
for this model without a stratified audit.

| Difficulty | Tasks | Rollout accuracy | 0/3 | Mixed | 3/3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5.0 | 1,765 | 17.47% | 69.07% | 25.16% | 5.78% |
| 5.5 | 2,031 | 18.86% | 66.62% | 27.23% | 6.15% |
| 6.0 | 2,446 | 20.99% | 64.47% | 27.06% | 8.46% |
| 6.5 | 1,151 | 23.66% | 59.25% | 32.15% | 8.60% |
| 7.0 | 1,285 | 25.89% | 57.12% | 32.37% | 10.51% |
| 7.5 | 1,100 | 29.88% | 50.82% | 37.36% | 11.82% |
| 8.0 | 1,992 | 36.61% | 43.22% | 40.71% | 16.06% |
| 8.5 | 803 | 43.50% | 31.26% | 51.18% | 17.56% |
| 9.0 | 325 | 48.92% | 19.38% | 65.85% | 14.77% |

The inversion may come from source/topic composition, answer/verifier effects,
or ordering bias in the contiguous prefix. It is too strong to silently assume
that level 9 is harder for this policy than level 5.

## Decision for the next slice

Do not spend several more GPU-hours evaluating every DeepMath problem before
training. First draw a deterministic stratified audit of 512 tasks per half-level
from 5.0 through 9.0, balanced across top-level topics where possible. This is
4,608 prompts and 13,824 rollouts, approximately 12 minutes at the measured pilot
throughput.

If the audit reproduces the curve, the first metadata-only RL slice should use
levels `6.5..9.0`. In the current pilot this range has at least 32% mixed groups
at every level and would project to roughly 45–50K tasks in the full dataset.
Levels `5.0..6.0` should remain available for later sampling rather than being
deleted. A single upper cutoff such as `difficulty <= 7` is not supported by the
observed data.

The 103K source contains mostly levels 5–9, explicitly intended as difficult RL
data, but its ratings are GPT-4o/AoPS estimates rather than measurements against
our policy. The final sampler should therefore keep `difficulty`, topic and
policy-relative success as separate signals.

## Durable artifacts

- strict merged model: `NotHotTryHard/qwen2.5-1.5b-openmath-native-eos-gsm8k-nontrivial-dapo-strict-merged`;
- strict training repo: `NotHotTryHard/qwen2.5-1.5b-openmath-native-eos-gsm8k-nontrivial-dapo-strict`;
  all checkpoints `250,500,...,2750,2901` are preserved with adapter, optimizer,
  scheduler, RNG and trainer state;
- permissive bs32 training repo: `NotHotTryHard/qwen2.5-1.5b-openmath-native-eos-gsm8k-nontrivial-dapo-bs32`;
  all checkpoints `100,200,...,900,967` are preserved;
- permissive bs64 training repo: `NotHotTryHard/qwen2.5-1.5b-openmath-native-eos-gsm8k-nontrivial-dapo`;
  all checkpoints `100,200,300,400,484` are preserved;
- strict training W&B: `db868g02`;
- private artifact dataset: `NotHotTryHard/strict-dapo-gsm8k-2026-08-24-artifacts`;
  it contains strict greedy and majority@8 predictions/summaries, both permissive
  evals, the GSM8K 8x rollout-filtering data, the 14,592-task DeepMath pilot,
  all live-pod configs and all experiment logs;
- source revision: `zwhe99/DeepMath-103K@5cf055d1fe3d7a2eb19719ac020211469736ae44`.

The remote repositories were checked after upload: all are private, every
expected checkpoint contains `adapter_model.safetensors`, `optimizer.pt`,
`scheduler.pt`, `rng_state.pth` and `trainer_state.json`, and the artifact
dataset contains 87 files (95,660,308 bytes). The disposable RunPod volume is
not required to recover these experiments.
