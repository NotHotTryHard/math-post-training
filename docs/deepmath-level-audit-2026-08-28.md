# DeepMath level audit: native-EOS SFT vs Qwen2.5-Math-1.5B-Instruct

Date: 2026-08-28

## Goal

Estimate which DeepMath-103K difficulty levels contain useful on-policy RL signal for the
native-EOS SFT checkpoint, and test whether Qwen2.5-Math-1.5B-Instruct is a safe upper
filter.

## Protocol

- Dataset: `zwhe99/DeepMath-103K`, pinned revision
  `5cf055d1fe3d7a2eb19719ac020211469736ae44`.
- Deterministic sample: 500 unique tasks at each exact level from 5.0 through 9.0 in
  increments of 0.5; 4,500 tasks total and identical stable IDs for both models. Sampling
  used a streaming shuffle with seed 42 and a 10,000-row buffer, so it is deterministic
  but only approximately globally random if the source parquet is ordered.
- Three sampled rollouts per task, temperature 0.8, top-p 0.95, seed 42.
- vLLM batch size 768, model context 4,096, maximum completion 3,584 tokens.
- SFT model: `NotHotTryHard/qwen2.5-1.5b-base-openmath-296k-1epoch-lora-cosine-native-eos-merged`
  with the native math-post-training prompt.
- Teacher model: `Qwen/Qwen2.5-Math-1.5B-Instruct` with its own published chat prompt.
- Total: 27,000 generated completions. Qwen-Math generation took 720 seconds; SFT took
  653 seconds, excluding model initialization.

The raw output exposed a verifier bug for DeepMath multiple-choice references stored as a
plain letter such as `A`: an extracted `A` was incorrectly rejected. The report uses the
corrected exact-choice score. The fix and regression cases are included with the audit
scripts. The correction moved aggregate SFT accuracy from 29.41% to 29.64%, and Qwen-Math
from 52.41% to 53.50%.

## Dataset population

The complete 103,110-row split contains 88,719 tasks at the nine audited levels:

| Level | Full dataset count |
| ---: | ---: |
| 5.0 | 16,337 |
| 5.5 | 14,806 |
| 6.0 | 17,488 |
| 6.5 | 7,750 |
| 7.0 | 8,561 |
| 7.5 | 6,746 |
| 8.0 | 11,686 |
| 8.5 | 3,989 |
| 9.0 | 1,356 |

## Results by level

`Mixed` means the SFT produced both correct and incorrect answers (`1/3` or `2/3`), which
is the directly useful group-relative RL pool. `Teacher frontier` means SFT scored `0/3`
while Qwen-Math scored at least `1/3`.

| Level | SFT rollout acc. | SFT 0/3 | SFT mixed | SFT 3/3 | Qwen-Math acc. | Teacher frontier | Both 0/3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5.0 | 18.27% | 330 (66.0%) | 140 (28.0%) | 30 (6.0%) | 54.27% | 195 (39.0%) | 135 (27.0%) |
| 5.5 | 21.53% | 308 (61.6%) | 157 (31.4%) | 35 (7.0%) | 54.73% | 181 (36.2%) | 127 (25.4%) |
| 6.0 | 22.13% | 308 (61.6%) | 156 (31.2%) | 36 (7.2%) | 51.00% | 155 (31.0%) | 153 (30.6%) |
| 6.5 | 21.73% | 299 (59.8%) | 167 (33.4%) | 34 (6.8%) | 49.67% | 157 (31.4%) | 142 (28.4%) |
| 7.0 | 26.07% | 280 (56.0%) | 177 (35.4%) | 43 (8.6%) | 51.07% | 130 (26.0%) | 150 (30.0%) |
| 7.5 | 30.80% | 253 (50.6%) | 175 (35.0%) | 72 (14.4%) | 50.13% | 108 (21.6%) | 145 (29.0%) |
| 8.0 | 35.87% | 206 (41.2%) | 223 (44.6%) | 71 (14.2%) | 53.87% | 92 (18.4%) | 114 (22.8%) |
| 8.5 | 44.33% | 147 (29.4%) | 269 (53.8%) | 84 (16.8%) | 57.13% | 60 (12.0%) | 87 (17.4%) |
| 9.0 | 46.07% | 135 (27.0%) | 279 (55.8%) | 86 (17.2%) | 59.67% | 64 (12.8%) | 71 (14.2%) |

Aggregate over the balanced audit sample:

- SFT: 29.64% rollout accuracy; 49.64% pass@3; 38.73% mixed groups; 10.91% `3/3`.
- Qwen-Math: 53.50% rollout accuracy; 68.69% pass@3; 30.71% mixed groups; 37.98% `3/3`.
- SFT completion diagnostics: 88.57% parse rate, 97.57% boxed-format rate, 2.37%
  truncation, 613 mean completion tokens.
- Qwen-Math diagnostics: 91.15% parse rate, 99.42% boxed-format rate, 0.39%
  truncation, 731 mean completion tokens.

## Paired model disagreement

Across the same 4,500 tasks:

- both models `0/3`: 1,124 tasks (24.98%);
- SFT `0/3`, Qwen-Math positive: 1,142 tasks (25.38%);
- SFT positive, Qwen-Math `0/3`: 285 tasks (6.33%);
- both models positive: 1,949 tasks (43.31%).

Therefore Qwen-Math must not be used as a destructive upper gate: it would reject 285
sampled tasks that the actual starting policy can solve. It is useful as a label for the
SFT frontier, not as the source of truth for deletion.

## Topic diagnostics

| Topic family | Tasks | SFT acc. | SFT mixed | Qwen-Math acc. |
| --- | ---: | ---: | ---: | ---: |
| Algebra | 1,502 | 31.56% | 41.21% | 51.71% |
| Applied Mathematics | 260 | 26.54% | 32.69% | 50.51% |
| Calculus | 1,294 | 26.87% | 35.78% | 56.00% |
| Differential Equations | 74 | 27.03% | 37.84% | 42.79% |
| Discrete Mathematics | 262 | 31.42% | 40.08% | 49.87% |
| Geometry | 357 | 30.44% | 42.58% | 48.18% |
| Number Theory | 216 | 22.99% | 35.19% | 47.07% |
| Other | 257 | 41.89% | 45.91% | 61.61% |
| Precalculus | 278 | 26.98% | 34.89% | 64.99% |

## Decision

There is no defensible numeric difficulty cutoff. In this sample, higher metadata levels
are not harder for either model; SFT mixed-group yield rises from 28.0% at level 5 to
55.8% at level 9. The metadata remains useful for stratification, but task-level on-policy
outcomes are the correct selection signal. The surprising inverse trend should not be
read as proof that level 9 is intrinsically easier: model-relative skill, topic/source
composition, metadata noise, and approximate streaming sampling can all contribute.

Recommended first DeepMath RL pool:

1. Select SFT `1/3` and `2/3` tasks across every level 5-9.
2. Preserve the original level distribution and topic balance rather than taking equal
   counts per level for training.
3. Give the `0/3` and `3/3` groups five additional rollouts before a durable bucket
   assignment. Three trials are enough for a pilot, but too noisy for a final hard filter.
4. Keep SFT `0/8` + Qwen-Math-positive tasks as a teacher-solvable frontier for later
   curriculum, distillation, or off-policy methods. Do not place them in round-0 vanilla
   group-relative RL because an all-zero group supplies no advantage signal.
5. Keep `8/8` tasks outside the core optimization pool. Sampling them does not prevent
   forgetting when every rollout has the same reward; use KL regularization or an explicit
   supervised/replay loss for retention instead.

Weighting the per-level pilot rates by the full dataset population suggests roughly
30,800 directly useful mixed tasks among levels 5-9, before the recommended five-rollout
confirmation. This is already more than enough for the next experiment; there is no need
to train on all 103K tasks or choose a crude level boundary.
