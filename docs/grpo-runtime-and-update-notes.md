# GRPO/DAPO runtime and update notes

This note records the operational settings and update semantics used by the
single-A100 strict DAPO run. Keep the effective rollout batch fixed when tuning
the microbatch split: changing both at once makes throughput and learning
comparisons uninterpretable.

## A100 colocate throughput baseline

Measured on one A100 80 GB with a 1.5B policy, LoRA `r32/a64`, 16 generations per
prompt, generation/effective completion batch 64 and `max_completion_length=512`.

| Training microbatch | Accumulation | vLLM sleep | Mean step time | Relative speed |
| ---: | ---: | :---: | ---: | ---: |
| 1 | 64 | on | 23.13 s | 1.00x |
| 4 | 16 | on | 9.49 s | 2.44x |
| 8 | 8 | on | 8.38 s | 2.76x |
| 16 | 4 | on | 8.24 s | 2.81x |
| 8 | 8 | off | **6.53 s** | **3.54x** |

Use `per_device_train_batch_size=8`, `gradient_accumulation_steps=8` and
`vllm_enable_sleep_mode=false` as the starting point for this exact model/GPU
combination. The full run confirmed roughly 6.55 seconds per step over 50 steps,
98% point-in-time GPU utilization and about 58.5/81.9 GiB allocated. Microbatch 16
saved almost no time and left less OOM headroom.

These values are not universal defaults. Re-benchmark after changing model size,
LoRA targets, completion length, number of generations, vLLM version, tensor
parallelism or GPU. Pick the largest training microbatch before throughput
plateaus while retaining enough memory for unusually long rollout batches.

Sleep mode is useful when the training and inference copies cannot remain resident
together. When they fit, repeatedly sleeping and waking vLLM adds weight-transfer
and allocator synchronization overhead. Disabling it does not change the RL
objective; changing the effective batch or number of rollouts does.

## One strict DAPO optimizer update

For the current configuration one optimizer update processes four distinct math
prompts, with 16 sampled completions for each prompt: 64 completion sequences.

1. The latest policy weights are made available to colocated vLLM. vLLM samples
   all 64 completions with temperature 0.8 and top-p 0.95, stopping at EOS or 512
   new tokens.
2. `strict_boxed_reward` extracts only the final complete `\\boxed{...}`. A
   verified correct box receives `+1`, a wrong box receives `0`, and a missing or
   incomplete box receives `-0.5`. There is no separate format bonus and no
   last-number fallback.
3. Rewards are grouped by original prompt. For each group of 16, TRL computes
   `A_i = (r_i - mean(group)) / std(group)` because `scale_rewards=group`.
   The advantage is sequence-level and is reused for every generated token in
   that completion. If all 16 rewards are equal, the group has no comparative
   learning signal and contributes zero advantage.
4. The trainer evaluates the sampled tokens under the trainable policy and forms
   the token importance ratio `rho_t = exp(log pi_theta - log pi_old)`. DAPO uses
   the clipped surrogate with the asymmetric interval `[0.8, 1.28]`. Clipping
   prevents one optimizer update from increasing or decreasing the probability
   of sampled tokens too aggressively. `beta=0` means there is currently no
   reference-policy KL penalty.
5. Completions cut off at 512 tokens are masked from the policy loss. For the
   remaining completions, positive-advantage tokens are made more likely and
   negative-advantage tokens less likely. `loss_type=dapo` sums token losses and
   normalizes by the number of active tokens across the globally accumulated
   batch, instead of averaging every response independently.
6. The 64 sequences are trained as eight microbatches of eight. Backpropagation
   accumulates their gradients without updating between microbatches. After the
   eighth microbatch, AdamW updates only the LoRA parameters, the cosine scheduler
   advances once, gradients are cleared, and the updated policy is used for the
   next rollout group.

There is no learned critic or value network: the within-prompt mean reward is the
baseline. RL does not differentiate through sampling or through the verifier; the
reward and advantage are fixed labels for the sampled token trajectories.

## Checks before a long run

- Benchmark at least 10--20 real steps; initialization and the first step are not
  representative.
- Keep generation batch, number of generations, seed and effective completion
  batch identical across throughput probes.
- Watch both utilization and step time. Colocated rollout/training phases make a
  single utilization snapshot misleading.
- Watch peak memory, clipped-completion ratio, reward zero-std fraction, importance
  ratios, gradient norm and NaN/Inf—not utilization alone.
- Preserve aborted probe logs and start the chosen full experiment clean unless a
  real checkpoint resume is intended.

The production recipe is
`configs/grpo/qwen2_5_1_5b_openmath_gsm8k_nontrivial_dapo_strict.yaml`.
