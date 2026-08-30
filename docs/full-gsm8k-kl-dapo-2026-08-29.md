# Full-GSM8K KL-DAPO — 2026-08-29

Этот отчёт фиксирует последний GSM8K RL-эксперимент и два финальных
self-consistency eval. Все значения accuracy приведены в процентах.

## Постановка

- стартовая модель: native-EOS SFT `Qwen2.5-1.5B`, обученная на 296k
  OpenMathInstruct-2;
- RL-данные: полный, нефильтрованный `openai/gsm8k` train;
- strict boxed reward: правильный `\\boxed{}` получает `+1`, неправильный `0`,
  отсутствующий или незавершённый box `-0.5`;
- DAPO `epsilon=0.2`, `epsilon_high=0.28`, `beta=1e-3`;
- LoRA `r32/a64`, 16 rollouts, effective completion batch 64;
- training microbatch / accumulation `8/8`, vLLM sleep выключен;
- LR `1e-5` cosine, 400 warmup steps, `max_completion_length=512`;
- полный greedy eval на GSM8K, GSM1K, MATH и MMLU-STEM каждые 500 шагов;
- запуск был рассчитан на 4000 шагов, но остановлен пользователем после
  полностью сохранённого checkpoint-3500.

PEFT reference log-probability вычислялась с отключённым LoRA adapter, поэтому
отдельная копия reference-модели в GPU-памяти не требовалась. В 20-шаговом
probe наблюдалось около 7.27 с/шаг, 99% GPU utilization и 58/81.9 GiB VRAM.

Конфиг:
`configs/grpo/qwen2_5_1_5b_openmath_gsm8k_full_dapo_strict_kl1e3_long.yaml`.

## Greedy checkpoint sweep

| Step | GSM8K | GSM1K | MATH | MMLU-STEM | mean(GSM8K,MATH) | Macro-4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 75.6634 | 69.8755 | 45.6400 | 41.9283 | 60.6517 | 58.2768 |
| 500 | 75.8908 | 72.6971 | 47.4000 | 42.2772 | 61.6454 | 59.5663 |
| 1000 | 78.0136 | **74.0249** | 47.8800 | 42.2772 | 62.9468 | 60.5489 |
| 1500 | 77.7862 | 72.6971 | 48.8000 | 42.5309 | 63.2931 | 60.4536 |
| 2000 | 78.0895 | 71.7012 | 49.2600 | 43.2287 | 63.6748 | 60.5698 |
| 2500 | **79.3783** | 71.7012 | **50.1000** | **43.4190** | **64.7392** | 61.1496 |
| 3000 | 78.9992 | 73.2780 | 49.8200 | 43.1652 | 64.4096 | **61.3156** |
| 3500 | 78.8476 | 72.7801 | 49.8800 | 43.3873 | 64.3638 | 61.2237 |

Для greedy-инференса checkpoint-2500 является лучшим по заранее выбранной
основной метрике `mean(GSM8K, MATH)`. Checkpoint-3000 максимизирует Macro-4,
но разница с 2500 составляет только `+0.1660` п.п.

Относительно SFT checkpoint-2500 даёт `+3.7149` п.п. GSM8K и `+4.4600` п.п.
MATH. Относительно лучшего full-GSM8K `beta=0` checkpoint по основной метрике
(`checkpoint-2500`, mean `64.4289`) KL-run улучшает mean на `+0.3103` п.п.
При этом лучший Macro-4 старого `beta=0` run (`61.3261` на шаге 1500)
практически совпадает с лучшим KL Macro-4 (`61.3156` на шаге 3000).

Вывод по greedy: KL позволил продолжать улучшение основной GSM8K+MATH метрики
до шага 2500, но не дал убедительного общего выигрыша на всех четырёх тестах.

## Self-consistency@8

Протокол: восемь независимых роллаутов, `temperature=0.8`, `top_p=0.95`,
`max_new_tokens=2048`; выбирается наиболее частый нормализованный финальный
ответ, при равенстве голосов — первый сгенерированный кандидат. Все восемь
ответов сохранены в predictions.

### Majority accuracy

| Checkpoint | GSM8K | GSM1K | MATH | MMLU-STEM | mean(GSM8K,MATH) | Macro-4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2500 greedy | 79.3783 | 71.7012 | 50.1000 | 43.4190 | 64.7392 | 61.1496 |
| 2500 SC@8 | 83.9272 | **80.0000** | **57.8000** | 49.0327 | 70.8636 | 67.6900 |
| 3000 greedy | 78.9992 | 73.2780 | 49.8200 | 43.1652 | 64.4096 | 61.3156 |
| 3000 SC@8 | **85.2919** | 78.3402 | 57.7800 | **49.3815** | **71.5359** | **67.6984** |

SC@8 меняет выбор checkpoint: по основной GSM8K+MATH метрике checkpoint-3000
лучше checkpoint-2500 на `+0.6723` п.п. Почти вся разница приходит из GSM8K
(`+1.3647` п.п.), тогда как MATH отличается лишь на `-0.0200` п.п. Macro-4
различается всего на `0.0084` п.п., то есть практически является ничьей.

Прирост majority относительно greedy:

| Checkpoint | GSM8K | GSM1K | MATH | MMLU-STEM |
| ---: | ---: | ---: | ---: | ---: |
| 2500 | +4.5489 | +8.2988 | +7.7000 | +5.6137 |
| 3000 | +6.2927 | +5.0622 | +7.9600 | +6.2163 |

### Rollout diagnostics

| Checkpoint / benchmark | Individual acc. | Observed pass@8 | Tie rate | Majority parse | Selected trunc. | Mean rollout tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2500 / GSM8K | 78.3359 | 92.7976 | 4.6247 | 100.0000 | 0 | 215.7 |
| 2500 / GSM1K | 73.3610 | 89.1286 | 8.2158 | 100.0000 | 0 | 254.6 |
| 2500 / MATH | 48.1725 | 72.4000 | 17.9000 | 99.9000 | 3 | 496.5 |
| 2500 / MMLU-STEM | 43.2644 | 72.1852 | 6.8189 | 95.6232 | 2 | 307.6 |
| 3000 / GSM8K | 79.3120 | 93.4799 | 4.7763 | 100.0000 | 0 | 221.2 |
| 3000 / GSM1K | 72.8320 | 89.2946 | 6.2241 | 100.0000 | 0 | 252.1 |
| 3000 / MATH | 48.3675 | 72.2200 | 17.6400 | 99.9000 | 3 | 496.4 |
| 3000 / MMLU-STEM | 43.0780 | 71.1386 | 7.0092 | 95.7818 | 5 | 309.6 |

В MATH отдельные роллауты достигали лимита в `4.53%` случаев для checkpoint-2500
и `4.66%` для checkpoint-3000. Для MMLU-STEM значения равны `3.04%` и `3.02%`.
После majority selection обрезанными остались только 3 MATH-ответа у каждого
checkpoint и 2/5 MMLU-STEM ответов. Значит текущий лимит 2048 достаточен для
SC@8 aggregation, хотя individual-rollout truncation нужно учитывать при
интерпретации pass@8.

## Решение

- один deterministic/greedy ответ: использовать checkpoint-2500;
- восемь sampled ответов с majority vote: использовать checkpoint-3000;
- если стоимость инференса важнее последних `0.67` п.п. GSM8K+MATH, checkpoint-2500
  остаётся почти эквивалентным SC@8 вариантом;
- выбирать следующие checkpoint по этим test-наборам нельзя без leakage: для
  будущего model selection нужен отдельный holdout из GSM8K train и независимый
  math validation set.

## Артефакты

- training/checkpoint repo:
  `NotHotTryHard/qwen2.5-1.5b-openmath-native-eos-gsm8k-full-dapo-strict-kl1e3-long`;
- checkpoint-2500 SC@8 dataset:
  `NotHotTryHard/qwen2.5-1.5b-openmath-kl-dapo-checkpoint2500-sc8-eval`;
- checkpoint-3000 SC@8 dataset:
  `NotHotTryHard/qwen2.5-1.5b-openmath-kl-dapo-checkpoint3000-sc8-eval`;
- W&B SC@8 runs: checkpoint-2500 `ryw7humf`, checkpoint-3000 `v9gyatff`;
- локальный CSV greedy-кривой: `docs/full-gsm8k-kl-dapo-2026-08-29.csv`.

В training repo проверены checkpoint-500/1000/1500/2000/2500/3000/3500:
каждый содержит adapter, optimizer, scheduler и trainer state. Оба SC@8 dataset
содержат `summary.json`, `predictions.jsonl.gz`, `sweep-summary.json`, точный
конфиг и полный eval log. Tokenizer использует EOS `<|endoftext|>`, файлов
`chat_template` нет.
