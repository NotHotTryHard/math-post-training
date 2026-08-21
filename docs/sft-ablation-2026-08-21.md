# Промежуточные результаты SFT ablation — 2026-08-21

## Протокол

- модель: `Qwen/Qwen2.5-1.5B`;
- датасет: `nvidia/OpenMathInstruct-2`, `train_1M`, revision
  `469216e3f46f4dacf476b382e192485ea51a143e`;
- обучение: 97 984 train + 2 048 validation примеров, одна эпоха;
- LoRA: `r=32`, dropout `0.05`, `all-linear`, BF16/TF32, max length 4096;
- внешний eval: полный GSM8K, GSM1k, MATH и MMLU-STEM, published Qwen2.5-Math Base
  few-shot CoT protocol, greedy decoding, `max_new_tokens=2048`, vLLM batch size 768;
- все accuracy ниже получены на одинаковых full splits. Только отдельный rsLoRA diagnostic
  использует seeded `100` примеров на benchmark и поэтому вынесен в отдельную таблицу.

## Full benchmark results

| Вариант | GSM8K | GSM1k | MATH | MMLU-STEM |
| --- | ---: | ---: | ---: | ---: |
| Base до SFT | 846/1319 (64.14%) | 701/1205 (58.17%) | 1604/5000 (32.08%) | 1334/3153 (42.31%) |
| `r32/a64/bs64` | **918/1319 (69.60%)** | **774/1205 (64.23%)** | **1687/5000 (33.74%)** | **1645/3153 (52.17%)** |
| `r32/a128/bs64` | 906/1319 (68.69%) | 753/1205 (62.49%) | 1644/5000 (32.88%) | 1590/3153 (50.43%) |
| `r32/a64/bs32` | 683/1319 (51.78%) | 582/1205 (48.30%) | 975/5000 (19.50%) | 1206/3153 (38.25%) |

Дельта относительно Base до SFT:

| Вариант | GSM8K | GSM1k | MATH | MMLU-STEM |
| --- | ---: | ---: | ---: | ---: |
| `r32/a64/bs64` | **+5.46 п.п.** | **+6.06 п.п.** | **+1.66 п.п.** | **+9.86 п.п.** |
| `r32/a128/bs64` | +4.55 п.п. | +4.32 п.п. | +0.80 п.п. | +8.12 п.п. |
| `r32/a64/bs32` | -12.36 п.п. | -9.87 п.п. | -12.58 п.п. | -4.06 п.п. |

Дельта абляций относительно текущего лучшего `r32/a64/bs64`:

| Вариант | GSM8K | GSM1k | MATH | MMLU-STEM |
| --- | ---: | ---: | ---: | ---: |
| `r32/a128/bs64` | -0.91 п.п. | -1.74 п.п. | -0.86 п.п. | -1.74 п.п. |
| `r32/a64/bs32` | -17.82 п.п. | -15.93 п.п. | -14.24 п.п. | -13.92 п.п. |

## Validation и время обучения

| Вариант | Steps | Лучший validation loss | Шаг лучшего checkpoint | Время до последнего train log |
| --- | ---: | ---: | ---: | ---: |
| `r32/a64/bs64` | 1531 | **0.325487** | 1531 | 2:19:34 |
| `r32/a128/bs64` | 1531 | 0.328967 | 1531 | 2:20:00 |
| `r32/a64/bs32` | 3062 | 0.328646 | 3062 | 2:25:41 |
| rsLoRA `r32/a64/bs64` | 1531 | 0.383637 | 1500 | 2:30:29 |

Одна внутренняя validation на 2 048 примерах занимала примерно 79–80 секунд. Маленькая разница
между validation loss у `a128/bs64` и `a64/bs32` не предсказывает огромную разницу внешнего
eval: у `bs32` сильно ломается поведение длинных completions.

## Parse rate и truncation

Формат в ячейках: `parse rate / truncated / mean completion tokens`.

| Вариант | GSM8K | GSM1k | MATH | MMLU-STEM |
| --- | ---: | ---: | ---: | ---: |
| `r32/a64/bs64` | 98.79% / 19 / 156.5 | 98.92% / 18 / 186.0 | 83.08% / 816 / 479.3 | 94.29% / 177 / 203.8 |
| `r32/a128/bs64` | 99.17% / 14 / 151.9 | 98.34% / 25 / 206.6 | 84.78% / 753 / 458.4 | 93.37% / 211 / 224.0 |
| `r32/a64/bs32` | 98.03% / 25 / 160.8 | 96.93% / 37 / 198.0 | **70.26% / 1468 / 708.2** | **88.74% / 335 / 297.2** |

Главный симптом `bs32` — не просто снижение точности: на MATH средний ответ вырос на 48%, число
обрезанных ответов — на 80%, а parse rate упал на 12.82 п.п. Значит, меньший effective batch в
этом прогоне изменил стабильность генерации и не является кандидатом для длинного обучения.

## LoRA против rsLoRA: seeded diagnostic

Это ранний eval по 100 примеров на benchmark, а не full-split результат.

| Вариант | GSM8K | GSM1k | MATH | MMLU-STEM |
| --- | ---: | ---: | ---: | ---: |
| LoRA `r32/a64/bs64` | 75% | 65% | 36% | 52% |
| rsLoRA `r32/a64/bs64` | 54% | 45% | 22% | 30% |

При зафиксированных `r=32, alpha=64` rsLoRA меняет effective scaling, поэтому это не чистое
сравнение одной реализации с другой. Однако одновременно худшие validation loss и все четыре
diagnostic accuracy не дают оснований продолжать именно этот rsLoRA preset.

## Вывод и следующие запуски

Текущий победитель — обычная LoRA `r32/a64` с effective batch 64. Увеличение alpha до 128
стабильно ухудшило все четыре benchmark. Уменьшение batch до 32 привело к явной деградации, поэтому
третий запланированный `a128/bs32` был остановлен: он уже не мог ответить на полезный вопрос после
двух первых абляций.

Следующий парный эксперимент сравнивает повторение качественной 100k-выборки с добавлением новых
примеров при строго одинаковом optimizer-step и sample budget:

| Параметр | `100k × 3` | `296k × 1` |
| --- | ---: | ---: |
| Source limit | 100 032 | 296 000 |
| Validation | 2 048 | 2 048 |
| Train examples за эпоху | 97 984 | 293 952 |
| Эпохи, эквивалентные бюджету | 3 | 1 |
| Увидено train examples | 293 952 | 293 952 |
| Optimizer steps | 4593 | 4593 |

Общие параметры: LoRA `r32/a64/dropout=0.05`, microbatch `4`, gradient accumulation `16`,
effective batch `64`, LR `2e-4`, cosine decay, warmup `138` steps, fused AdamW
`betas=(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, max grad norm `1.0`. Внутренняя
validation и checkpoint — каждые 500 steps. После каждого запуска нужен тот же полный greedy vLLM
eval на четырёх benchmarks.

Эти два long-run можно сравнивать между собой напрямую. С коротким `r32/a64/bs64` baseline их
следует сравнивать как практический результат, но не приписывать всю дельту объёму данных: baseline
использовал constant LR, long-run использует cosine с warmup.

## Сырые артефакты

- SFT Trainer states: `/workspace/outputs/sft/<experiment>/checkpoint-*/trainer_state.json`;
- full eval summaries: `/workspace/outputs/eval/<experiment>/<run>/<timestamp>/summary.json`;
- W&B training run IDs: best LoRA `qct4xsey`, alpha-128 `anq10qsa`, bs-32 `cecjuies`;
- W&B full-eval run IDs: best LoRA `1iacwry2`, alpha-128 `xlpx7v51`, bs-32 `0chg3u6t`;
- rsLoRA diagnostic: training `rcfwpip3`, eval `z7y6af2`.

Конфиги long-run:

- `configs/sft/qwen2_5_1_5b_base_openmath_100k_3epochs_lora.yaml`;
- `configs/sft/qwen2_5_1_5b_base_openmath_296k_1epoch_lora.yaml`.
