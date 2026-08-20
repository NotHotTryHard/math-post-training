# Qwen2.5-1.5B: vLLM greedy vs sampling — 2026-08-20

Полный baseline-эксперимент на четырёх prompt-конфигурациях и двух режимах декодирования.
Каждый из восьми запусков обработал одинаковые 10 677 задач: GSM8K 1 319, GSM1K 1 205,
MATH 5 000 и MMLU-STEM 3 153.

## Условия

- backend: vLLM;
- GPU: NVIDIA A100;
- batch size: 768;
- `max_new_tokens`: 2 048;
- seed: 42;
- greedy: `do_sample=false`, `temperature=0`, `top_p=1`;
- sampling: `do_sample=true`, `temperature=0.7`, `top_p=0.8`, `top_k=20`;
- код и конфиги: commit `b8c3730`.

## Accuracy

Значения в ячейках: greedy → sampling; изменение дано в процентных пунктах.

| Конфигурация       | GSM8K                 | GSM1K                 | MATH                  | MMLU-STEM             | Micro, все задачи     |
|--------------------|------------------------|------------------------|-----------------------|-----------------------|-----------------------|
| Instruct           | 74.678% → 71.569% −3.11 | 71.037% → 65.809% −5.23 | 53.900% → 52.920% −0.98 | 57.088% → 57.120% +0.03 | 59.343% → 57.919% −1.42 |
| Base zero-shot     | 61.789% → 55.497% −6.29 | 55.353% → 54.025% −1.33 | 29.480% → 28.500% −0.98 | 52.934% → 49.889% −3.04 | 43.317% → 41.032% −2.29 |
| Base zero-shot CoT | 49.507% → 44.807% −4.70 | 48.382% → 43.237% −5.15 | 29.520% → 27.960% −1.56 | 48.652% → 42.626% −6.03 | 39.768% → 36.096% −3.67 |
| Base few-shot CoT  | 64.139% → 59.060% −5.08 | 58.174% → 54.357% −3.82 | 32.080% → 29.200% −2.88 | 42.309% → 40.945% −1.36 | 42.006% → 39.196% −2.81 |

Sampling не дал содержательного улучшения ни в одной конфигурации. Единственный положительный
результат — Instruct MMLU-STEM, где изменился ровно один правильный ответ из 3 153; попарный
McNemar p=1.0. После Holm-коррекции значимы десять отрицательных сдвигов из шестнадцати.

Снижение accuracy нельзя объяснить только парсингом или лимитом длины: sampling одновременно
повысил общий parse rate и уменьшил число truncated-ответов во всех четырёх конфигурациях.
При одном ответе на задачу стохастическое декодирование чаще уводит рассуждение на менее надёжную
траекторию. Для этого eval-сценария greedy остаётся основным режимом; sampling имеет смысл отдельно
проверять с self-consistency или best-of-N.

## W&B runs и prediction-таблицы

| Конфигурация       | Greedy | Sampling |
|--------------------|--------|----------|
| Instruct           | [qac70va9](https://wandb.ai/nothottryhard-msu/math-post-training/runs/qac70va9) | [8oj7stws](https://wandb.ai/nothottryhard-msu/math-post-training/runs/8oj7stws) |
| Base zero-shot     | [w9en9spb](https://wandb.ai/nothottryhard-msu/math-post-training/runs/w9en9spb) | [3ywwcwci](https://wandb.ai/nothottryhard-msu/math-post-training/runs/3ywwcwci) |
| Base zero-shot CoT | [vbtkd6z9](https://wandb.ai/nothottryhard-msu/math-post-training/runs/vbtkd6z9) | [tqi4gvl7](https://wandb.ai/nothottryhard-msu/math-post-training/runs/tqi4gvl7) |
| Base few-shot CoT  | [kvg0glkg](https://wandb.ai/nothottryhard-msu/math-post-training/runs/kvg0glkg) | [y4z61d8u](https://wandb.ai/nothottryhard-msu/math-post-training/runs/y4z61d8u) |

Все восемь runs завершены со state `finished`. Полные prediction-таблицы содержат примерно 102 МБ
несжатого JSON и хранятся в W&B; дублировать их в обычной Git-истории намеренно не стали.

## Файлы

- [`manifest.json`](manifest.json) — provenance и список runs;
- [`runs.json`](runs.json) — полные сохранённые конфиги и скалярные метрики всех runs;
- [`benchmark_metrics.csv`](benchmark_metrics.csv) — 32 строки run × benchmark;
- [`source_metrics.csv`](source_metrics.csv) — результаты по каждому dataset source/subset;
- [`paired_comparison.csv`](paired_comparison.csv) — correct/correct, greedy-only, sampling-only,
  confidence intervals, McNemar p и Holm-adjusted p;
- [`paired_source_comparison.csv`](paired_source_comparison.csv) — изменения accuracy по отдельным
  MATH и MMLU-STEM категориям.
