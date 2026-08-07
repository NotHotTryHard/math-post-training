# Локальный baseline до post-training — 2026-08-07

## Среда

- MacBook Air, Apple M5, 24 GB unified memory;
- Python 3.11.15, PyTorch 2.13.0, Transformers 5.14.1, MPS;
- greedy decoding: `do_sample=false`, `temperature=0`, `top_p=1`, seed 42;
- локальный лимит генерации: `max_new_tokens=512`;
- все dataset revisions зафиксированы в `configs/current.yaml` и `summary.json`.

Instruct-модель оценивалась с протоколом `qwen2_5_math_instruct`: zero-shot chat prompt
официального Qwen evaluation harness с требованием ответа в `\boxed{}`. Base-модель оценивалась
с `qwen2_5_math_base`: фиксированные примеры из Appendix B Qwen2.5-Math — 8-shot для GSM8K
и 4-shot для MATH.

## Результаты

| Model                 | Benchmark | Scope      | Correct | Accuracy | 95% Wilson CI | Parse   | Format | Truncated   |
|                       |           |            |         |          |               |         |        |             |
| Qwen2.5-1.5B-Instruct | GSM8K     | full 1319  | 922     | 69.90%   | 67.37–72.32%  | 99.47%  | 91.58% | 58 (4.40%)  |
| Qwen2.5-1.5B-Instruct | GSM1k     | seeded 200 | 135     | 67.50%   | 60.73–73.61%  | 100.00% | 87.50% | 13 (6.50%)  |
| Qwen2.5-1.5B-Instruct | MATH      | seeded 140 | 44      | 31.43%   | 24.32–39.53%  | 97.86%  | 47.86% | 74 (52.86%) |
| Qwen2.5-1.5B Base     | GSM8K     | seeded 100 | 69      | 69.00%   | 59.37–77.22%  | 100.00% | n/a    | 1 (1.00%)   |

Limited runs сначала детерминированно перемешивают строки с `sample_seed=42`. MATH-140 также
сбалансирован round-robin: по 20 задач из каждой из семи категорий.

Разбивка MATH-140: algebra 70%, prealgebra 50%, geometry 30%, counting & probability 25%,
number theory 25%, precalculus 20%, intermediate algebra 0%. При 20 примерах на категорию это
диагностические, а не leaderboard-значения.

## Как это интерпретировать

GSM8K — основной пригодный baseline: использован полный test split, а ограничение длины затронуло
4.4% ответов. GSM1k согласуется с ним и не показывает очевидного провала вне GSM8K, но его
интервал шире из-за выборки в 200 задач.

MATH при `max_new_tokens=512` слишком часто обрезается. Его accuracy полезна как консервативный
локальный smoke baseline, но её нельзя выдавать за воспроизведение таблицы статьи. Для итоговых
сравнений после обучения MATH нужно прогнать с 2048 токенами через vLLM на Linux.

Числа `91.6 / 55.4` из статьи относятся к **Qwen2.5-Math-7B Base** на полных GSM8K и MATH
с 8/4-shot prompts. Здесь другая модель: general-purpose 1.5B. Самое
близкое локальное измерение — Base GSM8K-100 с тем же 8-shot prompt, однако ни размер выборки,
ни checkpoint не совпадают. В статье Qwen2.5-Math-1.5B Base получил 76.8 / 49.8.

Репозиторий DrEternity сообщает 68.5% для Qwen2.5-1.5B-Instruct на GSM8K; локальные 69.9%
с ним согласуются. Но это не точное воспроизведение: там включён sampling (`temperature=0.7`,
`top_p=0.8`, `top_k=20`) и другой system prompt. Его Base “CoT” также zero-shot sampled, а не
опубликованный 8-shot greedy protocol.

## Команды

```bash
# Полный локальный Instruct baseline на GSM8K
model-eval --benchmark gsm8k --batch-size 16 --max-new-tokens 512 --device mps

# Seeded random GSM1k-200
model-eval --benchmark gsm1k --limit 200 --batch-size 16 --max-new-tokens 512 --device mps

# Seeded и сбалансированный по категориям MATH-140
model-eval --benchmark math --limit 140 --batch-size 16 --max-new-tokens 512 --device mps

# Опубликованный Base 8-shot prompt на seeded GSM8K-100
model-eval \
  --config configs/eval/qwen2_5_1_5b_base_few_shot_cot.yaml \
  --benchmark gsm8k \
  --limit 100 \
  --batch-size 1 \
  --max-new-tokens 512 \
  --device mps
```

Сырые completions и `summary.json` лежат в `outputs/eval/` и намеренно игнорируются git.
Весь набор артефактов после этих запусков занимает меньше 1 MB.

## Источники протокола

- [Qwen2.5-Math technical report](https://arxiv.org/abs/2409.12122)
- [Official Qwen2.5-Math evaluation code](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation)
- [DrEternity/gsm8k-post-training](https://github.com/DrEternity/gsm8k-post-training)
