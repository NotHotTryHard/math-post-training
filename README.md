# math-post-training

Учебный проект по SFT и RL post-training небольших языковых моделей на математических задачах.
Планируются две ветки экспериментов: GRPO-дообучение `Qwen/Qwen2.5-1.5B-Instruct` и
SFT с последующим GRPO для `Qwen/Qwen2.5-1.5B` на GSM8K.

## Конфигурация

`configs/config.example.yaml` служит полным примером конфигурации со смесью источников, а
`configs/current.yaml` — текущим локальным экспериментом с одним источником. Параметры модели,
датасета, ручного inference и evaluation хранятся в YAML; секреты и настройки конкретного W&B workspace
берутся из окружения:

- `HF_TOKEN` — доступ к Hugging Face;
- `WANDB_API_KEY` — авторизация в Weights & Biases;
- `WANDB_ENTITY` — пользователь или команда;
- `WANDB_PROJECT` — проект для логирования запусков.

Каждый эксперимент хранит настройки своего датасета прямо в этом же YAML: Hub ID, commit
revision, split, streaming и ограничение числа примеров. Python-адаптер источника отвечает только
за приведение его исходных колонок к общей math-схеме.

`dataset.sources` используется и для одного источника, и для смеси. `limit` ограничивает число
доступных строк конкретного источника после shuffle. `probability` нужна только при нескольких
источниках и определяет вероятность выбрать источник при получении следующей строки смеси.

Для локальной разработки скопируй `.env.example` в `.env` и подставь свои значения. Сам файл
`.env` игнорируется git. Параметры supervised fine-tuning находятся в секции `sft`; секция
`eval` используется только отдельной командой evaluation.

## Supervised fine-tuning

SFT запускается независимо от evaluation:

```bash
uv sync --group train
model-sft --config configs/current.yaml
```

Исходная модель берётся из `model.name_or_path`, а итоговая сохраняется в `sft.output_dir`.
Текущие `max_steps: 20` предназначены для первого GPU smoke-run. Продолжить прерванный запуск
можно из полного checkpoint-а Trainer:

```bash
model-sft \
  --config configs/current.yaml \
  --resume-from-checkpoint outputs/sft/qwen2.5-1.5b-instruct-gsm8k/checkpoint-10
```

Команда не запускает evaluation автоматически. Полученный checkpoint проверяется явно:

```bash
model-eval \
  --config configs/current.yaml \
  --model outputs/sft/qwen2.5-1.5b-instruct-gsm8k
```

## Локальная генерация

CLI использует `model` и `inference` из `configs/current.yaml`:

```bash
model-generate "If x + 3 = 7, what is x?"
```

По умолчанию prompt оборачивается chat template-ом tokenizer-а. Для raw completion Base-модели:

```bash
model-generate --model Qwen/Qwen2.5-1.5B --raw "The answer to 2 + 2 is"
```

Чтобы увидеть точную строку, которая будет передана tokenizer-у перед генерацией:

```bash
model-generate --show-prompt "Write a short story about a robot"
```

Логика построения этой строки целиком находится в `prompts/inference.py`; CLI только загружает
модель, вызывает её и печатает completion.

По умолчанию inference и evaluation используют in-process vLLM backend. Параметры движка
`tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len` и prefix caching находятся в
верхнеуровневой секции `vllm`. Для отладки без vLLM можно вернуть `backend: transformers`; флаг
`--device` относится только к этому backend-у.

Контейнер с vLLM нужно запускать с доступом к NVIDIA GPU и достаточной shared memory, например:

```bash
docker run --gpus all --ipc=host math-post-training
```

## Evaluation

Evaluation запускается отдельно от train-зависимостей:

```bash
uv sync --group eval
model-eval --limit 10
```

`--limit 10` — быстрый локальный прогон десяти примеров из каждого benchmark. Без этого флага
используются полные test splits GSM8K, GSM1k, MATH и MMLU-STEM. Один benchmark можно
выбрать отдельно (флаг разрешено повторять):

```bash
model-eval --benchmark gsm8k
```

Для MATH и MMLU-STEM ограниченная выборка набирается round-robin по категориям, а не только из
первой. Перед limited-run строки детерминированно перемешиваются с
`eval.sample_seed`, поэтому это не просто первые N строк. Размер батча для конкретной машины можно подобрать через
`--batch-size`; это не меняет greedy-предсказания. `--max-new-tokens` полезен для ограниченного
локального прогона, но уже меняет evaluation protocol и поэтому сохраняется в `summary.json`.

Для исходных 1.5B checkpoints зафиксированы четыре полных baseline-конфига:

```bash
model-eval --config configs/eval/qwen2_5_1_5b_base_zero_shot.yaml
model-eval --config configs/eval/qwen2_5_1_5b_base_zero_shot_cot.yaml
model-eval --config configs/eval/qwen2_5_1_5b_base_few_shot_cot.yaml
model-eval --config configs/eval/qwen2_5_1_5b_instruct.yaml
```

Первые три запуска измеряют Base checkpoint с raw zero-shot, raw zero-shot+CoT и опубликованным
few-shot+CoT prompt соответственно. Последний запускает Instruct checkpoint через chat template
с официальной CoT system instruction. Published Base protocol содержит 8 GSM8K/GSM1K и 4 MATH
demonstrations из Appendix B Qwen2.5-Math.

В двух zero-shot Base-конфигах MMLU-STEM тоже запускается zero-shot. Published Base protocol
использует фиксированный 4-shot CoT prompt из Appendix B, а Instruct protocol — фиксированный
5-shot CoT prompt из официального Qwen evaluation harness. В обоих случаях метрика считается
на `test`; дополнительные строки из `dev` не загружаются.

MMLU-STEM здесь измеряется генерацией ответа и exact-match по букве A–D. Это удобно для общего
pipeline, но не следует сравнивать с MMLU-числами, полученными через log-likelihood scoring, без
проверки совпадения протоколов.

Для проверки другого checkpoint достаточно заменить веса, не создавая новый evaluator:

```bash
model-eval --model outputs/checkpoints/my-model
```

Каждый запуск создаёт `summary.json` с метриками. Если `eval.save_predictions` включён,
рядом сохраняется `predictions.jsonl.gz`: сжатые задачи, ответы и результаты проверки нужны для
разбора ошибок и обычно занимают мегабайты или десятки мегабайт, а не гигабайты.

При `eval.wandb.enabled: true` тот же запуск создаёт W&B run с `job_type=eval`. Итоговые
`accuracy`, `parse_rate`, `format_rate`, число обрезанных ответов, время и throughput каждого
benchmark записываются в summary run-а. Полные строки eval-а доступны как отдельные
`wandb.Table`: `eval/gsm8k/predictions`, `eval/math/predictions` и так далее. Поэтому в интерфейсе
можно фильтровать ошибки по способу извлечения ответа и truncation, а не только смотреть на
графики. Во время запуска W&B
также получает счётчик `eval/<benchmark>/processed`; в терминале тот же прогресс показывает
`tqdm`.

W&B берёт `WANDB_PROJECT`, `WANDB_ENTITY` и `WANDB_API_KEY` из окружения. Для полностью локальной
проверки без загрузки в облако можно запустить:

```bash
WANDB_MODE=offline model-eval --limit 1
```

Такой run позже можно отправить командой `wandb sync`. Чтобы совсем отключить интеграцию,
достаточно поставить `eval.wandb.enabled: false`.

Ответ извлекается каскадом: последнее `\\boxed{}`, затем `####`, затем последний явный маркер
`The answer is`/`final answer is`. Только если модель не соблюла ни один формат, используется
эвристика последнего числа или последней A–D буквы. Поле `extraction_method` позволяет отдельно
увидеть такие fallback-ответы; в summary сохраняется их распределение.

## Структура

```text
configs/
├── eval/                # Замороженные full-split baseline-прогоны.
├── config.example.yaml  # Полный пример конфигурации эксперимента.
└── current.yaml         # Текущий локальный эксперимент.

src/math_post_training/
├── cli.py               # Тонкая сборка model + prompt + generation для CLI.
├── config.py            # Загрузка YAML-конфигурации.
├── model.py             # Общая загрузка model/tokenizer из HF или checkpoint-а.
├── sft.py               # Независимый запуск supervised fine-tuning через TRL.
├── prompts/             # Финальный model input, разнесённый по сценариям.
│   ├── inference.py     # Prompt для ручного model-generate.
│   └── eval.py          # Явные Qwen Base/Instruct evaluation protocols.
├── eval.py              # Evaluation loop, метрики, tqdm и W&B-логирование.
├── data/
│   ├── schema.py        # Канонический MathExample.
│   ├── loaders.py       # Загрузка, sampling и нормализация датасета.
│   ├── preprocessing.py # SFT- и GRPO-представления одного MathExample.
│   └── sources/         # Адаптеры GSM8K, MATH, MMLU и training datasets.
├── generation/
│   ├── base.py          # Общий контракт и backend-neutral параметры.
│   ├── transformers.py  # Адаптер для transformers.generate.
│   └── vllm.py          # In-process offline inference через vLLM.
└── verifiers/
    ├── extraction.py    # Извлечение финального ответа из completion.
    ├── math.py          # Numeric/symbolic equivalence через Math-Verify.
    └── choice.py        # Exact-match для multiple-choice ответов.
```

GRPO и reward functions будут добавлены отдельной стадией; `model-sft` их не запускает.
