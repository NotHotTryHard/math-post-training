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
`.env` игнорируется git. Конфиги обучения и W&B появятся тогда же, когда появится соответствующий
training-код: сейчас YAML не притворяется, будто уже реализованные стадии существуют.

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

## Evaluation

Evaluation запускается отдельно от train-зависимостей:

```bash
uv sync --group eval
model-evaluate --limit 10
```

`--limit 10` — быстрый локальный прогон десяти примеров из каждого benchmark. Без этого флага
используются полные test splits GSM8K, GSM1k и MATH. Один benchmark можно
выбрать отдельно (флаг разрешено повторять):

```bash
model-evaluate --benchmark gsm8k
```

Для MATH ограниченная выборка набирается round-robin по всем семи категориям, а не только из
первой категории `algebra`. Перед limited-run строки детерминированно перемешиваются с
`evaluation.sample_seed`, поэтому это не просто первые N строк. Размер батча для конкретной машины можно подобрать через
`--batch-size`; это не меняет greedy-предсказания. `--max-new-tokens` полезен для ограниченного
локального прогона, но уже меняет evaluation protocol и поэтому сохраняется в `summary.json`.

`evaluation.protocol: qwen2_5_math_instruct` воспроизводит официальный zero-shot CoT prompt с
ответом в `\\boxed{}`. Для Base checkpoint используется опубликованный few-shot протокол:

```bash
model-evaluate \
  --model Qwen/Qwen2.5-1.5B \
  --protocol qwen2_5_math_base \
  --benchmark gsm8k
```

Это фиксированные примеры из Appendix B Qwen2.5-Math: 8-shot для GSM8K/GSM1k и 4-shot для
MATH. В `summary.json` сохраняются protocol, число shots, dataset
revision и `is_full_split`; сравнивать с опубликованной таблицей можно только полный split с тем
же протоколом. Для проверки другого checkpoint:

```bash
model-evaluate --model outputs/checkpoints/my-model
```

Каждый запуск создаёт `summary.json` с метриками. Если `evaluation.save_predictions` включён,
рядом сохраняется `predictions.jsonl.gz`: сжатые задачи, ответы и результаты проверки нужны для
разбора ошибок и обычно занимают мегабайты или десятки мегабайт, а не гигабайты.

## Структура

```text
configs/
├── config.example.yaml  # Полный пример конфигурации эксперимента.
└── current.yaml         # Текущий локальный эксперимент.

src/math_post_training/
├── cli.py               # Тонкая сборка model + prompt + generation для CLI.
├── config.py            # Загрузка YAML-конфигурации.
├── model.py             # Общая загрузка model/tokenizer из HF или checkpoint-а.
├── prompts/             # Финальный model input, разнесённый по сценариям.
│   ├── inference.py     # Prompt для ручного model-generate.
│   └── evaluation.py    # Instruct/Base evaluation prompts и их settings.
├── evaluation.py        # Evaluation loop и метрики.
├── data/
│   ├── schema.py        # Канонический MathExample.
│   ├── loaders.py       # Загрузка, sampling и нормализация датасета.
│   ├── preprocessing.py # SFT- и GRPO-представления одного MathExample.
│   └── sources/         # Отдельный адаптер исходных колонок каждого датасета.
├── generation/
│   ├── base.py          # Общий контракт и backend-neutral параметры.
│   ├── transformers.py  # Адаптер для transformers.generate.
│   └── vllm.py          # Явная заглушка будущего vLLM backend-а.
└── verifiers/
    ├── extraction.py    # Извлечение финального ответа из completion.
    └── math.py          # Numeric/symbolic equivalence через Math-Verify.
```

Пустых `training.py` и `rewards.py` пока намеренно нет: они появятся вместе с первой рабочей SFT
стадией, чтобы наличие файла не создавало впечатление, будто за ним уже стоит реализованный путь.
