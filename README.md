# math-post-training

Учебный проект по SFT и RL post-training небольших языковых моделей на математических задачах.
Планируются две ветки экспериментов: GRPO-дообучение `Qwen/Qwen2.5-1.5B-Instruct` и
SFT с последующим GRPO для `Qwen/Qwen2.5-1.5B` на GSM8K.

## Конфигурация

`configs/config.yaml.example` служит полным примером конфигурации, а `configs/current.yaml` —
текущим локальным экспериментом. Параметры модели, датасета, генерации и обучения хранятся в
YAML; секреты и настройки конкретного W&B workspace берутся из окружения:

- `HF_TOKEN` — доступ к Hugging Face;
- `WANDB_API_KEY` — авторизация в Weights & Biases;
- `WANDB_ENTITY` — пользователь или команда;
- `WANDB_PROJECT` — проект для логирования запусков.

Каждый эксперимент хранит настройки своего датасета прямо в этом же YAML: Hub ID, commit
revision, split, streaming и ограничение числа примеров. Python-адаптер источника отвечает только
за приведение его исходных колонок к общей math-схеме.

Для локальной разработки скопируй `.env.example` в `.env` и подставь свои значения. Сам файл
`.env` игнорируется git. Код загрузки `.env` будет добавлен вместе с CLI; сами Transformers и
W&B также умеют читать эти переменные окружения напрямую.

## Локальная генерация

CLI использует модель и sampling-параметры из `configs/current.yaml`:

```bash
model-generate "If x + 3 = 7, what is x?"
```

По умолчанию prompt оборачивается chat template-ом tokenizer-а. Для raw completion Base-модели:

```bash
model-generate --model Qwen/Qwen2.5-1.5B --raw "The answer to 2 + 2 is"
```

## Структура

```text
configs/
├── config.yaml.example  # Полный пример конфигурации эксперимента.
└── current.yaml         # Текущий локальный эксперимент.

src/math_post_training/
├── cli.py               # Тонкая сборка model + prompt + generation для CLI.
├── config.py            # Загрузка YAML-конфигурации.
├── model.py             # Общая загрузка model/tokenizer из HF или checkpoint-а.
├── prompts.py           # Chat template и построение математических prompts.
├── rewards.py           # Reward-функции, используемые trainer-ом.
├── evaluation.py        # Evaluation loop и метрики.
├── training.py          # SFT и GRPO training entry points.
├── data/
│   ├── schema.py        # Канонический MathExample.
│   ├── loaders.py       # Загрузка, sampling и нормализация датасета.
│   ├── preprocessing.py # SFT- и GRPO-представления одного MathExample.
│   └── sources/         # Отдельный адаптер исходных колонок каждого датасета.
├── generation/
│   ├── base.py          # Общий контракт и backend-neutral параметры.
│   ├── transformers.py  # Адаптер для transformers.generate.
│   └── vllm.py          # Адаптер для vLLM с ленивым optional import.
└── verifiers/
    ├── extraction.py    # Извлечение финального ответа из completion.
    └── numeric.py       # Сравнение числовых ответов.
```
