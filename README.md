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

Для локальной разработки скопируй `.env.example` в `.env` и подставь свои значения. Сам файл
`.env` игнорируется git. Код загрузки `.env` будет добавлен вместе с CLI; сами Transformers и
W&B также умеют читать эти переменные окружения напрямую.

## Структура

```text
configs/
├── config.yaml.example  # Полный пример конфигурации эксперимента.
└── current.yaml         # Текущий локальный эксперимент.

src/math_post_training/
├── config.py            # Загрузка и валидация конфигурации.
├── model.py             # Общая загрузка model/tokenizer из HF или checkpoint-а.
├── prompts.py           # Chat template и построение математических prompts.
├── rewards.py           # Reward-функции, используемые trainer-ом.
├── evaluation.py        # Evaluation loop и метрики.
├── training.py          # SFT и GRPO training entry points.
├── data/
│   ├── loaders.py       # Загрузка GSM8K и будущих датасетов.
│   └── preprocessing.py # Приведение датасетов к общей схеме.
├── generation/
│   ├── base.py          # Общий контракт и backend-neutral параметры.
│   ├── transformers.py  # Адаптер для transformers.generate.
│   └── vllm.py          # Адаптер для vLLM с ленивым optional import.
└── verifiers/
    ├── extraction.py    # Извлечение финального ответа из completion.
    └── numeric.py       # Сравнение числовых ответов.
```
