# math-rl

Учебный проект по RL post-training небольших языковых моделей на математических задачах.
Первый эксперимент: GRPO-дообучение `Qwen/Qwen2.5-1.5B-Instruct` на GSM8K.

## Структура

```text
src/math_rl/
├── config.py            # Загрузка и валидация конфигурации.
├── prompts.py           # Chat template и построение математических prompts.
├── rewards.py           # Reward-функции, используемые trainer-ом.
├── evaluation.py        # Evaluation loop и метрики.
├── training.py          # Сборка и запуск GRPOTrainer.
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
