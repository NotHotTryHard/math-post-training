# math-post-training

Учебный проект по SFT и RL post-training небольших языковых моделей на математических задачах.
Основной pipeline: SFT с последующим GRPO для `Qwen/Qwen2.5-1.5B`.
Instruct checkpoint используется только как отдельный внешний evaluation baseline.

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
`eval` используется только отдельной командой benchmark evaluation.

## Supervised fine-tuning

SFT запускается независимо от evaluation. Обучается LoRA-адаптер, параметры которого находятся
в секции `lora` экспериментального конфига:

```bash
uv sync --group train
model-sft --config configs/current.yaml
```

Первичные OpenMathInstruct-2 прогоны запускаются отдельными полными конфигами:

```bash
model-sft --config configs/sft/qwen2_5_1_5b_base_openmath_1m_1epoch_lora.yaml
model-sft --config configs/sft/qwen2_5_1_5b_base_openmath_1m_1epoch_rslora.yaml
```

Исходная модель берётся из `model.name_or_path`. Адаптер сохраняется в
`<sft.output_dir>/adapter`, а совместимый с vLLM объединённый checkpoint — непосредственно в
`sft.output_dir`.
SFT использует plain-text `prompt`/`completion`, без ChatML и без `apply_chat_template`:

```text
Solve the following math problem step by step. Put your final answer within \boxed{}.

Problem: <problem>

Solution:
<solution><|endoftext|>
```

`<|endoftext|>` берётся из Base tokenizer-а и добавляется к completion ровно один раз.
SFT и GRPO откажутся запускаться с tokenizer-ом, чей native EOS отличается, поэтому случайно
подставить `-Instruct`/`<|im_end|>` в этот pipeline нельзя.
`dataset.validation` задаёт детерминированный holdout, который исключается из training split.
Во время обучения TRL пишет `eval_loss` в W&B на шагах `sft.eval_steps`, а после последней
проверки восстанавливает лучший checkpoint перед сохранением адаптера и объединённой модели.
Это внутренняя validation для выбора checkpoint-а; GSM1k, GSM8K, MATH и MMLU-STEM остаются
нетронутыми внешними benchmarks и запускаются отдельно через `model-eval`.
Текущие `max_steps: 20` предназначены для первого GPU smoke-run. Продолжить прерванный запуск
можно из полного checkpoint-а Trainer:

```bash
model-sft \
  --config configs/current.yaml \
  --resume-from-checkpoint outputs/sft/qwen2.5-1.5b-base-gsm8k-lora-native-eos/checkpoint-10
```

Команда не запускает evaluation автоматически. Полученный checkpoint проверяется явно:

```bash
model-eval \
  --config configs/current.yaml \
  --model outputs/sft/qwen2.5-1.5b-base-gsm8k-lora-native-eos
```

## Reinforcement learning

GRPO запускается отдельной стадией и не использует размеченные цепочки решения. Trainer получает
условие задачи, генерирует несколько ответов через vLLM и оптимизирует два проверяемых сигнала:
математическую корректность с весом `1.0` и наличие финального `\\boxed{}` с весом `0.1`.
`loss_type: dapo` включает DAPO-нормализацию policy loss; остальные элементы полного DAPO recipe,
например dynamic sampling, этим параметром автоматически не добавляются.

Для SFT → GRPO можно указать merged checkpoint в `model.name_or_path`. Если SFT checkpoint
сохранён только как PEFT adapter, `model.name_or_path` задаёт исходную base model, а
`model.adapter_name_or_path` — локальный путь или Hub ID адаптера. Перед созданием нового RL LoRA
код явно объединит SFT adapter с base weights. Пример такого запуска:

```bash
model-grpo \
  --config configs/grpo/qwen2_5_1_5b_base_openmath_296k_grpo_smoke.yaml
```

Результат сохраняется так же, как после SFT: адаптер находится в `<grpo.output_dir>/adapter`, а
готовый к vLLM merged checkpoint — непосредственно в `grpo.output_dir`.

Продолжение из полного Trainer checkpoint:

```bash
model-grpo \
  --config configs/grpo/qwen2_5_1_5b_base_openmath_296k_grpo_smoke.yaml \
  --resume-from-checkpoint /workspace/outputs/grpo/<experiment>/checkpoint-5
```

В `vllm_mode: colocate` генератор делит GPU с обучаемой моделью; долю памяти задаёт
`vllm_gpu_memory_utilization`. Для отдельного generation GPU можно переключить режим на `server`
и запустить совместимый `trl vllm-serve`. Встроенный GRPO backend намеренно не использует
`generation/vllm.py`: TRL сам синхронизирует обновлённые policy weights и корректирует расхождение
log probabilities между Transformers и vLLM.

После RL внешний benchmark остаётся отдельным явным шагом:

```bash
model-eval \
  --config configs/current.yaml \
  --model /workspace/outputs/grpo/qwen2.5-1.5b-base-openmath-296k-native-eos-grpo-smoke
```

## Локальная генерация

CLI использует `model` и `inference` из `configs/current.yaml`:

```bash
model-generate "If x + 3 = 7, what is x?"
```

По умолчанию используется тот же plain-text math prompt, что в SFT, GRPO и post-training eval.
Для произвольного raw completion без этого шаблона:

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
├── grpo/                # Самодостаточные GRPO experiment-конфиги.
├── config.example.yaml  # Полный пример конфигурации эксперимента.
└── current.yaml         # Текущий локальный эксперимент.

src/math_post_training/
├── cli.py               # Тонкая сборка model + prompt + generation для CLI.
├── config.py            # Загрузка YAML-конфигурации.
├── grpo.py              # Online RL через TRL GRPOTrainer и rule-based rewards.
├── model.py             # Общая загрузка model/tokenizer из HF или checkpoint-а.
├── rewards.py           # Проверяемые correctness и format rewards.
├── sft.py               # Независимый запуск supervised fine-tuning через TRL.
├── prompts/             # Финальный model input, разнесённый по сценариям.
│   ├── eval.py          # Явные Qwen Base/Instruct evaluation protocols.
│   ├── inference.py     # Prompt для ручного model-generate.
│   └── training.py      # Единый plain-text prompt для SFT, GRPO, eval и inference.
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

`model-sft` и `model-grpo` остаются независимыми стадиями и никогда не запускают друг друга
неявно.
