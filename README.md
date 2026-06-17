# Расчет ragas метрик для LLM моделей

## Обзор проекта

Проект нужен для сравнительной оценки качества ответов разных LLM-моделей на заранее подготовленном наборе вопросов.

Основная идея:

```text
questions.csv -> run_questions.py -> answers.csv -> score_with_ragas.py -> answers_ragas.csv
```

Сначала формируется входной датасет с вопросами и эталонными ответами. Затем выбранная модель отвечает на эти вопросы через OpenAI-compatible API. После этого отдельный скрипт заполняет автоматические оценочные метрики: фактическую корректность, смысловую близость и итоговый score по шкале 0/1/2.

Задачи проекта:

- хранить вопросы в воспроизводимом CSV-формате;
- прогонять один и тот же набор вопросов через разные модели;
- сохранять ответы, latency, token usage и ошибки;
- считать автоматические метрики качества;
- оставлять место для ручной оценки и комментариев;
- сравнивать модели по одним и тем же вопросам.

## Стек

Проект использует `uv` и Python 3.12.

Файл `.python-version`:

```text
3.12
```

Ограничение в `pyproject.toml`:

```text
requires-python = >=3.10,<3.13
```

Основные библиотеки:

```text
openai                  - вызов OpenAI-compatible chat API
pandas                  - чтение и запись CSV, работа с датасетами
numpy                   - расчет cosine similarity для embeddings
ragas==0.2.15           - библиотека для LLM/RAG evaluation
onnxruntime==1.20.1     - локальный запуск ONNX embedding-модели
transformers>=5.12.0    - tokenizer для embedding-модели
huggingface-hub         - скачивание embedding-модели с Hugging Face
jupyter / notebook      - работа с ноутбуками
ipykernel               - kernel для Jupyter
openpyxl                - работа с Excel-файлами
```

Важно: `transformers` используется здесь только для tokenizer. PyTorch для текущего ONNX-embedder не нужен.

## Установка

В папке проекта:

```powershell
uv sync
```

Если старая `.venv` была создана под Python 3.14 или заблокирована Windows, ее можно удалить и заново выполнить:

```powershell
uv sync
```

После успешной синхронизации основное окружение должно быть:

```text
.venv
```

Если временно используется отдельное окружение `.venv312`, можно запускать так:

```powershell
$env:UV_PROJECT_ENVIRONMENT="C:\Users\v.makarov\projects\asking_questions_to_LLM\.venv312"
```

Проверка:

```powershell
uv run python --version
uv run python score_with_ragas.py --help
```

## Входной датасет questions.csv

`questions.csv` содержит вопросы, эталонные ответы и метаданные.

Колонки:

```text
id
domain
source_type
source_id
source_url
question
context
expected_answer
question_type
difficulty
scoring_rubric
```

Описание:

```text
id               - уникальный идентификатор вопроса
domain           - предметная область, например data_structures или physics
source_type      - тип источника: article, textbook, general
source_id        - короткий id источника
source_url       - ссылка на источник
question         - вопрос, который будет задан модели
context          - контекст или краткая привязка к источнику
expected_answer  - эталонный ответ
question_type    - тип вопроса: factual, reasoning, comparison, application
difficulty       - сложность: easy, medium, hard
scoring_rubric   - правила оценки, может быть пустым
```

Пример чтения:

```python
import pandas as pd

questions_df = pd.read_csv("questions.csv", encoding="utf-8")
```

## Запуск вопросов: run_questions.py

`run_questions.py` читает `questions.csv`, задает вопросы выбранной модели и сохраняет итоговый CSV с ответами.

Обычный запуск:

```powershell
uv run python run_questions.py
```

Запуск с явным указанием модели:

```powershell
uv run python run_questions.py --model "qwen/qwen3.5-35b-a3b"
```

Запуск на первых 3 вопросах:

```powershell
uv run python run_questions.py --limit 3
```

Запуск с другим выходным файлом:

```powershell
uv run python run_questions.py --output qwenqwen3535b_answers.csv
```

В bash перенос строк делается через `\`:

```bash
uv run python run_questions.py \
  --input questions.csv \
  --output qwenqwen3535b_answers.csv \
  --model "qwen/qwen3.5-35b-a3b"
```

В PowerShell перенос строк делается через обратную кавычку:

```powershell
uv run python run_questions.py `
  --input questions.csv `
  --output qwenqwen3535b_answers.csv `
  --model "qwen/qwen3.5-35b-a3b"
```

### Параметры run_questions.py

```text
--input             входной CSV с вопросами, по умолчанию questions.csv
--output            выходной CSV с ответами, по умолчанию model_answers.csv
--model             имя модели
--base-url          OpenAI-compatible endpoint
--api-key           API key для endpoint
--temperature       temperature генерации, по умолчанию 0.0
--request-timeout   hard timeout на один вопрос, по умолчанию 300 секунд
--limit             прогнать только первые N вопросов
--save-every        сохранять промежуточный результат каждые N строк
```

Дефолтный endpoint:

```text
http://192.168.15.182:1234/v1
```

Дефолтная модель:

```text
qwen/qwen3.5-35b-a3b
```

### Qwen no-think режим

Для этих Qwen-моделей скрипт автоматически добавляет `/no_think` в конец user prompt:

```text
qwen3.5-vl-122b-a10b-mlx-crack
qwen/qwen3.5-35b-a3b
qwen3.5-397b-a17b
```

Это сделано, чтобы попытаться отключить длинные reasoning-блоки и уменьшить расход tokens. Для `gemma-4-31b-it-mlx` и других моделей `/no_think` не добавляется.

### Timeout

Запрос к модели выполняется в отдельном процессе. Если модель зависла дольше `--request-timeout`, процесс завершается, а в CSV записывается:

```text
model_answer = failed to answer
error = hard_timeout_after_300.0_seconds
```

Пример timeout 60 секунд:

```powershell
uv run python run_questions.py --request-timeout 60
```

## Итоговый датасет с ответами

После `run_questions.py` получается CSV, например:

```text
qwenqwen3535b_answers.csv
```

Основные колонки:

```text
id
domain
source_type
source_id
source_url
question_type
difficulty
question
context
expected_answer
scoring_rubric
model_answer
ragas_factual_correctness
ragas_semantic_similarity
ragas_final_score
manual_final_score
manual_comment
model
temperature
latency_sec
created_at
prompt_tokens
completion_tokens
total_tokens
error
```

После прогона модели поля `ragas_*`, `manual_final_score` и `manual_comment` остаются пустыми. Их заполняет следующий этап.

## Ragas и выбранные метрики

В проекте выбраны три автоматические метрики:

```text
ragas_factual_correctness
ragas_semantic_similarity
ragas_final_score
```

### ragas_factual_correctness

Оценивает фактическую правильность `model_answer` относительно `expected_answer`.

Шкала:

```text
1.0 - ответ фактически полностью правильный
0.5 - ответ частично правильный
0.0 - ответ неправильный
```

Это LLM-as-judge оценка: judge-модель получает вопрос, эталонный ответ и ответ проверяемой модели, затем возвращает score.

### ragas_semantic_similarity

Оценивает смысловую близость между:

```text
expected_answer
model_answer
```

Считается через embeddings и cosine similarity.

Интерпретация:

```text
ближе к 1 - ответы очень похожи по смыслу
ближе к 0 - ответы мало похожи по смыслу
```

Эта метрика не является LLM-as-judge. Она не “понимает” фактическую правильность так же строго, как judge-модель. Например ответы могут быть семантически близкими, но отличаться важной деталью вроде `O(1)` и `O(n)`.

### ragas_final_score

Итоговая автоматическая оценка по простой rubric:

```text
2 - ответ полностью правильный
1 - ответ частично правильный
0 - ответ неправильный
```

Это тоже LLM-as-judge оценка.

### Ручные поля

```text
manual_final_score  - ручная оценка по шкале 0/1/2
manual_comment      - ручной комментарий
```

Они нужны, чтобы сравнить автоматические оценки с человеческой разметкой.

## Заполнение метрик: score_with_ragas.py

`score_with_ragas.py` читает CSV с ответами модели и заполняет:

```text
ragas_factual_correctness
ragas_semantic_similarity
ragas_final_score
```

Минимальный запуск:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas.csv `
  --judge-model gemma-4-31b-it-mlx
```

Тест на первых 3 строках:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas_test.csv `
  --judge-model gemma-4-31b-it-mlx `
  --limit 3
```

То же для bash:

```bash
uv run python score_with_ragas.py \
  --input qwenqwen3535b_answers.csv \
  --output qwenqwen3535b_answers_ragas_test.csv \
  --judge-model gemma-4-31b-it-mlx \
  --limit 3
```

### Параметры score_with_ragas.py

```text
--input                  входной CSV с ответами модели
--output                 выходной CSV с заполненными ragas_* метриками
--judge-base-url         OpenAI-compatible endpoint для judge-модели
--judge-api-key          API key для judge endpoint
--judge-model            модель-судья, по умолчанию gemma-4-31b-it-mlx
--judge-temperature      temperature judge-модели, по умолчанию 0.0
--request-timeout        timeout одного judge-запроса
--embedding-model        Hugging Face repo embedding-модели
--embedding-onnx-file    путь к ONNX-файлу внутри repo или auto
--embedding-cache-dir    локальный cache для Hugging Face файлов
--embedding-max-length   максимальная длина tokenizer input
--limit                  обработать только первые N строк
--save-every             сохранять результат каждые N строк
--skip-existing          пропускать строки, где все ragas_* уже заполнены
```

## Judge модель и embedding модель

В проекте используются два разных типа моделей.

### Judge модель

Judge модель - это обычная chat/instruct LLM. Она читает вопрос, эталонный ответ и ответ проверяемой модели, после чего выставляет score.

По умолчанию:

```text
gemma-4-31b-it-mlx
```

Она используется для:

```text
ragas_factual_correctness
ragas_final_score
```

Judge-модель вызывается через локальный OpenAI-compatible endpoint.

### Embedding модель

Embedding модель не генерирует текстовый ответ. Она превращает текст в вектор чисел.

Она используется для:

```text
ragas_semantic_similarity
```

По умолчанию:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Эта модель скачивается с Hugging Face и запускается локально через ONNX Runtime.

## ONNX и почему не нужен PyTorch

ONNX - это формат для запуска нейросетевых моделей в inference-режиме без исходного framework.

В этом проекте embedding-модель запускается через:

```text
onnxruntime
```

Поэтому PyTorch не нужен. `transformers` используется только для tokenizer: он разбивает текст на `input_ids`, `attention_mask` и, если нужно, `token_type_ids`. После этого ONNX Runtime считает embedding-векторы.

Предупреждение вида:

```text
[transformers] PyTorch was not found. Models won't be available...
```

не является ошибкой для этого проекта. Tokenizer работает без PyTorch, а сама модель запускается через ONNX.

## Hugging Face cache и .hf-cache

Первый запуск `score_with_ragas.py` скачивает tokenizer и ONNX-веса embedding-модели с Hugging Face.

По умолчанию они сохраняются в:

```text
.hf-cache/
```

Это локальный cache внутри проекта. Повторные запуски будут использовать уже скачанные файлы.

Предупреждение про symlinks на Windows можно игнорировать:

```text
huggingface_hub cache-system uses symlinks...
```

Cache будет работать, просто может занимать больше места.

## Выбор embedding модели

Текущий дефолт:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Почему она выбрана:

- поддерживает русский и другие языки;
- легче, чем крупные multilingual embedding-модели;
- подходит для semantic similarity;
- имеет ONNX-вариант.

Альтернативы:

```text
BAAI/bge-m3
intfloat/multilingual-e5-large
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Они могут быть качественнее, но обычно тяжелее. Для них нужно проверять наличие ONNX-файлов в Hugging Face repo и совместимость с tokenizer/model inputs.

Пример смены embedding-модели:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas.csv `
  --embedding-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Если auto-поиск ONNX-файла не подходит, можно указать файл явно:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas.csv `
  --embedding-onnx-file onnx/model.onnx
```

## Summary of Workflow

1. Подготовить вопросы:

```text
questions.csv
```

2. Прогнать модель:

```powershell
uv run python run_questions.py `
  --input questions.csv `
  --output qwenqwen3535b_answers.csv `
  --model "qwen/qwen3.5-35b-a3b"
```

3. Проверить первые строки с Ragas-метриками:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas_test.csv `
  --judge-model gemma-4-31b-it-mlx `
  --limit 3
```

4. Посчитать метрики на всем файле:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas.csv `
  --judge-model gemma-4-31b-it-mlx
```

5. Открыть результат в pandas:

```python
import pandas as pd

df = pd.read_csv("qwenqwen3535b_answers_ragas.csv", encoding="utf-8-sig")
df.head()
```

6. Посмотреть средние оценки:

```python
df[[
    "ragas_factual_correctness",
    "ragas_semantic_similarity",
    "ragas_final_score",
    "manual_final_score",
]].mean(numeric_only=True)
```

7. Сравнить модели по одинаковым вопросам:

```python
df.groupby("question_type")[["ragas_final_score", "manual_final_score"]].mean()
```

## Частые замечания

Если команда с переносами строк падает в bash:

```text
bash: --input: command not found
```

значит использован PowerShell-перенос строки. В bash нужен `\`, а в PowerShell нужна обратная кавычка.

Если модель не ответила за timeout, в CSV будет:

```text
model_answer = failed to answer
```

Если `ragas_semantic_similarity` кажется высокой, но ответ неверный, это нормально: semantic similarity измеряет близость смысла, а не строгую фактическую правильность. Для строгой проверки смотрите `ragas_factual_correctness`, `ragas_final_score` и ручную оценку.

CSV-файлы сохраняются в `utf-8-sig`, чтобы их было проще открывать в Excel на Windows.
