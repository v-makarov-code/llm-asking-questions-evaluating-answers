# Расчет метрик и оценка ответов AI-чат бота

## 1. Обзор проекта

Проект нужен для пакетной проверки качества ответов AI-чат бота на заранее подготовленном наборе вопросов.

Основной workflow:

```text
input_questions.csv
  -> request_to_chat_bot.py
  -> chatbot_answers.csv
  -> score_with_ragas.py
  -> chatbot_answers_scored.csv
```

Сначала `request_to_chat_bot.py` читает вопросы из CSV, отправляет их в AI-чат бот и сохраняет ответы в колонку `model_answer`. Затем `score_with_ragas.py` считает выбранные метрики качества и добавляет только релевантные для них колонки.

Проект ориентирован именно на оценку ответов чат-бота, поэтому из новых CSV убраны старые технические поля для локальных LLM: `prompt_tokens`, `completion_tokens`, `total_tokens`, `temperature`, `difficulty`, `question_type`, `source_url`, `source_type`, `source_id`, `scoring_rubric`.

`error` оставлен намеренно: он нужен, чтобы отличать плохой ответ модели от технического сбоя запроса или scoring.

## 2. Стек

Проект использует `uv` и Python 3.12.

Файл `.python-version`:

```text
3.12
```

Ограничение из `pyproject.toml`:

```text
requires-python = ">=3.10,<3.13"
```

Основные зависимости:

```text
requests              - запросы к AI-чат боту
python-dotenv         - чтение AUTH_TOKEN из .env
pandas                - чтение и запись CSV
numpy                 - расчет cosine similarity
openai                - OpenAI-compatible client для judge-модели
ragas                 - factual correctness через Ragas
instructor            - структурированный JSON-ответ judge-модели через Ragas
onnxruntime           - локальный запуск embedding-модели без PyTorch
transformers          - tokenizer для embedding-модели
huggingface-hub       - скачивание ONNX embedding-модели
jupyter / notebook    - работа с ноутбуками
ipykernel             - kernel для Jupyter
matplotlib / seaborn  - анализ и визуализация результатов
openpyxl              - работа с Excel-файлами
```

Важно: предупреждение `PyTorch was not found` для текущего embedding-подхода не критично. PyTorch не нужен, потому что embedding-модель запускается через ONNX Runtime, а `transformers` используется только для tokenizer.

## 3. Установка

В папке проекта:

```bash
uv sync
```

Проверка окружения:

```bash
uv run python --version
uv run python request_to_chat_bot.py --help
uv run python score_with_ragas.py --help
```

Для запросов к AI-чат боту нужен файл `.env`:

```text
AUTH_TOKEN=your_token_here
```

Токен используется как:

```text
Authorization: Token AUTH_TOKEN
```

## 4. Пример использования

### 4.1. Шаг 1: отправить вопросы в AI-чат бот

Тестовый запуск на 3 вопросах:

```bash
uv run python request_to_chat_bot.py \
  --input input_questions.csv \
  --output chatbot_answers_test.csv \
  --limit 3 \
  --delay 2 \
  --save-every 1
```

Запуск на весь файл:

```bash
uv run python request_to_chat_bot.py \
  --input input_questions.csv \
  --output chatbot_answers.csv \
  --delay 2 \
  --retry-delays 2,5,10 \
  --timeout 300 \
  --save-every 1
```

Продолжить после остановки или ошибки:

```bash
uv run python request_to_chat_bot.py \
  --input chatbot_answers.csv \
  --output chatbot_answers.csv \
  --skip-existing \
  --delay 2 \
  --retry-delays 2,5,10 \
  --timeout 300 \
  --save-every 1
```

`--skip-existing` пропускает строки, где `model_answer` уже заполнен. Значение `failed to answer` не считается нормальным заполненным ответом, поэтому такие строки будут отправлены повторно.

### Входной CSV для `request_to_chat_bot.py`

Минимально обязательные колонки:

```text
question
expected_answer
```

Рекомендуемый входной формат:

```text
id
domain
question
context
expected_answer
manual_final_score
manual_comment
error
```

Описание колонок:

```text
id                  - идентификатор вопроса
domain              - предметная область или раздел
question            - вопрос, который будет отправлен в чат-бот
context             - дополнительный контекст для анализа, если он есть
expected_answer     - эталонный ответ для последующей оценки
manual_final_score  - ручная итоговая оценка, заполняется человеком
manual_comment      - ручной комментарий, заполняется человеком
error               - техническая ошибка, если она возникла
```

### Выходной CSV из `request_to_chat_bot.py`

Скрипт сохраняет упрощенный формат:

```text
id
domain
question
context
expected_answer
model_answer
manual_final_score
manual_comment
latency_sec
created_at
error
```

Описание новых колонок:

```text
model_answer  - ответ AI-чат бота
latency_sec   - время получения ответа в секундах
created_at    - время записи ответа
error         - текст ошибки, если запрос не удался
```

Если запрос завершился ошибкой:

```text
model_answer = failed to answer
error = SSLError: ...
```

### 4.2. Шаг 2: посчитать метрики

Тестовый запуск на 1 строке только для `ragas_final_score`:

```bash
uv run python score_with_ragas.py \
  --input chatbot_answers.csv \
  --output chatbot_answers_scored_test.csv \
  --metrics ragas_final_score \
  --judge-model gemma-4-31b-it-mlx \
  --limit 1 \
  --save-every 1
```

Запуск только кастомной LLM-as-a-judge оценки:

```bash
uv run python score_with_ragas.py \
  --input chatbot_answers.csv \
  --output chatbot_answers_scored.csv \
  --metrics ragas_final_score \
  --judge-model gemma-4-31b-it-mlx \
  --skip-existing \
  --save-every 1
```

Запуск всех текущих метрик:

```bash
uv run python score_with_ragas.py \
  --input chatbot_answers.csv \
  --output chatbot_answers_scored.csv \
  --metrics ragas_factual_correctness,ragas_semantic_similarity,ragas_final_score \
  --judge-model gemma-4-31b-it-mlx \
  --skip-existing \
  --save-every 1
```

Посчитать только следующие 20 незаполненных строк:

```bash
uv run python score_with_ragas.py \
  --input chatbot_answers_scored.csv \
  --output chatbot_answers_scored.csv \
  --metrics ragas_final_score \
  --skip-existing \
  --max-new 20 \
  --save-every 1
```

### Входной CSV для `score_with_ragas.py`

Входом является выходной CSV из `request_to_chat_bot.py`.

Обязательные колонки:

```text
question
expected_answer
model_answer
```

Рекомендуемый входной формат:

```text
id
domain
question
context
expected_answer
model_answer
manual_final_score
manual_comment
latency_sec
created_at
error
```

`latency_sec` и `created_at` читаются, но в итоговый scored CSV не сохраняются, потому что они относятся к запросу чат-бота, а не к оценке.

### Выходной CSV из `score_with_ragas.py`

Выходной формат динамически зависит от `--metrics`.

Если выбрана только:

```bash
--metrics ragas_final_score
```

выходной CSV:

```text
id
domain
question
context
expected_answer
model_answer
ragas_final_score
ragas_final_explanation
manual_final_score
manual_comment
judge_model
error
```

Если выбрана только:

```bash
--metrics ragas_semantic_similarity
```

выходной CSV:

```text
id
domain
question
context
expected_answer
model_answer
ragas_semantic_similarity
manual_final_score
manual_comment
error
```

Если выбраны все метрики:

```bash
--metrics ragas_factual_correctness,ragas_semantic_similarity,ragas_final_score
```

выходной CSV:

```text
id
domain
question
context
expected_answer
model_answer
ragas_factual_correctness
ragas_semantic_similarity
ragas_final_score
ragas_final_explanation
manual_final_score
manual_comment
judge_model
error
```

Описание scoring-колонок:

```text
ragas_factual_correctness  - factual correctness по Ragas, диапазон 0..1
ragas_semantic_similarity  - cosine similarity между embedding expected_answer и model_answer
ragas_final_score          - кастомная оценка judge-модели по шкале 0/1/2
ragas_final_explanation    - короткое объяснение judge-модели на русском языке
judge_model                - модель, которая использовалась как LLM-as-a-judge
manual_final_score         - ручная оценка человека
manual_comment             - ручной комментарий человека
error                      - техническая ошибка запроса к чат-боту, если была
```

`judge_model` добавляется только для метрик, где реально используется LLM-as-a-judge:

```text
ragas_factual_correctness
ragas_final_score
```

Для одной `ragas_semantic_similarity` колонка `judge_model` не добавляется.

## 5. Какие есть метрики

### `ragas_final_score`

Кастомная LLM-as-a-judge оценка по простой шкале:

```text
2 - ответ полностью правильный
1 - ответ частично правильный
0 - ответ неправильный
```

Judge-модель получает:

```text
question
expected_answer
model_answer
```

и возвращает JSON:

```json
{"score": 2, "explanation": "Ответ полностью соответствует эталону."}
```

Эта метрика удобна как основной грубый score, который легко сравнивать с ручной оценкой `manual_final_score`.

### `ragas_factual_correctness`

Настоящая метрика Ragas `FactualCorrectness`.

Она сравнивает `model_answer` с `expected_answer` через разбор на фактические утверждения:

```text
1. Ragas разбивает ответ модели на claims.
2. Ragas разбивает эталонный ответ на claims.
3. Judge-модель проверяет, какие утверждения подтверждаются.
4. Итоговый score считается как F1 между factual precision и factual recall.
```

Интерпретация:

```text
1.0 - фактически полное соответствие эталону
0.5 - частичное соответствие
0.0 - фактического соответствия почти нет
```

Для этой метрики нужна judge-модель, например:

```bash
--judge-model gemma-4-31b-it-mlx
```

### `ragas_semantic_similarity`

Смысловая близость между `expected_answer` и `model_answer`.

Как считается:

```text
1. expected_answer переводится в embedding-вектор.
2. model_answer переводится в embedding-вектор.
3. Между векторами считается cosine similarity.
```

По умолчанию используется embedding-модель:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Она скачивается с Hugging Face в `.hf-cache` и запускается локально через ONNX Runtime. PyTorch для этого не нужен.

Метрика полезна как быстрый сигнал смысловой близости, но она не заменяет factual correctness: два ответа могут быть похожими по смыслу, но отличаться важными фактами.

### Метрики, которые пока не используются

`faithfulness`, `answer_relevancy`, `context_precision` полезны для полноценного RAG-workflow.

`faithfulness` проверяет, подтверждается ли ответ найденными контекстами `retrieved_contexts`.

`answer_relevancy` проверяет, насколько ответ относится к вопросу.

`context_precision` оценивает качество retrieval: насколько найденные контексты релевантны и хорошо отсортированы.

В текущем workflow они не являются основными, потому что чат-бот возвращает финальный ответ, а не список реально использованных `retrieved_contexts`.

## 6. Параметры скриптов

### `request_to_chat_bot.py`

```text
--input
```

Входной CSV с вопросами. По умолчанию:

```text
RAG_questions_answers.csv
```

```text
--output
```

Выходной CSV с ответами чат-бота. По умолчанию:

```text
AI_chat_bot_answers.csv
```

```text
--url
```

Endpoint AI-чат бота. По умолчанию:

```text
https://ai.sapiens.solutions/api/v1/conversations/ask/stream
```

```text
--env-file
```

Файл с переменными окружения. По умолчанию:

```text
.env
```

```text
--delimiter
```

Разделитель CSV. По умолчанию:

```text
;
```

```text
--timeout
```

Timeout одного HTTP-запроса в секундах. По умолчанию:

```text
300
```

```text
--limit
```

Обработать только первые N строк. Удобно для тестового запуска.

```text
--save-every
```

Сохранять промежуточный результат каждые N обработанных строк. По умолчанию:

```text
1
```

```text
--delay
```

Пауза между вопросами в секундах. Полезно, если сервер нестабилен или ограничивает частоту запросов.

```text
--retry-delays
```

Задержки между повторными попытками при сетевых ошибках. Формат:

```text
2,5,10
```

Это значит: после первой ошибки подождать 2 секунды, после второй 5 секунд, после третьей 10 секунд.

```text
--skip-existing
```

Пропускать строки, где `model_answer` уже заполнен. `failed to answer` не считается заполненным ответом.

### `score_with_ragas.py`

```text
--input
```

Входной CSV с ответами чат-бота.

```text
--output
```

Выходной CSV с рассчитанными метриками.

```text
--delimiter
```

Разделитель CSV. По умолчанию:

```text
;
```

```text
--judge-base-url
```

OpenAI-compatible endpoint judge-модели. По умолчанию:

```text
http://192.168.15.182:1234/v1
```

```text
--judge-api-key
```

API key для judge endpoint. Для локального сервера по умолчанию используется:

```text
sk-no-key-required
```

```text
--judge-model
```

Модель, которая используется как LLM-as-a-judge. По умолчанию:

```text
gemma-4-31b-it-mlx
```

```text
--judge-temperature
```

Temperature judge-модели. По умолчанию:

```text
0.0
```

Для оценки лучше оставлять `0.0`, чтобы результаты были стабильнее.

```text
--judge-max-tokens
```

Максимальное количество токенов для ответа judge-модели. По умолчанию:

```text
8192
```

Особенно важно для `ragas_factual_correctness`, потому что Ragas может просить judge-модель разложить длинные ответы на утверждения.

```text
--request-timeout
```

Timeout запроса к judge-модели в секундах. По умолчанию:

```text
300
```

```text
--embedding-model
```

Hugging Face repo embedding-модели. По умолчанию:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

```text
--embedding-onnx-file
```

Имя ONNX-файла внутри Hugging Face repo. По умолчанию:

```text
auto
```

При `auto` скрипт сам выбирает подходящий `.onnx` файл из папки `onnx/`.

```text
--embedding-cache-dir
```

Папка для кеша Hugging Face. По умолчанию:

```text
.hf-cache
```

```text
--embedding-max-length
```

Максимальная длина текста для tokenizer embedding-модели. По умолчанию:

```text
256
```

```text
--metrics
```

Список метрик через запятую. Доступные значения:

```text
ragas_factual_correctness
ragas_semantic_similarity
ragas_final_score
```

Можно писать коротко:

```text
factual_correctness
semantic_similarity
final_score
```

Дефолт:

```text
ragas_factual_correctness,ragas_semantic_similarity,ragas_final_score
```

```text
--limit
```

Взять только первые N строк входного CSV. Важно: `--limit` применяется до `--skip-existing`.

```text
--max-new
```

Обработать максимум N новых незаполненных строк после учета `--skip-existing`. Удобно для запуска батчами по 20 строк.

```text
--save-every
```

Сохранять промежуточный результат каждые N обработанных строк. По умолчанию:

```text
1
```

```text
--skip-existing
```

Пропускать строки, где уже заполнены все выбранные метрики.

Для `ragas_final_score` строка считается заполненной только если заполнены оба поля:

```text
ragas_final_score
ragas_final_explanation
```

Для `ragas_factual_correctness` проверяется:

```text
ragas_factual_correctness
```

Для `ragas_semantic_similarity` проверяется:

```text
ragas_semantic_similarity
```

### Streamlit review judge

Для ручной проверки качества judge-модели используется:

```text
review_judge_streamlit.py
```

Пример запуска для готового файла с оценками Qwen judge:

```bash
uv run streamlit run review_judge_streamlit.py -- \
  --input data/forqwen_judge_scored_qwen397b.csv \
  --output data/forqwen_judge_scored_qwen397b_reviewed.csv \
  --reviewer v.makarov
```

Приложение показывает вопрос, эталонный ответ, ответ чат-бота, оценку judge и объяснение judge. Ревьюер заполняет:

```text
judge_verdict_correct
judge_review_comment
reviewed_at
reviewer
```

`judge_verdict_correct` принимает значения:

```text
yes    - оценка judge корректна
no     - оценка judge некорректна
unsure - спорный случай
```
