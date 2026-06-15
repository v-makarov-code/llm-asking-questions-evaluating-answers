# asking-questions-to-llm

Проект для проверки качества ответов OpenAI-compatible LLM на заранее подготовленном наборе вопросов.

Сейчас проект настроен под модель:

```text
qwen/qwen3.5-35b-a3b
```

и локальный endpoint:

```text
http://192.168.15.182:1234/v1
```

Основной сценарий: подготовить вопросы в `questions.csv`, прогнать их через модель скриптом `run_questions.py`, получить `model_answers.csv`, затем отдельно заполнить автоматические Ragas-метрики и ручную оценку.

## Структура

```text
asking_questions.ipynb  - ноутбук для экспериментов с моделью
query_RAG.ipynb         - дополнительный ноутбук для RAG-экспериментов
questions.csv           - входной датасет с вопросами
run_questions.py        - скрипт для прогона вопросов через модель
model_answers.csv       - итоговый датасет с ответами модели, создается после запуска
pyproject.toml          - зависимости проекта
uv.lock                 - lock-файл зависимостей
.venv/                  - локальное окружение, создается через uv sync
```

## Установка

```powershell
uv sync
```

## Jupyter

Запустить Jupyter Notebook:

```powershell
uv run jupyter notebook
```

Запустить JupyterLab:

```powershell
uv run jupyter lab
```

## Входной датасет questions.csv

`questions.csv` содержит вопросы, ожидаемые ответы и метаданные.

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
domain           - область вопроса, например data_structures или physics
source_type      - тип источника: article, general, textbook
source_id        - короткий идентификатор источника
source_url       - ссылка на источник
question         - вопрос для модели
context          - контекст или привязка к разделу источника
expected_answer  - эталонный ответ
question_type    - тип вопроса: factual, reasoning, comparison, calculation, application
difficulty       - сложность: easy, medium, hard
scoring_rubric   - правила оценки, пока может быть пустым
```

Загрузка в pandas:

```python
import pandas as pd

questions_df = pd.read_csv("questions.csv", encoding="utf-8")
```

## Запуск вопросов

Обычный запуск:

```powershell
uv run python run_questions.py
```

Тестовый запуск на первых 3 вопросах:

```powershell
uv run python run_questions.py --limit 3
```

Задать другой выходной файл:

```powershell
uv run python run_questions.py --output model_answers_test.csv
```

Задать другой endpoint:

```powershell
uv run python run_questions.py --base-url http://localhost:1234/v1
```

Задать другую модель:

```powershell
uv run python run_questions.py --model qwen/qwen3.5-35b-a3b
```

## Qwen no-think режим

Для Qwen-моделей скрипт по умолчанию добавляет `/no_think` в конец user prompt, чтобы отключить длинные reasoning-блоки и уменьшить расход tokens.

No-think включается автоматически для моделей:

```text
qwen3.5-vl-122b-a10b-mlx-crack
qwen/qwen3.5-35b-a3b
qwen3.5-397b-a17b
```

Для `gemma-4-31b-it-mlx` и других моделей `/no_think` не добавляется.

## Timeout

На один вопрос по умолчанию дается 300 секунд:

```text
--request-timeout 300
```

Если модель зависла дольше timeout, скрипт завершает дочерний процесс и записывает:

```text
model_answer = failed to answer
error = hard_timeout_after_300.0_seconds
```

Пример с timeout 60 секунд:

```powershell
uv run python run_questions.py --request-timeout 60
```

## Промежуточное сохранение

`--save-every` задает, как часто сохранять промежуточный результат.

По умолчанию файл сохраняется после каждого вопроса:

```text
--save-every 1
```

Сохранять после каждого 5-го вопроса:

```powershell
uv run python run_questions.py --save-every 5
```

## Итоговый датасет model_answers.csv

`run_questions.py` создает `model_answers.csv`. В него попадают исходные поля из `questions.csv`, ответ модели, технические поля запуска и пустые колонки для последующей оценки.

Основные итоговые колонки:

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

Описание оценочных колонок:

```text
ragas_factual_correctness  - автоматическая Ragas-оценка фактической правильности
ragas_semantic_similarity  - автоматическая Ragas-оценка смысловой близости к expected_answer
ragas_final_score          - автоматическая Ragas-оценка по rubric 0/1/2
manual_final_score         - твоя ручная итоговая оценка 0/1/2
manual_comment             - твой ручной комментарий к оценке
```

После запуска `run_questions.py` Ragas-поля и ручные поля остаются пустыми. Скрипт на этом этапе только собирает ответы модели. Оценки заполняются следующим отдельным шагом.

## Выбранная схема оценки

Берем три автоматические оценки через Ragas:

```text
Factual Correctness
Semantic Similarity
Ragas final score по rubric
```

Rubric для `ragas_final_score`:

```text
2 - ответ полностью правильный
1 - ответ частично правильный
0 - ответ неправильный
```

Также оставляем ручную оценку:

```text
manual_final_score
manual_comment
```

Рекомендуемая ручная шкала такая же:

```text
2 - ответ правильный по сути
1 - ответ частично правильный, но есть важный пропуск или неточность
0 - ответ неверный, не по вопросу или модель не ответила
```

## Что пока не используем

Пока не используем:

```text
completeness_score
faithfulness_score
judge_explanation
```

Причины:

```text
completeness_score  - в Ragas нет простой универсальной completeness-метрики под этот проект
faithfulness_score  - имеет смысл только при наличии полноценного context из источника
judge_explanation   - не берем, чтобы сначала оставить датасет проще
```

## Работа с результатами в pandas

Загрузить итоговый датасет:

```python
import pandas as pd

results_df = pd.read_csv("model_answers.csv", encoding="utf-8-sig")
results_df.head()
```

Посчитать среднюю ручную оценку:

```python
results_df["manual_final_score"].mean()
```

Посмотреть качество по типам вопросов:

```python
results_df.groupby("question_type")["manual_final_score"].mean()
```

Посмотреть ошибки и timeout:

```python
results_df[results_df["error"].notna()]
```

Сравнить ручную оценку и Ragas final score:

```python
results_df[["id", "ragas_final_score", "manual_final_score"]]
```

## Заполнение Ragas-метрик

После получения CSV с ответами модели можно заполнить автоматические метрики отдельным скриптом:

```powershell
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas.csv `
  --judge-model gemma-4-31b-it-mlx
```

Если старая `.venv` заблокирована Windows, используйте Python 3.12 окружение `.venv312`:

```powershell
$env:UV_PROJECT_ENVIRONMENT="C:\Users\v.makarov\projects\asking_questions_to_LLM\.venv312"
uv run python score_with_ragas.py `
  --input qwenqwen3535b_answers.csv `
  --output qwenqwen3535b_answers_ragas.csv `
  --judge-model gemma-4-31b-it-mlx
```

Скрипт заполняет:

```text
ragas_factual_correctness
ragas_semantic_similarity
ragas_final_score
```

Для LLM-as-judge метрик используется локальный OpenAI-compatible endpoint:

```text
--judge-base-url http://192.168.15.182:1234/v1
--judge-api-key sk-no-key-required
--judge-model gemma-4-31b-it-mlx
```

Для `ragas_semantic_similarity` используется локальный ONNX embedder:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

По умолчанию берется ONNX-файл:

```text
onnx/model_quint8_avx2.onnx
```

Первый запуск скачает embedding-модель в локальный cache:

```text
.hf-cache/
```

Полезные параметры:

```text
--limit 3                 - проверить только первые 3 строки
--save-every 1            - сохранять результат после каждой строки
--skip-existing           - пропускать строки, где все ragas_* поля уже заполнены
--request-timeout 300     - timeout для одного judge-запроса
```

## Кодировка

CSV-файлы сохраняются в UTF-8. Итоговый `model_answers.csv` сохраняется как `utf-8-sig`, чтобы его было проще открывать в Excel на Windows.

Если PowerShell показывает русский текст как `РџСЂ...`, это проблема отображения терминала. В pandas используйте:

```python
pd.read_csv("questions.csv", encoding="utf-8")
pd.read_csv("model_answers.csv", encoding="utf-8-sig")
```
