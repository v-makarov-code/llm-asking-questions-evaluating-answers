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

Основной сценарий: подготовить вопросы в `questions.csv`, прогнать их через модель скриптом `run_questions.py`, получить итоговый датасет `model_answers.csv`, а затем вручную или автоматически заполнить оценочные поля.

## Структура проекта

```text
asking_questions.ipynb  - ноутбук для экспериментов с моделью
query_RAG.ipynb         - дополнительный ноутбук для RAG-экспериментов
questions.csv           - входной датасет с вопросами
run_questions.py        - скрипт для массового прогона вопросов через модель
model_answers.csv       - итоговый датасет с ответами модели, создается после запуска
pyproject.toml          - зависимости проекта
uv.lock                 - lock-файл зависимостей
.venv/                  - локальное окружение, создается через uv sync
```

## Установка

Проект использует `uv`.

```powershell
uv sync
```

После этого можно запускать Python, Jupyter и основной скрипт через `uv run`.

## Jupyter Notebook

Запустить Jupyter Notebook:

```powershell
uv run jupyter notebook
```

Запустить JupyterLab:

```powershell
uv run jupyter lab
```

## Входной датасет questions.csv

`questions.csv` содержит вопросы, ожидаемые ответы и метаданные для оценки.

Текущие колонки:

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

Описание основных полей:

```text
id               - уникальный идентификатор вопроса
domain           - область вопроса, например data_structures или physics
source_type      - тип источника: article, general, textbook и т.д.
source_id        - короткий идентификатор источника
source_url       - ссылка на источник, если есть
question         - сам вопрос для модели
context          - контекст или краткая привязка к разделу источника
expected_answer  - эталонный ответ
question_type    - тип вопроса: factual, reasoning, comparison, calculation, application
difficulty       - сложность: easy, medium, hard
scoring_rubric   - правила оценки ответа, пока может быть пустым
```

Сейчас `questions.csv` содержит 30 вопросов по статье Habr:

```text
https://habr.com/ru/articles/879914/
```

Файл читается в pandas так:

```python
import pandas as pd

questions_df = pd.read_csv("questions.csv", encoding="utf-8")
```

## Запуск прогона вопросов

Обычный запуск:

```powershell
uv run python run_questions.py
```

Скрипт прочитает `questions.csv`, задаст каждый вопрос модели и сохранит результат в:

```text
model_answers.csv
```

Тестовый запуск только на первых 3 вопросах:

```powershell
uv run python run_questions.py --limit 3
```

Задать другой выходной файл:

```powershell
uv run python run_questions.py --output model_answers_test.csv
```

Задать другую модель:

```powershell
uv run python run_questions.py --model qwen/qwen3.5-35b-a3b
```

Задать другой endpoint:

```powershell
uv run python run_questions.py --base-url http://localhost:1234/v1
```

## Timeout и зависшие ответы

По умолчанию на один вопрос дается 300 секунд, то есть 5 минут:

```text
--request-timeout 300
```

Если модель зависла и не отвечает дольше этого времени, скрипт завершает дочерний процесс с запросом и записывает:

```text
model_answer = failed to answer
error = hard_timeout_after_300.0_seconds
```

Поставить другой timeout, например 60 секунд:

```powershell
uv run python run_questions.py --request-timeout 60
```

## Промежуточное сохранение

Параметр `--save-every` задает, как часто сохранять промежуточный результат.

По умолчанию:

```text
--save-every 1
```

Это значит, что `model_answers.csv` обновляется после каждого вопроса.

Сохранять после каждого 5-го вопроса:

```powershell
uv run python run_questions.py --save-every 5
```

В конце скрипт все равно сохраняет полный результат.

## Итоговый датасет model_answers.csv

После запуска создается `model_answers.csv`.

Он содержит исходные поля из `questions.csv` и дополнительные поля с результатом работы модели:

```text
model_answer
correctness_score
completeness_score
faithfulness_score
final_score
judge_explanation
model
temperature
latency_sec
created_at
prompt_tokens
completion_tokens
total_tokens
error
```

Описание новых полей:

```text
model_answer        - ответ модели
correctness_score   - оценка фактической правильности, пока заполняется вручную
completeness_score  - оценка полноты ответа, пока заполняется вручную
faithfulness_score  - оценка опоры на контекст, пока заполняется вручную
final_score         - итоговая оценка ответа
judge_explanation   - пояснение к оценке
model               - имя модели
temperature         - temperature при генерации
latency_sec         - время ответа в секундах
created_at          - время создания записи
prompt_tokens       - число input tokens, если сервер возвращает usage
completion_tokens   - число output tokens, если сервер возвращает usage
total_tokens        - общее число tokens, если сервер возвращает usage
error               - ошибка запроса или timeout, если был сбой
```

## Простая ручная оценка

На первом этапе достаточно заполнять только `final_score`.

Рекомендуемая простая шкала:

```text
2 - ответ правильный по сути
1 - ответ частично правильный, но есть важный пропуск или неточность
0 - ответ неверный, не по вопросу или модель не ответила
```

`judge_explanation` можно заполнять только для оценок `0` и `1`, чтобы фиксировать причину снижения оценки.

Пример:

```text
final_score = 1
judge_explanation = Ответ верный в общем, но модель не упомянула коллизии в хеш-таблице.
```

## scoring_rubric

`scoring_rubric` - это не оценка, а правило оценки.

Например:

```text
5: ответ объясняет key-value, хеш-функцию, O(1) в среднем и коллизии.
3: ответ объясняет только часть идеи.
1: ответ неверный.
```

Пока это поле можно оставить пустым. Позже его можно заполнить для более строгой ручной оценки или для LLM-as-judge.

## Работа с результатами в pandas

Загрузить итоговый датасет:

```python
import pandas as pd

results_df = pd.read_csv("model_answers.csv", encoding="utf-8-sig")
results_df.head()
```

Посчитать средний score:

```python
results_df["final_score"].mean()
```

Посмотреть качество по типам вопросов:

```python
results_df.groupby("question_type")["final_score"].mean()
```

Посмотреть зависшие или ошибочные ответы:

```python
results_df[results_df["error"].notna()]
```

## Кодировка

CSV-файлы сохраняются в UTF-8. Итоговый `model_answers.csv` сохраняется как `utf-8-sig`, чтобы его было проще открывать в Excel на Windows.

Если PowerShell показывает русский текст как набор символов вида `РџСЂ...`, это проблема отображения терминала, а не обязательно проблема файла. В pandas используйте:

```python
pd.read_csv("questions.csv", encoding="utf-8")
pd.read_csv("model_answers.csv", encoding="utf-8-sig")
```
