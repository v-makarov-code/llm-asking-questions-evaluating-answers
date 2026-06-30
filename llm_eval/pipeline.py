import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from llm_eval.chat_bot_client import ChatBotClient, is_answer_filled
from llm_eval.io import (
    chatbot_output_columns,
    ensure_columns,
    require_columns,
    save_dataframe,
    score_output_columns,
)


FINAL_EXPLANATION_COLUMN = "ragas_final_explanation"


def run_chatbot_pipeline(
    df: pd.DataFrame,
    client: ChatBotClient,
    output_path: Path,
    delimiter: str = ";",
    limit: int | None = None,
    save_every: int = 1,
    delay: float = 0.0,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Fill `model_answer` for a dataframe of questions.

    The function mutates and returns `df`. It supports resumable runs through
    `skip_existing`, incremental persistence through `save_every`, and optional
    throttling through `delay`.
    """

    require_columns(df, {"question", "expected_answer"})
    ensure_columns(df, ["model_answer", "latency_sec", "created_at", "error"])

    indexes = list(range(len(df)))
    if limit is not None:
        indexes = indexes[:limit]

    processed = 0
    for position, index in enumerate(indexes, start=1):
        row = df.loc[index]
        question_id = row.get("id") or str(index + 1)

        if skip_existing and is_answer_filled(str(row.get("model_answer", ""))):
            print(f"[{position}/{len(indexes)}] {question_id}: skipped")
            continue

        question = str(row.get("question", "")).strip()
        if not question:
            df.loc[index, "model_answer"] = "failed to answer"
            df.loc[index, "error"] = "Question is empty"
            continue

        print(f"[{position}/{len(indexes)}] {question_id}")
        started_at = time.time()
        df.loc[index, "error"] = ""

        try:
            df.loc[index, "model_answer"] = client.ask(question)
        except Exception as exc:
            df.loc[index, "model_answer"] = "failed to answer"
            df.loc[index, "error"] = f"{type(exc).__name__}: {exc}"

        df.loc[index, "latency_sec"] = str(round(time.time() - started_at, 3))
        df.loc[index, "created_at"] = datetime.now().isoformat(timespec="seconds")
        processed += 1

        if save_every > 0 and processed % save_every == 0:
            save_dataframe(df, output_path, delimiter, chatbot_output_columns())

        if delay > 0:
            time.sleep(delay)

    save_dataframe(df, output_path, delimiter, chatbot_output_columns())
    return df


def run_scoring_pipeline(
    df: pd.DataFrame,
    evaluators: list[object],
    metrics: list[str],
    judge_model: str,
    output_path: Path,
    delimiter: str = ";",
    limit: int | None = None,
    max_new: int | None = None,
    save_every: int = 1,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Run selected evaluators over a dataframe of chat-bot answers.

    The pipeline keeps the existing CSV workflow resumable: `skip_existing`
    checks only columns required by selected metrics, `max_new` limits the
    number of newly processed rows, and evaluator failures are written to the
    row-level `error` column instead of stopping the whole batch.
    """

    from llm_eval.evaluators import (
        append_error,
        is_metric_filled,
        required_columns_for_metrics,
    )

    require_columns(df, {"question", "expected_answer", "model_answer"})

    metric_columns = [
        "ragas_factual_correctness",
        "ragas_semantic_similarity",
        "ragas_final_score",
        FINAL_EXPLANATION_COLUMN,
    ]
    ensure_columns(df, metric_columns)
    ensure_columns(df, ["error"])

    if any(metric in metrics for metric in ["ragas_factual_correctness", "ragas_final_score"]):
        ensure_columns(df, ["judge_model"])
        df.loc[df["judge_model"].astype(str).str.strip() == "", "judge_model"] = judge_model

    output_columns = score_output_columns(df, metrics)
    required_metric_columns = required_columns_for_metrics(metrics)

    df_to_score = df.head(limit) if limit is not None else df
    processed = 0
    for index, row in df_to_score.iterrows():
        if skip_existing and all(
            is_metric_filled(row.get(column))
            for column in required_metric_columns
        ):
            continue

        if max_new is not None and processed >= max_new:
            break

        print(f"[{processed + 1}] scoring row {index}: {row.get('id', index)}")
        start = time.time()

        for evaluator in evaluators:
            try:
                result = evaluator.evaluate(row)
                df.loc[index, result.name] = result.score
                if result.name == "ragas_final_score":
                    df.loc[index, FINAL_EXPLANATION_COLUMN] = result.explanation
            except Exception as exc:
                for column in evaluator.output_columns:
                    df.loc[index, column] = ""
                df.loc[index, "error"] = append_error(
                    df.loc[index, "error"],
                    f"{evaluator.name}_error: {type(exc).__name__}: {exc}",
                )
                print(f"  {evaluator.name} failed: {type(exc).__name__}: {exc}")

        processed += 1
        print(f"  done in {round(time.time() - start, 3)}s")

        if save_every > 0 and processed % save_every == 0:
            save_dataframe(df, output_path, delimiter, output_columns)

    save_dataframe(df, output_path, delimiter, output_columns)
    return df
