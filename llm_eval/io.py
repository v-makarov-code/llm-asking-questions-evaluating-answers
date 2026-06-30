from pathlib import Path

import pandas as pd


CHATBOT_OUTPUT_COLUMNS = [
    "id",
    "domain",
    "question",
    "context",
    "expected_answer",
    "model_answer",
    "manual_final_score",
    "manual_comment",
    "latency_sec",
    "created_at",
    "error",
]

BASE_SCORE_OUTPUT_COLUMNS = [
    "id",
    "domain",
    "question",
    "context",
    "expected_answer",
    "model_answer",
]

MANUAL_OUTPUT_COLUMNS = [
    "manual_final_score",
    "manual_comment",
]


def read_dataframe(path: Path, delimiter: str = ";") -> pd.DataFrame:
    """Read a semicolon-separated UTF-8 CSV and replace missing values with blanks."""

    return pd.read_csv(path, encoding="utf-8-sig", sep=delimiter).fillna("")


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    delimiter: str = ";",
    columns: list[str] | None = None,
) -> None:
    """Save a dataframe as UTF-8 CSV, optionally restricting it to known columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        for column in columns:
            if column not in df.columns:
                df[column] = ""
        df = df.loc[:, columns]
    df.to_csv(path, index=False, encoding="utf-8-sig", sep=delimiter)


def require_columns(df: pd.DataFrame, columns: set[str]) -> None:
    """Fail early when the input dataset does not contain required columns."""

    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create missing columns and cast them to object so pandas accepts mixed values."""

    for column in columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")
    return df


def chatbot_output_columns() -> list[str]:
    """Return the intentionally compact output schema for chat-bot answers."""

    return list(CHATBOT_OUTPUT_COLUMNS)


def uses_judge_model(metrics: list[str]) -> bool:
    return any(
        metric in metrics
        for metric in ["ragas_factual_correctness", "ragas_final_score"]
    )


def score_output_columns(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    """Build the scored CSV schema from selected metrics.

    The output keeps only analysis-relevant columns. For example, `judge_model`
    is included only when a selected metric actually uses a judge LLM.
    """

    from llm_eval.evaluators import required_columns_for_metrics

    columns: list[str] = []
    for column in BASE_SCORE_OUTPUT_COLUMNS:
        if column in df.columns:
            columns.append(column)

    columns.extend(required_columns_for_metrics(metrics))

    for column in MANUAL_OUTPUT_COLUMNS:
        if column in df.columns:
            columns.append(column)

    if uses_judge_model(metrics):
        columns.append("judge_model")

    if "error" in df.columns:
        columns.append("error")

    return list(dict.fromkeys(columns))
