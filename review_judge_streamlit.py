import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_INPUT = "data/forqwen_judge_scored_qwen397b.csv"
REVIEW_COLUMNS = [
    "judge_verdict_correct",
    "judge_review_comment",
    "reviewed_at",
    "reviewer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual review UI for judge scoring results."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--reviewer", default="")
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_reviewed{input_path.suffix}")


def read_csv(path: Path, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=delimiter, encoding="utf-8-sig").fillna("")


def ensure_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in REVIEW_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")
    return df


def save_csv(df: pd.DataFrame, path: Path, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=delimiter, encoding="utf-8-sig")


def is_filled(value: object) -> bool:
    return str(value or "").strip() != ""


def filtered_indexes(df: pd.DataFrame, filter_name: str) -> list[int]:
    if filter_name == "Непроверенные":
        mask = df["judge_verdict_correct"].astype(str).str.strip() == ""
    elif filter_name == "Проверенные":
        mask = df["judge_verdict_correct"].astype(str).str.strip() != ""
    elif filter_name == "Judge correct = no":
        mask = df["judge_verdict_correct"].astype(str).str.strip() == "no"
    elif filter_name == "Judge correct = unsure":
        mask = df["judge_verdict_correct"].astype(str).str.strip() == "unsure"
    elif filter_name == "Score = 0":
        mask = df["ragas_final_score"].astype(str).str.strip() == "0"
    elif filter_name == "Score = 1":
        mask = df["ragas_final_score"].astype(str).str.strip() == "1"
    elif filter_name == "Score = 2":
        mask = df["ragas_final_score"].astype(str).str.strip() == "2"
    elif filter_name == "Error не пустой":
        mask = df["error"].astype(str).str.strip() != ""
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    return df.index[mask].tolist()


def get_text(row: pd.Series, column: str) -> str:
    return str(row.get(column, "") or "")


def render_text_area(label: str, value: str, height: int = 220) -> None:
    st.text_area(label, value=value, height=height, disabled=True)


def save_current_review(
    df: pd.DataFrame,
    row_index: int,
    verdict: str,
    comment: str,
    reviewer: str,
) -> None:
    df.loc[row_index, "judge_verdict_correct"] = verdict
    df.loc[row_index, "judge_review_comment"] = comment.strip()
    df.loc[row_index, "reviewed_at"] = datetime.now().isoformat(timespec="seconds")
    df.loc[row_index, "reviewer"] = reviewer.strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)

    st.set_page_config(
        page_title="Judge Review",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Проверка scoring judge")

    source_path = output_path if output_path.exists() else input_path
    if "df" not in st.session_state or st.session_state.get("source_path") != str(source_path):
        df = ensure_review_columns(read_csv(source_path, args.delimiter))
        st.session_state.df = df
        st.session_state.source_path = str(source_path)
        st.session_state.position = 0

    df = st.session_state.df

    with st.sidebar:
        st.subheader("Файлы")
        st.caption(f"Input: `{input_path}`")
        st.caption(f"Output: `{output_path}`")

        reviewer = st.text_input("Reviewer", value=args.reviewer)

        filter_name = st.selectbox(
            "Фильтр",
            [
                "Все",
                "Непроверенные",
                "Проверенные",
                "Judge correct = no",
                "Judge correct = unsure",
                "Score = 0",
                "Score = 1",
                "Score = 2",
                "Error не пустой",
            ],
        )

        indexes = filtered_indexes(df, filter_name)
        if not indexes:
            st.warning("Нет строк под выбранный фильтр.")
            st.stop()

        if st.session_state.position >= len(indexes):
            st.session_state.position = max(0, len(indexes) - 1)

        st.metric("Всего строк", len(df))
        st.metric("В фильтре", len(indexes))
        reviewed_count = (df["judge_verdict_correct"].astype(str).str.strip() != "").sum()
        st.metric("Проверено", int(reviewed_count))

        st.session_state.position = st.number_input(
            "Позиция в фильтре",
            min_value=1,
            max_value=len(indexes),
            value=st.session_state.position + 1,
            step=1,
        ) - 1

        if st.button("Сохранить весь CSV", use_container_width=True):
            save_csv(df, output_path, args.delimiter)
            st.success(f"Сохранено: {output_path}")

    row_index = indexes[st.session_state.position]
    row = df.loc[row_index]

    st.caption(
        f"Строка {st.session_state.position + 1} из {len(indexes)} в фильтре, "
        f"index={row_index}"
    )

    title_left, title_right = st.columns([2, 1])
    with title_left:
        st.subheader(f"{get_text(row, 'id') or row_index}")
        st.caption(f"Domain: {get_text(row, 'domain')}")
    with title_right:
        st.metric("Judge score", get_text(row, "ragas_final_score") or "empty")
        st.caption(f"Judge model: {get_text(row, 'judge_model')}")

    if is_filled(row.get("error")):
        st.error(get_text(row, "error"))

    question_col, judge_col = st.columns([2, 1])
    with question_col:
        render_text_area("Вопрос", get_text(row, "question"), height=130)
    with judge_col:
        render_text_area(
            "Объяснение judge",
            get_text(row, "ragas_final_explanation"),
            height=130,
        )

    expected_col, model_col = st.columns(2)
    with expected_col:
        render_text_area("Эталонный ответ", get_text(row, "expected_answer"), height=360)
    with model_col:
        render_text_area("Ответ чат-бота", get_text(row, "model_answer"), height=360)

    st.divider()
    st.subheader("Ручная проверка judge")

    current_verdict = get_text(row, "judge_verdict_correct")
    verdict_options = ["", "yes", "no", "unsure"]
    verdict = st.radio(
        "Вердикт judge корректен?",
        verdict_options,
        index=verdict_options.index(current_verdict)
        if current_verdict in verdict_options
        else 0,
        horizontal=True,
        format_func=lambda value: {
            "": "не выбрано",
            "yes": "yes",
            "no": "no",
            "unsure": "unsure",
        }[value],
    )

    comment = st.text_area(
        "Комментарий ревьюера",
        value=get_text(row, "judge_review_comment"),
        height=120,
    )

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 1, 2])
    with nav_col1:
        if st.button("Назад", use_container_width=True):
            st.session_state.position = max(0, st.session_state.position - 1)
            st.rerun()
    with nav_col2:
        if st.button("Сохранить", use_container_width=True):
            save_current_review(df, row_index, verdict, comment, reviewer)
            save_csv(df, output_path, args.delimiter)
            st.success("Сохранено")
    with nav_col3:
        if st.button("Дальше", use_container_width=True):
            st.session_state.position = min(len(indexes) - 1, st.session_state.position + 1)
            st.rerun()
    with nav_col4:
        if st.button("Сохранить и дальше", use_container_width=True):
            save_current_review(df, row_index, verdict, comment, reviewer)
            save_csv(df, output_path, args.delimiter)
            st.session_state.position = min(len(indexes) - 1, st.session_state.position + 1)
            st.rerun()

    with st.expander("Текущая строка как таблица"):
        st.dataframe(pd.DataFrame([df.loc[row_index]]), use_container_width=True)


if __name__ == "__main__":
    main()
