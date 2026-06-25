import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import instructor
from huggingface_hub import hf_hub_download, list_repo_files
from openai import AsyncOpenAI, OpenAI
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import FactualCorrectness
from transformers import AutoTokenizer


DEFAULT_BASE_URL = "http://192.168.15.182:1234/v1"
DEFAULT_API_KEY = "sk-no-key-required"
DEFAULT_JUDGE_MODEL = "gemma-4-31b-it-mlx"
DEFAULT_EMBEDDING_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ONNX_FILE = "auto"
DEFAULT_METRICS = "ragas_factual_correctness,ragas_semantic_similarity,ragas_final_score"
BASE_OUTPUT_COLUMNS = [
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
METRIC_ALIASES = {
    "factual_correctness": "ragas_factual_correctness",
    "ragas_factual_correctness": "ragas_factual_correctness",
    "semantic_similarity": "ragas_semantic_similarity",
    "ragas_semantic_similarity": "ragas_semantic_similarity",
    "final_score": "ragas_final_score",
    "ragas_final_score": "ragas_final_score",
}
FINAL_EXPLANATION_COLUMN = "ragas_final_explanation"


class OnnxSentenceEmbedder:
    def __init__(
        self,
        model_repo: str,
        onnx_file: str,
        cache_dir: str | None,
        max_length: int,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, cache_dir=cache_dir)
        resolved_onnx_file = resolve_onnx_file(model_repo, onnx_file)
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=resolved_onnx_file,
            cache_dir=cache_dir,
        )
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self.max_length = max_length

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        inputs = {
            key: value
            for key, value in batch.items()
            if key in {item.name for item in self.session.get_inputs()}
        }
        required_inputs = {item.name for item in self.session.get_inputs()}
        if "token_type_ids" in required_inputs and "token_type_ids" not in inputs:
            inputs["token_type_ids"] = np.zeros_like(batch["input_ids"])
        outputs = self.session.run(None, inputs)
        token_embeddings = outputs[0]
        attention_mask = batch["attention_mask"]
        embeddings = self._mean_pool(token_embeddings, attention_mask)
        return self._normalize(embeddings)

    @staticmethod
    def _mean_pool(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        return embeddings / norms


def resolve_onnx_file(model_repo: str, onnx_file: str) -> str:
    if onnx_file != "auto":
        return onnx_file

    files = list_repo_files(model_repo)
    onnx_files = [file for file in files if file.startswith("onnx/") and file.endswith(".onnx")]
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in Hugging Face repo: {model_repo}")

    preferred_markers = [
        "qint8_avx2",
        "quint8_avx2",
        "quantized",
        "int8",
        "model.onnx",
    ]
    for marker in preferred_markers:
        for file in onnx_files:
            if marker in file:
                return file

    return sorted(onnx_files)[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("JSON\n", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def clamp(value: float, minimum: float, maximum: float) -> float:
    if math.isnan(value):
        return minimum
    return max(minimum, min(maximum, value))


def ask_judge_json(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты строгий оценщик ответов. Возвращай только валидный JSON "
                    "без markdown и без поясняющего текста вне JSON. "
                    "Поле explanation всегда пиши на русском языке."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


def final_score_prompt(question: str, reference: str, response: str) -> str:
    return f"""
Оцени ответ модели по шкале 0/1/2.

Верни только валидный JSON строго такого вида:
{{"score": 2, "explanation": "Короткая причина выбранной оценки."}}

Шкала:
2 - ответ полностью правильный
1 - ответ частично правильный
0 - ответ неправильный

Поле explanation обязательно пиши на русском языке.
Объяснение должно быть коротким: 1 предложение, максимум 2 предложения.

Вопрос:
{question}

Эталонный ответ:
{reference}

Ответ модели:
{response}
""".strip()

def score_factual_correctness(
    metric: FactualCorrectness,
    reference: str,
    response: str,
) -> float:
    result = metric.score(response=response, reference=reference)
    return round(float(result.value), 4)


def score_final(
    client: OpenAI,
    model: str,
    question: str,
    reference: str,
    response: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
) -> tuple[int, str]:
    payload = ask_judge_json(
        client=client,
        model=model,
        prompt=final_score_prompt(question, reference, response),
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
    )
    score = int(clamp(float(payload["score"]), 0.0, 2.0))
    explanation = str(payload.get("explanation", "")).strip()
    return score, explanation


def is_filled(value: object) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def parse_metrics(value: str) -> list[str]:
    metrics: list[str] = []
    for raw_metric in value.split(","):
        metric = raw_metric.strip()
        if not metric:
            continue
        canonical_metric = METRIC_ALIASES.get(metric)
        if canonical_metric is None:
            allowed = ", ".join(sorted(METRIC_ALIASES))
            raise ValueError(f"Unknown metric '{metric}'. Allowed metrics: {allowed}")
        if canonical_metric not in metrics:
            metrics.append(canonical_metric)
    if not metrics:
        raise ValueError("At least one metric must be selected.")
    return metrics


def required_columns_for_metrics(metrics: list[str]) -> list[str]:
    columns = list(metrics)
    if "ragas_final_score" in metrics:
        columns.append(FINAL_EXPLANATION_COLUMN)
    return columns


def uses_judge_model(metrics: list[str]) -> bool:
    return any(
        metric in metrics
        for metric in ["ragas_factual_correctness", "ragas_final_score"]
    )


def output_columns_for_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    columns: list[str] = []
    for column in BASE_OUTPUT_COLUMNS:
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


def save(
    df: pd.DataFrame,
    output_path: Path,
    delimiter: str,
    output_columns: list[str],
) -> None:
    for column in output_columns:
        if column not in df.columns:
            df[column] = ""
    df.loc[:, output_columns].to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        sep=delimiter,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill ragas_* metric columns for a model answers CSV."
    )
    parser.add_argument("--input", default="qwenqwen3535b_answers.csv")
    parser.add_argument("--output", default="qwenqwen3535b_answers_ragas.csv")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_REPO)
    parser.add_argument("--embedding-onnx-file", default=DEFAULT_ONNX_FILE)
    parser.add_argument("--embedding-cache-dir", default=".hf-cache")
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=(
            "Comma-separated metrics to fill. Allowed: factual_correctness, "
            "semantic_similarity, final_score, or full ragas_* column names."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
        help="Process at most N non-skipped rows.",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows where all selected metric columns are already filled.",
    )
    args = parser.parse_args()
    selected_metrics = parse_metrics(args.metrics)
    required_metric_columns = required_columns_for_metrics(selected_metrics)

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        sep=args.delimiter,
    ).fillna("")

    required_input_columns = {"question", "expected_answer", "model_answer"}
    missing_input_columns = required_input_columns.difference(df.columns)
    if missing_input_columns:
        raise ValueError(
            f"Missing required input columns: {sorted(missing_input_columns)}"
        )

    for column in [
        "ragas_factual_correctness",
        "ragas_semantic_similarity",
        "ragas_final_score",
        FINAL_EXPLANATION_COLUMN,
    ]:
        if column not in df.columns:
            df[column] = ""

    if uses_judge_model(selected_metrics):
        if "judge_model" not in df.columns:
            df["judge_model"] = ""
        df.loc[df["judge_model"].astype(str).str.strip() == "", "judge_model"] = (
            args.judge_model
        )

    output_columns = output_columns_for_metrics(df, selected_metrics)

    if args.limit is not None:
        df_to_score = df.head(args.limit)
    else:
        df_to_score = df

    judge_client = None
    if "ragas_final_score" in selected_metrics:
        judge_client = OpenAI(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout=args.request_timeout,
        )

    factual_correctness_metric = None
    if "ragas_factual_correctness" in selected_metrics:
        ragas_judge_client = AsyncOpenAI(
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            timeout=args.request_timeout,
        )
        instructor_client = instructor.from_openai(
            ragas_judge_client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        ragas_judge_llm = InstructorLLM(
            client=instructor_client,
            model=args.judge_model,
            provider="openai",
            model_args=InstructorModelArgs(
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
            ),
        )
        factual_correctness_metric = FactualCorrectness(
            llm=ragas_judge_llm,
            mode="f1",
        )

    embedder = None
    if "ragas_semantic_similarity" in selected_metrics:
        embedder = OnnxSentenceEmbedder(
            model_repo=args.embedding_model,
            onnx_file=args.embedding_onnx_file,
            cache_dir=args.embedding_cache_dir,
            max_length=args.embedding_max_length,
        )

    processed = 0
    for index, row in df_to_score.iterrows():
        if args.skip_existing and all(
            is_filled(row.get(column))
            for column in required_metric_columns
        ):
            continue

        if args.max_new is not None and processed >= args.max_new:
            break

        question = str(row.get("question", "")).strip()
        reference = str(row.get("expected_answer", "")).strip()
        response = str(row.get("model_answer", "")).strip()

        print(f"[{processed + 1}] scoring row {index}: {row.get('id', index)}")
        start = time.time()

        if not reference or not response or response == "failed to answer":
            if "ragas_factual_correctness" in selected_metrics:
                df.loc[index, "ragas_factual_correctness"] = 0.0
            if "ragas_semantic_similarity" in selected_metrics:
                df.loc[index, "ragas_semantic_similarity"] = 0.0
            if "ragas_final_score" in selected_metrics:
                df.loc[index, "ragas_final_score"] = 0
                df.loc[index, FINAL_EXPLANATION_COLUMN] = (
                    "Ответ отсутствует или помечен как failed to answer."
                )
        else:
            if "ragas_semantic_similarity" in selected_metrics:
                assert embedder is not None
                embeddings = embedder.encode([reference, response])
                df.loc[index, "ragas_semantic_similarity"] = round(
                    cosine_similarity(embeddings[0], embeddings[1]),
                    4,
                )
            if "ragas_factual_correctness" in selected_metrics:
                assert factual_correctness_metric is not None
                df.loc[index, "ragas_factual_correctness"] = score_factual_correctness(
                    metric=factual_correctness_metric,
                    reference=reference,
                    response=response,
                )
            if "ragas_final_score" in selected_metrics:
                assert judge_client is not None
                final_score, final_explanation = score_final(
                    client=judge_client,
                    model=args.judge_model,
                    question=question,
                    reference=reference,
                    response=response,
                    temperature=args.judge_temperature,
                    timeout=args.request_timeout,
                    max_tokens=args.judge_max_tokens,
                )
                df.loc[index, "ragas_final_score"] = final_score
                df.loc[index, FINAL_EXPLANATION_COLUMN] = final_explanation

        processed += 1
        print(f"  done in {round(time.time() - start, 3)}s")

        if args.save_every > 0 and processed % args.save_every == 0:
            save(df, output_path, args.delimiter, output_columns)

    save(df, output_path, args.delimiter, output_columns)
    print(f"Saved scored CSV to {output_path}")


if __name__ == "__main__":
    main()
