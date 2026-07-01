import json
import math
from abc import ABC, abstractmethod

import instructor
import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from openai import AsyncOpenAI, OpenAI
from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.metrics.collections import FactualCorrectness
from transformers import AutoTokenizer

from llm_eval.models import MetricResult


DEFAULT_BASE_URL = "http://192.168.15.182:1234/v1"
DEFAULT_API_KEY = "sk-no-key-required"
DEFAULT_JUDGE_MODEL = "gemma-4-31b-it-mlx"
DEFAULT_EMBEDDING_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ONNX_FILE = "auto"
DEFAULT_METRICS = "ragas_factual_correctness,ragas_semantic_similarity,ragas_final_score"
FINAL_EXPLANATION_COLUMN = "ragas_final_explanation"

METRIC_ALIASES = {
    "factual_correctness": "ragas_factual_correctness",
    "ragas_factual_correctness": "ragas_factual_correctness",
    "semantic_similarity": "ragas_semantic_similarity",
    "ragas_semantic_similarity": "ragas_semantic_similarity",
    "final_score": "ragas_final_score",
    "ragas_final_score": "ragas_final_score",
}


class BaseEvaluator(ABC):
    """Base contract for all row-level evaluators.

    An evaluator receives one dataframe row and returns a `MetricResult`. The
    batch pipeline decides how to persist the returned value into CSV columns.
    """

    name: str
    output_columns: list[str]

    @abstractmethod
    def evaluate(self, row: pd.Series) -> MetricResult:
        raise NotImplementedError


class OnnxSentenceEmbedder:
    """Small ONNX Runtime sentence embedder used for semantic similarity."""

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


class SemanticSimilarityEvaluator(BaseEvaluator):
    """Compute cosine similarity between expected and generated answers.

    This metric uses local ONNX embeddings and does not call a judge LLM.
    """

    name = "ragas_semantic_similarity"
    output_columns = ["ragas_semantic_similarity"]

    def __init__(
        self,
        model_repo: str = DEFAULT_EMBEDDING_REPO,
        onnx_file: str = DEFAULT_ONNX_FILE,
        cache_dir: str | None = ".hf-cache",
        max_length: int = 256,
    ) -> None:
        self.embedder = OnnxSentenceEmbedder(
            model_repo=model_repo,
            onnx_file=onnx_file,
            cache_dir=cache_dir,
            max_length=max_length,
        )

    def evaluate(self, row: pd.Series) -> MetricResult:
        reference = str(row.get("expected_answer", "")).strip()
        response = str(row.get("model_answer", "")).strip()
        if not reference or not response or response == "failed to answer":
            return MetricResult(self.name, 0.0)

        embeddings = self.embedder.encode([reference, response])
        score = round(float(np.dot(embeddings[0], embeddings[1])), 4)
        return MetricResult(self.name, score)


class FactualCorrectnessEvaluator(BaseEvaluator):
    """Compute Ragas FactualCorrectness against the expected answer.

    Ragas decomposes reference/response into claims and uses the configured
    judge LLM for claim verification. The stored value is the Ragas F1 score.
    """

    name = "ragas_factual_correctness"
    output_columns = ["ragas_factual_correctness"]

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        timeout: float = 300.0,
        max_tokens: int = 8192,
    ) -> None:
        ragas_judge_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        instructor_client = instructor.from_openai(
            ragas_judge_client,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        ragas_judge_llm = InstructorLLM(
            client=instructor_client,
            model=model,
            provider="openai",
            model_args=InstructorModelArgs(
                temperature=temperature,
                max_tokens=max_tokens,
            ),
        )
        self.metric = FactualCorrectness(llm=ragas_judge_llm, mode="f1")

    def evaluate(self, row: pd.Series) -> MetricResult:
        reference = str(row.get("expected_answer", "")).strip()
        response = str(row.get("model_answer", "")).strip()
        if not reference or not response or response == "failed to answer":
            return MetricResult(self.name, 0.0)

        result = self.metric.score(response=response, reference=reference)
        return MetricResult(self.name, round(float(result.value), 4))


class FinalScoreEvaluator(BaseEvaluator):
    """Custom LLM-as-a-judge evaluator for the 0/1/2 final score.

    All judge models are called with JSON schema structured output. Some
    reasoning models return the JSON in `reasoning_content`, so the response
    reader falls back to that field when `content` is empty.
    """

    name = "ragas_final_score"
    output_columns = ["ragas_final_score", FINAL_EXPLANATION_COLUMN]

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_JUDGE_MODEL,
        temperature: float = 0.0,
        timeout: float = 300.0,
        max_tokens: int = 8192,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def evaluate(self, row: pd.Series) -> MetricResult:
        question = str(row.get("question", "")).strip()
        reference = str(row.get("expected_answer", "")).strip()
        response = str(row.get("model_answer", "")).strip()
        if not reference or not response or response == "failed to answer":
            return MetricResult(
                self.name,
                0,
                "Ответ отсутствует или помечен как failed to answer.",
            )

        score, explanation = self._score(question, reference, response)
        return MetricResult(self.name, score, explanation)

    def _score(self, question: str, reference: str, response: str) -> tuple[int, str]:
        payload = ask_judge_json_schema(
            client=self.client,
            model=self.model,
            prompt=final_score_json_prompt(question, reference, response),
            temperature=self.temperature,
            timeout=self.timeout,
            max_tokens=self.max_tokens,
        )
        score = int(clamp(float(payload["score"]), 0.0, 2.0))
        explanation = str(payload.get("explanation", "")).strip()
        return score, explanation


def build_evaluators(metrics: list[str], args) -> list[BaseEvaluator]:
    """Instantiate evaluator objects for selected metric names."""

    evaluators: list[BaseEvaluator] = []
    if "ragas_semantic_similarity" in metrics:
        evaluators.append(
            SemanticSimilarityEvaluator(
                model_repo=args.embedding_model,
                onnx_file=args.embedding_onnx_file,
                cache_dir=args.embedding_cache_dir,
                max_length=args.embedding_max_length,
            )
        )
    if "ragas_factual_correctness" in metrics:
        evaluators.append(
            FactualCorrectnessEvaluator(
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                model=args.judge_model,
                temperature=args.judge_temperature,
                timeout=args.request_timeout,
                max_tokens=args.judge_max_tokens,
            )
        )
    if "ragas_final_score" in metrics:
        evaluators.append(
            FinalScoreEvaluator(
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
                model=args.judge_model,
                temperature=args.judge_temperature,
                timeout=args.request_timeout,
                max_tokens=args.judge_max_tokens,
            )
        )
    return evaluators


def resolve_onnx_file(model_repo: str, onnx_file: str) -> str:
    """Resolve an ONNX file from a Hugging Face repo.

    `auto` prefers quantized/AVX2 models when available because they are smaller
    and faster for local CPU inference.
    """

    if onnx_file != "auto":
        return onnx_file

    files = list_repo_files(model_repo)
    onnx_files = [file for file in files if file.startswith("onnx/") and file.endswith(".onnx")]
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in Hugging Face repo: {model_repo}")

    preferred_markers = ["qint8_avx2", "quint8_avx2", "quantized", "int8", "model.onnx"]
    for marker in preferred_markers:
        for file in onnx_files:
            if marker in file:
                return file

    return sorted(onnx_files)[0]


def parse_metrics(value: str) -> list[str]:
    """Parse CLI metric aliases into canonical output column names."""

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
    """Return columns that must be filled for `--skip-existing`.

    `ragas_final_score` requires both the numeric score and its explanation.
    """

    columns = list(metrics)
    if "ragas_final_score" in metrics:
        columns.append(FINAL_EXPLANATION_COLUMN)
    return columns


def is_metric_filled(value: object) -> bool:
    """Check whether a metric cell contains a meaningful value."""

    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def append_error(existing: object, message: str) -> str:
    """Append a new error message while preserving any previous row error."""

    existing_text = "" if existing is None or pd.isna(existing) else str(existing).strip()
    if not existing_text:
        return message
    return f"{existing_text} | {message}"


def ask_judge_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    response_format: dict | None = None,
) -> str:
    """Call an OpenAI-compatible chat endpoint and return raw message text.

    Most models put the final answer into `message.content`. Some reasoning
    models return structured output in a non-standard `reasoning_content` field
    while leaving `content` empty, so that field is used as a fallback.
    """

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "timeout": timeout,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    content = message.content or ""
    if content:
        return content

    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        return str(reasoning_content)

    if hasattr(message, "model_dump"):
        message_data = message.model_dump()
        reasoning_content = message_data.get("reasoning_content")
        if reasoning_content:
            return str(reasoning_content)

    return ""


def ask_judge_json_schema(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
) -> dict:
    """Call a judge model with JSON schema structured output."""

    content = ask_judge_text(
        client=client,
        model=model,
        system_prompt=(
            "Ты строгий оценщик ответов. Возвращай только валидный JSON "
            "без markdown и без поясняющего текста вне JSON. "
            "Поле explanation всегда пиши на русском языке."
        ),
        prompt=prompt,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_score",
                "schema": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "enum": [0, 1, 2]},
                        "explanation": {"type": "string"},
                    },
                    "required": ["score", "explanation"],
                    "additionalProperties": False,
                },
            },
        },
    )
    return parse_json_object(content)


def final_score_json_prompt(question: str, reference: str, response: str) -> str:
    """Build the final-score prompt for models with structured JSON output."""

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


def parse_json_object(text: str) -> dict:
    """Extract and parse a JSON object from model output."""

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
    """Clamp a numeric value and map NaN to the minimum."""

    if math.isnan(value):
        return minimum
    return max(minimum, min(maximum, value))
