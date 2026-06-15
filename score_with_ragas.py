import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI
from transformers import AutoTokenizer


DEFAULT_BASE_URL = "http://192.168.15.182:1234/v1"
DEFAULT_API_KEY = "sk-no-key-required"
DEFAULT_JUDGE_MODEL = "gemma-4-31b-it-mlx"
DEFAULT_EMBEDDING_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ONNX_FILE = "auto"


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
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты строгий оценщик ответов. Возвращай только валидный JSON "
                    "без markdown и без поясняющего текста."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        timeout=timeout,
    )
    content = response.choices[0].message.content or "{}"
    return parse_json_object(content)


def factual_correctness_prompt(question: str, reference: str, response: str) -> str:
    return f"""
Оцени фактическую корректность ответа модели относительно эталонного ответа.

Верни JSON строго такого вида:
{{"score": 0.0}}

Где score от 0 до 1:
1.0 - ответ фактически полностью корректен
0.5 - ответ частично корректен
0.0 - ответ фактически неверен

Вопрос:
{question}

Эталонный ответ:
{reference}

Ответ модели:
{response}
""".strip()


def final_score_prompt(question: str, reference: str, response: str) -> str:
    return f"""
Оцени ответ модели по шкале 0/1/2.

Верни JSON строго такого вида:
{{"score": 2}}

Шкала:
2 - ответ полностью правильный
1 - ответ частично правильный
0 - ответ неправильный

Вопрос:
{question}

Эталонный ответ:
{reference}

Ответ модели:
{response}
""".strip()


def score_factual_correctness(
    client: OpenAI,
    model: str,
    question: str,
    reference: str,
    response: str,
    temperature: float,
    timeout: float,
) -> float:
    payload = ask_judge_json(
        client=client,
        model=model,
        prompt=factual_correctness_prompt(question, reference, response),
        temperature=temperature,
        timeout=timeout,
    )
    return round(clamp(float(payload["score"]), 0.0, 1.0), 4)


def score_final(
    client: OpenAI,
    model: str,
    question: str,
    reference: str,
    response: str,
    temperature: float,
    timeout: float,
) -> int:
    payload = ask_judge_json(
        client=client,
        model=model,
        prompt=final_score_prompt(question, reference, response),
        temperature=temperature,
        timeout=timeout,
    )
    return int(clamp(float(payload["score"]), 0.0, 2.0))


def is_filled(value: object) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def save(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill ragas_* metric columns for a model answers CSV."
    )
    parser.add_argument("--input", default="qwenqwen3535b_answers.csv")
    parser.add_argument("--output", default="qwenqwen3535b_answers_ragas.csv")
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_REPO)
    parser.add_argument("--embedding-onnx-file", default=DEFAULT_ONNX_FILE)
    parser.add_argument("--embedding-cache-dir", default=".hf-cache")
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows where all three ragas_* metric columns are already filled.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(input_path, encoding="utf-8-sig").fillna("")

    for column in [
        "ragas_factual_correctness",
        "ragas_semantic_similarity",
        "ragas_final_score",
    ]:
        if column not in df.columns:
            df[column] = ""

    if args.limit is not None:
        df_to_score = df.head(args.limit)
    else:
        df_to_score = df

    judge_client = OpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key)
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
            for column in [
                "ragas_factual_correctness",
                "ragas_semantic_similarity",
                "ragas_final_score",
            ]
        ):
            continue

        question = str(row.get("question", "")).strip()
        reference = str(row.get("expected_answer", "")).strip()
        response = str(row.get("model_answer", "")).strip()

        print(f"[{processed + 1}] scoring row {index}: {row.get('id', index)}")
        start = time.time()

        if not reference or not response or response == "failed to answer":
            df.loc[index, "ragas_factual_correctness"] = 0.0
            df.loc[index, "ragas_semantic_similarity"] = 0.0
            df.loc[index, "ragas_final_score"] = 0
        else:
            embeddings = embedder.encode([reference, response])
            df.loc[index, "ragas_semantic_similarity"] = round(
                cosine_similarity(embeddings[0], embeddings[1]),
                4,
            )
            df.loc[index, "ragas_factual_correctness"] = score_factual_correctness(
                client=judge_client,
                model=args.judge_model,
                question=question,
                reference=reference,
                response=response,
                temperature=args.judge_temperature,
                timeout=args.request_timeout,
            )
            df.loc[index, "ragas_final_score"] = score_final(
                client=judge_client,
                model=args.judge_model,
                question=question,
                reference=reference,
                response=response,
                temperature=args.judge_temperature,
                timeout=args.request_timeout,
            )

        processed += 1
        print(f"  done in {round(time.time() - start, 3)}s")

        if args.save_every > 0 and processed % args.save_every == 0:
            save(df, output_path)

    save(df, output_path)
    print(f"Saved scored CSV to {output_path}")


if __name__ == "__main__":
    main()
