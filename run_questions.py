import argparse
import multiprocessing as mp
import queue as queue_module
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI


DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_BASE_URL = "http://192.168.15.182:1234/v1"
DEFAULT_API_KEY = "sk-no-key-required"


def build_prompt(row: pd.Series) -> str:
    context = str(row.get("context", "") or "").strip()
    question = str(row["question"]).strip()

    if context:
        return f"""Ответь на вопрос, используя контекст, если он полезен.

Контекст:
{context}

Вопрос:
{question}
""".strip()

    return f"""Ответь на вопрос.

Вопрос:
{question}
""".strip()


def ask_model(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
) -> tuple[str, dict | None]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь точно, кратко и по существу. "
                    "Если данных недостаточно, прямо скажи об этом."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        timeout=request_timeout,
    )

    usage = response.usage.model_dump() if response.usage else None
    return response.choices[0].message.content, usage


def ask_model_worker(
    queue: mp.Queue,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
) -> None:
    try:
        answer, usage = ask_model(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            request_timeout=request_timeout,
        )
        queue.put({"answer": answer, "usage": usage, "error": None})
    except Exception as exc:
        queue.put({"answer": "failed to answer", "usage": None, "error": str(exc)})


def ask_model_with_hard_timeout(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
) -> tuple[str, dict | None, str | None]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=ask_model_worker,
        args=(queue, base_url, api_key, model, prompt, temperature, request_timeout),
    )

    process.start()
    process.join(request_timeout)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        return (
            "failed to answer",
            None,
            f"hard_timeout_after_{request_timeout}_seconds",
        )

    try:
        payload = queue.get_nowait()
        return payload["answer"], payload["usage"], payload["error"]
    except queue_module.Empty:
        pass

    return (
        "failed to answer",
        None,
        f"worker_exited_without_response_exitcode_{process.exitcode}",
    )


def usage_value(usage: dict | None, key: str) -> int | None:
    if not usage:
        return None
    return usage.get(key)


def row_to_result(
    row: pd.Series,
    answer: str | None,
    model: str,
    temperature: float,
    latency_sec: float | None,
    usage: dict | None,
    error: str | None,
) -> dict:
    return {
        "id": row.get("id"),
        "domain": row.get("domain"),
        "source_type": row.get("source_type"),
        "source_id": row.get("source_id"),
        "source_url": row.get("source_url"),
        "question_type": row.get("question_type"),
        "difficulty": row.get("difficulty"),
        "question": row.get("question"),
        "context": row.get("context"),
        "expected_answer": row.get("expected_answer"),
        "scoring_rubric": row.get("scoring_rubric"),
        "model_answer": answer,
        "ragas_factual_correctness": None,
        "ragas_semantic_similarity": None,
        "ragas_final_score": None,
        "manual_final_score": None,
        "manual_comment": None,
        "model": model,
        "temperature": temperature,
        "latency_sec": latency_sec,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_tokens": usage_value(usage, "prompt_tokens"),
        "completion_tokens": usage_value(usage, "completion_tokens"),
        "total_tokens": usage_value(usage, "total_tokens"),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run questions from CSV through an OpenAI-compatible chat model."
    )
    parser.add_argument("--input", default="questions.csv", help="Path to questions CSV.")
    parser.add_argument("--output", default="model_answers.csv", help="Path to output CSV.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Max seconds to wait for one model response.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only first N questions.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save partial results after every N questions.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    questions_df = pd.read_csv(input_path, encoding="utf-8").fillna("")
    if args.limit is not None:
        questions_df = questions_df.head(args.limit)

    results: list[dict] = []

    for index, row in questions_df.iterrows():
        question_id = row.get("id", index)
        print(f"[{len(results) + 1}/{len(questions_df)}] {question_id}")

        prompt = build_prompt(row)
        start = time.time()
        answer = None
        usage = None
        error = None
        latency_sec = None

        answer, usage, error = ask_model_with_hard_timeout(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            request_timeout=args.request_timeout,
        )
        latency_sec = round(time.time() - start, 3)

        results.append(
            row_to_result(
                row=row,
                answer=answer,
                model=args.model,
                temperature=args.temperature,
                latency_sec=latency_sec,
                usage=usage,
                error=error,
            )
        )

        if args.save_every > 0 and len(results) % args.save_every == 0:
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(results_df)} rows to {output_path}")


if __name__ == "__main__":
    main()
