import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


DEFAULT_URL = "https://ai.sapiens.solutions/api/v1/conversations/ask/stream"
DEFAULT_INPUT = "RAG_questions_answers.csv"
DEFAULT_OUTPUT = "AI_chat_bot_answers.csv"
DEFAULT_RETRY_DELAYS = (2.0, 5.0, 10.0)


def ask_chat_bot_once(
    url: str,
    auth_token: str,
    question: str,
    timeout: float,
) -> str:
    with requests.Session() as session:
        with session.post(
            url,
            headers={
                "Authorization": f"Token {auth_token}",
                "Content-Type": "application/json",
                "Connection": "close",
            },
            json={"query": question},
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            response.encoding = "utf-8"

            answer_parts: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue

                payload = line[6:].strip()
                if payload == "[DONE]":
                    break

                event = json.loads(payload)
                if event.get("type") == "text":
                    answer_parts.append(event.get("data", {}).get("delta", ""))
                elif event.get("type") == "error":
                    raise RuntimeError(str(event.get("data", event)))

    answer = "".join(answer_parts).strip()
    if not answer:
        raise RuntimeError("Chat bot returned an empty answer")
    return answer


def should_retry(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        status_code = exc.response.status_code if exc.response is not None else None
        return status_code in {408, 425, 429, 500, 502, 503, 504}

    return isinstance(
        exc,
        (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.SSLError,
            json.JSONDecodeError,
            RuntimeError,
        ),
    )


def ask_chat_bot(
    url: str,
    auth_token: str,
    question: str,
    timeout: float,
    retry_delays: tuple[float, ...],
) -> str:
    attempts = len(retry_delays) + 1
    for attempt in range(attempts):
        try:
            return ask_chat_bot_once(
                url=url,
                auth_token=auth_token,
                question=question,
                timeout=timeout,
            )
        except Exception as exc:
            is_last_attempt = attempt == attempts - 1
            if is_last_attempt or not should_retry(exc):
                raise

            delay = retry_delays[attempt]
            print(
                f"  attempt {attempt + 1}/{attempts} failed: "
                f"{type(exc).__name__}; retry in {delay:g}s"
            )
            time.sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")


def read_csv(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def save_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
    delimiter: str,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter=delimiter,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def is_filled(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().casefold()
    return bool(normalized and normalized != "failed to answer")


def parse_retry_delays(value: str) -> tuple[float, ...]:
    delays = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if any(delay < 0 for delay in delays):
        raise argparse.ArgumentTypeError("Retry delays must be non-negative")
    return delays


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send questions from CSV to AI chat bot and save model answers."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--retry-delays",
        type=parse_retry_delays,
        default=DEFAULT_RETRY_DELAYS,
        help="Comma-separated retry delays in seconds. Default: 2,5,10.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows where model_answer is already filled.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    auth_token = os.getenv("AUTH_TOKEN")
    if not auth_token:
        raise RuntimeError(f"AUTH_TOKEN is not set in {args.env_file}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    source_path = output_path if args.skip_existing and output_path.exists() else input_path
    fieldnames, rows = read_csv(source_path, args.delimiter)

    required_columns = {"question", "model_answer"}
    missing_columns = required_columns.difference(fieldnames)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    for column in ["latency_sec", "created_at", "error"]:
        if column not in fieldnames:
            fieldnames.append(column)
            for row in rows:
                row[column] = ""

    indexes = list(range(len(rows)))
    if args.limit is not None:
        indexes = indexes[: args.limit]

    processed = 0
    for position, index in enumerate(indexes, start=1):
        row = rows[index]
        question_id = row.get("id") or str(index + 1)

        if args.skip_existing and is_filled(row.get("model_answer")):
            print(f"[{position}/{len(indexes)}] {question_id}: skipped")
            continue

        question = (row.get("question") or "").strip()
        if not question:
            row["model_answer"] = "failed to answer"
            row["error"] = "Question is empty"
            continue

        print(f"[{position}/{len(indexes)}] {question_id}")
        started_at = time.time()
        row["error"] = ""

        try:
            row["model_answer"] = ask_chat_bot(
                url=args.url,
                auth_token=auth_token,
                question=question,
                timeout=args.timeout,
                retry_delays=args.retry_delays,
            )
        except Exception as exc:
            row["model_answer"] = "failed to answer"
            row["error"] = f"{type(exc).__name__}: {exc}"

        row["latency_sec"] = str(round(time.time() - started_at, 3))
        row["created_at"] = datetime.now().isoformat(timespec="seconds")
        processed += 1

        if args.save_every > 0 and processed % args.save_every == 0:
            save_csv(output_path, fieldnames, rows, args.delimiter)

        if args.delay > 0:
            time.sleep(args.delay)

    save_csv(output_path, fieldnames, rows, args.delimiter)
    print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
