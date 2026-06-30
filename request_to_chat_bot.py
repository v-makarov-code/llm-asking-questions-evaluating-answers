import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_eval.chat_bot_client import (
    DEFAULT_CHATBOT_URL,
    DEFAULT_RETRY_DELAYS,
    ChatBotClient,
    parse_retry_delays,
)
from llm_eval.io import read_dataframe
from llm_eval.pipeline import run_chatbot_pipeline


DEFAULT_INPUT = "RAG_questions_answers.csv"
DEFAULT_OUTPUT = "AI_chat_bot_answers.csv"


def retry_delays_arg(value: str) -> tuple[float, ...]:
    try:
        return parse_retry_delays(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send questions from CSV to AI chat bot and save model answers."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=DEFAULT_CHATBOT_URL)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--retry-delays",
        type=retry_delays_arg,
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
    df = read_dataframe(source_path, args.delimiter)

    client = ChatBotClient(
        url=args.url,
        auth_token=auth_token,
        timeout=args.timeout,
        retry_delays=args.retry_delays,
    )

    run_chatbot_pipeline(
        df=df,
        client=client,
        output_path=output_path,
        delimiter=args.delimiter,
        limit=args.limit,
        save_every=args.save_every,
        delay=args.delay,
        skip_existing=args.skip_existing,
    )
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
