import argparse
from pathlib import Path

from llm_eval.evaluators import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_REPO,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_METRICS,
    DEFAULT_ONNX_FILE,
    build_evaluators,
    parse_metrics,
)
from llm_eval.io import read_dataframe
from llm_eval.pipeline import run_scoring_pipeline


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

    metrics = parse_metrics(args.metrics)
    evaluators = build_evaluators(metrics, args)
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = read_dataframe(input_path, args.delimiter)

    run_scoring_pipeline(
        df=df,
        evaluators=evaluators,
        metrics=metrics,
        judge_model=args.judge_model,
        output_path=output_path,
        delimiter=args.delimiter,
        limit=args.limit,
        max_new=args.max_new,
        save_every=args.save_every,
        skip_existing=args.skip_existing,
    )
    print(f"Saved scored CSV to {output_path}")


if __name__ == "__main__":
    main()
