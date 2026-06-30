from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of one evaluator for one dataset row.

    The pipeline writes `score` into the column named by `name`. For final-score
    evaluation, `explanation` is also written to `ragas_final_explanation`.
    """

    name: str
    score: float | int | str
    explanation: str = ""
    error: str = ""
