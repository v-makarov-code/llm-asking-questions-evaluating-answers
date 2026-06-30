import json
import time

import requests


DEFAULT_CHATBOT_URL = "https://ai.sapiens.solutions/api/v1/conversations/ask/stream"
DEFAULT_RETRY_DELAYS = (2.0, 5.0, 10.0)


class ChatBotClient:
    """Client for the AI chat-bot streaming endpoint.

    The endpoint returns server-sent-event-like `data: ...` lines. This client
    collects text deltas into one final answer and retries transient failures.
    """

    def __init__(
        self,
        url: str,
        auth_token: str,
        timeout: float = 300.0,
        retry_delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS,
    ) -> None:
        self.url = url
        self.auth_token = auth_token
        self.timeout = timeout
        self.retry_delays = retry_delays

    def ask(self, question: str) -> str:
        """Ask one question and return the complete text answer.

        Retries are applied only for transient network/server errors. Empty
        answers are treated as failures so the caller can mark the row as
        `failed to answer`.
        """

        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            try:
                return self._ask_once(question)
            except Exception as exc:
                is_last_attempt = attempt == attempts - 1
                if is_last_attempt or not should_retry(exc):
                    raise

                delay = self.retry_delays[attempt]
                print(
                    f"  attempt {attempt + 1}/{attempts} failed: "
                    f"{type(exc).__name__}; retry in {delay:g}s"
                )
                time.sleep(delay)

        raise RuntimeError("Retry loop exited unexpectedly")

    def _ask_once(self, question: str) -> str:
        """Perform one HTTP streaming request without retry handling."""

        with requests.Session() as session:
            with session.post(
                self.url,
                headers={
                    "Authorization": f"Token {self.auth_token}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
                json={"query": question},
                stream=True,
                timeout=self.timeout,
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
    """Return True for errors that are worth retrying."""

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


def is_answer_filled(value: str | None) -> bool:
    """Check whether `model_answer` should be treated as already completed."""

    if not value:
        return False
    normalized = value.strip().casefold()
    return bool(normalized and normalized != "failed to answer")


def parse_retry_delays(value: str) -> tuple[float, ...]:
    """Parse CLI retry delays like `2,5,10` into a tuple of seconds."""

    delays = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if any(delay < 0 for delay in delays):
        raise ValueError("Retry delays must be non-negative")
    return delays
