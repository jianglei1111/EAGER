from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import httpx
import pytest

from eager.config import LLMSettings
from eager.runtime.llm_client import (
    ChatMessage,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMResponseFormatError,
    OpenAICompatibleClient,
)


class MockHandler(BaseHTTPRequestHandler):
    server: "MockServer"

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(content_length))
        self.server.requests.append(
            {
                "path": self.path,
                "body": body,
                "authorization": self.headers.get("Authorization"),
            }
        )
        status, payload, headers = self.server.responses.pop(0)
        encoded = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class MockServer(ThreadingHTTPServer):
    responses: list[tuple[int, Any, dict[str, str]]]
    requests: list[dict[str, Any]]


@contextmanager
def mock_server(responses: list[tuple[int, Any, dict[str, str]]]) -> Iterator[MockServer]:
    server = MockServer(("127.0.0.1", 0), MockHandler)
    server.responses = list(responses)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def completion_payload(content: str = "Paris") -> dict[str, Any]:
    return {
        "id": "request-1",
        "model": "mock-model-2026-07-14",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def settings(base_url: str, *, retries: int = 2) -> LLMSettings:
    return LLMSettings(
        base_url=base_url,
        model="mock-model",
        timeout_seconds=2,
        max_retries=retries,
        retry_backoff_seconds=0,
    )


def test_real_http_contract_parses_usage_and_redacts_repr() -> None:
    fake_key = "super-" + "private-key"
    with mock_server([(200, completion_payload(), {"x-request-id": "header-request"})]) as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        with httpx.Client(timeout=2, trust_env=False) as transport:
            client = OpenAICompatibleClient(settings(base_url), api_key=fake_key, http_client=transport)
            response = client.complete([ChatMessage("user", "What is the capital of France?")])

        assert response.content == "Paris"
        assert response.usage.total_tokens == 4
        assert response.request_id == "header-request"
        assert response.attempts == 1
        assert len(response.response_sha256) == 64
        assert server.requests[0]["path"] == "/v1/chat/completions"
        assert server.requests[0]["authorization"] == "Bearer " + fake_key
        assert fake_key not in repr(client)


def test_rate_limit_retries_then_succeeds() -> None:
    responses = [
        (429, {"error": {"message": "slow down"}}, {}),
        (200, completion_payload("ok"), {}),
    ]
    with mock_server(responses) as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        delays: list[float] = []
        with httpx.Client(timeout=2, trust_env=False) as transport:
            client = OpenAICompatibleClient(
                settings(base_url),
                api_key="key",
                http_client=transport,
                sleeper=delays.append,
            )
            response = client.complete([ChatMessage("user", "hello")])
        assert response.attempts == 2
        assert delays == [0.0]
        assert len(server.requests) == 2


def test_authentication_failure_is_not_retried() -> None:
    with mock_server([(401, {"error": {"message": "bad key"}}, {})]) as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        with httpx.Client(timeout=2, trust_env=False) as transport:
            client = OpenAICompatibleClient(settings(base_url), api_key="bad", http_client=transport)
            with pytest.raises(LLMAuthenticationError):
                client.complete([ChatMessage("user", "hello")])
        assert len(server.requests) == 1


def test_exhausted_rate_limit_raises_typed_error() -> None:
    responses = [
        (429, {"error": {"message": "busy"}}, {}),
        (429, {"error": {"message": "still busy"}}, {}),
    ]
    with mock_server(responses) as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        with httpx.Client(timeout=2, trust_env=False) as transport:
            client = OpenAICompatibleClient(
                settings(base_url, retries=1),
                api_key="key",
                http_client=transport,
                sleeper=lambda _delay: None,
            )
            with pytest.raises(LLMRateLimitError) as exc_info:
                client.complete([ChatMessage("user", "hello")])
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 429


def test_malformed_success_response_fails_closed() -> None:
    with mock_server([(200, {"choices": []}, {})]) as server:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        with httpx.Client(timeout=2, trust_env=False) as transport:
            client = OpenAICompatibleClient(settings(base_url), api_key="key", http_client=transport)
            with pytest.raises(LLMResponseFormatError):
                client.complete([ChatMessage("user", "hello")])
