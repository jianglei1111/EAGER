from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import httpx

from eager.config import LLMSettings, ResolvedConfig, canonical_json_bytes


class LLMClientError(RuntimeError):
    code = "llm_error"
    retryable = False

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMConfigurationError(LLMClientError):
    code = "configuration_error"


class LLMAuthenticationError(LLMClientError):
    code = "authentication_error"


class LLMRateLimitError(LLMClientError):
    code = "rate_limit_error"
    retryable = True


class LLMTimeoutError(LLMClientError):
    code = "timeout_error"
    retryable = True


class LLMTransportError(LLMClientError):
    code = "transport_error"
    retryable = True


class LLMServerError(LLMClientError):
    code = "server_error"
    retryable = True


class LLMRequestError(LLMClientError):
    code = "request_error"


class LLMResponseFormatError(LLMClientError):
    code = "response_format_error"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise LLMConfigurationError(f"unsupported chat role: {self.role}")
        if not self.content:
            raise LLMConfigurationError("chat message content cannot be empty")
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str | None
    finish_reason: str | None
    usage: LLMUsage
    request_id: str | None
    response_sha256: str
    attempts: int
    elapsed_seconds: float


def _safe_error_message(payload: Any, fallback: str, api_key: str) -> str:
    message = fallback
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            message = str(error["message"])
        elif isinstance(error, str):
            message = error
    if api_key:
        message = message.replace(api_key, "<redacted>")
    return message[:1000]


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("type") in {"text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    raise LLMResponseFormatError("response message content is not text")


class OpenAICompatibleClient:
    def __init__(
        self,
        settings: LLMSettings,
        *,
        api_key: str,
        http_client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("an API key is required")
        self.settings = settings
        self._api_key = api_key
        self._sleeper = sleeper
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=settings.timeout_seconds)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_url={self.settings.base_url!r}, "
            f"model={self.settings.model!r}, api_key=<redacted>)"
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenAICompatibleClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return f"{self.settings.base_url}/chat/completions"

    def complete(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                normalized_messages.append(message.as_dict())
            elif isinstance(message, Mapping):
                normalized_messages.append(
                    ChatMessage(role=str(message.get("role", "")), content=str(message.get("content", ""))).as_dict()
                )
            else:
                raise LLMConfigurationError("messages must contain ChatMessage or mapping values")
        if not normalized_messages:
            raise LLMConfigurationError("at least one chat message is required")

        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": normalized_messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_completion_tokens": (
                self.settings.max_completion_tokens
                if max_completion_tokens is None
                else max_completion_tokens
            ),
        }
        if response_format is not None:
            body["response_format"] = dict(response_format)
        if extra_body:
            forbidden = {"model", "messages"} & set(extra_body)
            if forbidden:
                raise LLMConfigurationError(f"extra_body cannot replace required fields: {sorted(forbidden)}")
            body.update(extra_body)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": "eager/0.1",
        }
        started = time.perf_counter()
        attempts = 0
        last_error: LLMClientError | None = None
        while attempts <= self.settings.max_retries:
            attempts += 1
            try:
                response = self._client.post(self.endpoint, headers=headers, json=body)
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(str(exc) or "LLM request timed out")
            except httpx.TransportError as exc:
                last_error = LLMTransportError(str(exc) or "LLM transport failed")
            else:
                if 200 <= response.status_code < 300:
                    return self._parse_response(
                        response,
                        attempts=attempts,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                last_error = self._http_error(response)

            if not last_error.retryable or attempts > self.settings.max_retries:
                raise last_error
            delay = self.settings.retry_backoff_seconds * (2 ** (attempts - 1))
            self._sleeper(delay)

        raise last_error or LLMClientError("LLM request failed")

    def _http_error(self, response: httpx.Response) -> LLMClientError:
        try:
            payload: Any = response.json()
        except (ValueError, json.JSONDecodeError):
            payload = None
        message = _safe_error_message(payload, response.reason_phrase or "LLM request failed", self._api_key)
        status = response.status_code
        if status in {401, 403}:
            return LLMAuthenticationError(message, status_code=status)
        if status == 429:
            return LLMRateLimitError(message, status_code=status)
        if status in {408, 504}:
            return LLMTimeoutError(message, status_code=status)
        if status >= 500:
            return LLMServerError(message, status_code=status)
        return LLMRequestError(message, status_code=status)

    def _parse_response(
        self,
        response: httpx.Response,
        *,
        attempts: int,
        elapsed_seconds: float,
    ) -> LLMResponse:
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise LLMResponseFormatError("LLM response is not valid JSON") from exc
        try:
            choice = payload["choices"][0]
            content = _content_text(choice["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseFormatError("LLM response has no chat completion choice") from exc
        usage_payload = payload.get("usage") if isinstance(payload, Mapping) else None
        usage_payload = usage_payload if isinstance(usage_payload, Mapping) else {}
        raw_bytes = response.content or canonical_json_bytes(payload)
        return LLMResponse(
            content=content,
            model=str(payload.get("model")) if payload.get("model") is not None else None,
            finish_reason=str(choice.get("finish_reason")) if choice.get("finish_reason") is not None else None,
            usage=LLMUsage(
                prompt_tokens=_optional_int(usage_payload.get("prompt_tokens")),
                completion_tokens=_optional_int(usage_payload.get("completion_tokens")),
                total_tokens=_optional_int(usage_payload.get("total_tokens")),
            ),
            request_id=response.headers.get("x-request-id") or payload.get("id"),
            response_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            attempts=attempts,
            elapsed_seconds=round(elapsed_seconds, 6),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise LLMResponseFormatError(f"usage token value is not an integer: {value!r}") from exc


def client_from_resolved_config(
    resolved: ResolvedConfig,
    *,
    environment: Mapping[str, str] | None = None,
    http_client: httpx.Client | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> OpenAICompatibleClient:
    values = environment if environment is not None else os.environ
    api_key = resolved.secret("llm.api_key") or values.get(resolved.config.llm.api_key_env, "")
    if not api_key:
        raise LLMConfigurationError(
            f"missing API key; set {resolved.config.llm.api_key_env} or provide it in the ignored local override"
        )
    return OpenAICompatibleClient(
        resolved.config.llm,
        api_key=api_key,
        http_client=http_client,
        sleeper=sleeper,
    )


def messages_from_pairs(values: Iterable[tuple[str, str]]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in values]
