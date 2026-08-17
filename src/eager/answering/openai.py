from __future__ import annotations

import hashlib
import re

from eager.answering.base import AnswerGeneration, AnswerGenerationError, AnswerRequest
from eager.config import canonical_json_bytes
from eager.runtime.llm_client import ChatMessage, OpenAICompatibleClient


_ANSWER_MARKER = re.compile(r"\bAnswer\s*:\s*", re.IGNORECASE)
_PLACEHOLDER_ANSWER = re.compile(r"^<\s*non-empty\s+answer\s*>$", re.IGNORECASE)


def parse_answer_text(value: str) -> str:
    text = value.strip()
    markers = list(_ANSWER_MARKER.finditer(text))
    answer = text[markers[-1].end() :].strip() if markers else text
    answer = answer.strip().strip('"').strip("'").strip()
    if not answer or _PLACEHOLDER_ANSWER.fullmatch(answer):
        raise ValueError("answer response does not contain a non-empty answer")
    return answer


class OpenAIAnswerGenerator:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        system_prompt: str,
    ) -> None:
        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("answer system prompt must be non-empty")
        self.client = client
        self.system_prompt = prompt
        self.prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        self.max_format_attempts = 2

    def _user_prompt(self, request: AnswerRequest) -> str:
        evidence = "\n".join(
            f"[{index}] {item.title}: {item.text}"
            for index, item in enumerate(request.evidence, start=1)
        )
        step_answers = "\n".join(
            f"{item.step_id}: {item.answer}"
            for item in request.step_answers
        )
        operation = request.final_operation
        operation_text = ""
        if operation is not None:
            operation_text = (
                "\n\nPlanned final operation:\n"
                f"Type: {operation.type}\n"
                f"Inputs: {', '.join(operation.inputs)}\n"
                f"Instruction: {operation.instruction}"
            )
        question = request.question_override or request.payload.question
        return (
            f"Question:\n{question}\n\n"
            f"Resolved intermediate answers:\n{step_answers or '[none]'}\n\n"
            f"Evidence:\n{evidence or '[no retrieved evidence]'}"
            f"{operation_text}"
        )

    def generate_audited(self, request: AnswerRequest) -> AnswerGeneration:
        user_prompt = self._user_prompt(request)
        call_records = []
        total_input_tokens = 0
        total_output_tokens = 0
        last_error: ValueError | None = None
        for format_attempt in range(1, self.max_format_attempts + 1):
            attempt_prompt = user_prompt
            if format_attempt > 1:
                attempt_prompt += (
                    "\n\nYour previous response did not contain a non-empty final answer. "
                    "Return the response again and end with `Answer:` followed by the "
                    "actual supported answer text. Do not write angle brackets or "
                    "the words 'non-empty answer'."
                )
            response = self.client.complete(
                (
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(role="user", content=attempt_prompt),
                )
            )
            prompt_sha256 = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "system_prompt_sha256": self.prompt_sha256,
                        "user_prompt": attempt_prompt,
                    }
                )
            ).hexdigest()
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            call_records.append(
                {
                    "kind": "answer_generation",
                    "model": response.model,
                    "prompt_sha256": prompt_sha256,
                    "system_prompt_sha256": self.prompt_sha256,
                    "response_sha256": response.response_sha256,
                    "request_id": response.request_id,
                    "finish_reason": response.finish_reason,
                    "attempts": response.attempts,
                    "elapsed_seconds": response.elapsed_seconds,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "format_attempt": format_attempt,
                    "format_retry": format_attempt > 1,
                }
            )
            try:
                answer = parse_answer_text(response.content)
            except ValueError as exc:
                last_error = exc
                continue
            return AnswerGeneration(
                answer=answer,
                llm_calls=len(call_records),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                call_record=call_records[0],
                call_records=tuple(call_records),
            )
        raise AnswerGenerationError(
            str(last_error or "answer response does not contain a non-empty answer"),
            llm_calls=len(call_records),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            call_records=call_records,
        )

    def generate(self, request: AnswerRequest) -> str:
        return self.generate_audited(request).answer


__all__ = ["OpenAIAnswerGenerator", "parse_answer_text"]
