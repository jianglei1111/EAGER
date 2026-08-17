from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from eager.config import canonical_json_bytes
from eager.runtime.llm_client import ChatMessage, OpenAICompatibleClient
from eager.schemas import EvaluationAnnotations


@dataclass(frozen=True, slots=True)
class JudgeEvaluation:
    correct: bool
    llm_calls: int
    input_tokens: int
    output_tokens: int
    call_record: Mapping[str, Any]


class GPTAnswerJudge:
    """Evaluator-only GPTAcc judge compatible with the frozen legacy prompt."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        system_prompt: str,
    ) -> None:
        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("judge system prompt must be non-empty")
        self.client = client
        self.system_prompt = prompt
        self.prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def judge_audited(
        self,
        *,
        prediction: str,
        annotations: EvaluationAnnotations,
    ) -> JudgeEvaluation:
        user_prompt = (
            "Please evaluate if the generated answer is correct by comparing it with the gold answer.\n"
            f"Generated answer: {prediction}\n"
            f"Gold answer: {annotations.answer}\n\n"
            "The generated answer should be considered correct if it:\n"
            "1. Contains the key information from the gold answer\n"
            "2. Is factually accurate and consistent with the gold answer\n"
            "3. Does not contain any contradicting information\n\n"
            "Respond with ONLY 'correct' or 'incorrect'.\nResponse:\n"
        )
        response = self.client.complete(
            (
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ),
            temperature=0.0,
            max_completion_tokens=200,
        )
        normalized = response.content.strip().casefold().rstrip(".")
        if normalized not in {"correct", "incorrect"}:
            raise ValueError("GPT judge response must be exactly 'correct' or 'incorrect'")
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0
        return JudgeEvaluation(
            correct=normalized == "correct",
            llm_calls=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_record={
                "kind": "gpt_accuracy_judge",
                "model": response.model,
                "prompt_sha256": hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "system_prompt_sha256": self.prompt_sha256,
                            "user_prompt": user_prompt,
                        }
                    )
                ).hexdigest(),
                "system_prompt_sha256": self.prompt_sha256,
                "response_sha256": response.response_sha256,
                "request_id": response.request_id,
                "finish_reason": response.finish_reason,
                "attempts": response.attempts,
                "elapsed_seconds": response.elapsed_seconds,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    def judge(
        self,
        *,
        prediction: str,
        annotations: EvaluationAnnotations,
    ) -> bool:
        return self.judge_audited(
            prediction=prediction,
            annotations=annotations,
        ).correct


__all__ = ["GPTAnswerJudge", "JudgeEvaluation"]
