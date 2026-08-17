"""Final-answer contracts shared by EAGER and baselines."""

from eager.answering.base import (
    AnswerGeneration,
    AnswerGenerationError,
    AnswerGenerator,
    AnswerRequest,
    AuditedAnswerGenerator,
    CallableAnswerGenerator,
    StaticAnswerGenerator,
)
from eager.answering.openai import OpenAIAnswerGenerator, parse_answer_text

__all__ = [
    "AnswerGeneration",
    "AnswerGenerationError",
    "AnswerGenerator",
    "AnswerRequest",
    "AuditedAnswerGenerator",
    "CallableAnswerGenerator",
    "StaticAnswerGenerator",
    "OpenAIAnswerGenerator",
    "parse_answer_text",
]
