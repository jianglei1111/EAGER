from __future__ import annotations

import pytest

from eager.answering import (
    AnswerGenerationError,
    AnswerRequest,
    OpenAIAnswerGenerator,
    StaticAnswerGenerator,
    parse_answer_text,
)
from eager.evaluation import (
    GPTAnswerJudge,
    containment_accuracy,
    evaluate_answer,
    exact_match,
    normalize_answer,
    token_f1,
)
from eager.schemas import (
    EvaluationAnnotations,
    EvidenceUnit,
    FinalOperation,
    InferencePayload,
    QuestionHandle,
    StepAnswer,
)
from eager.runtime.llm_client import LLMResponse, LLMUsage


def test_official_style_normalization_and_overlap_metrics() -> None:
    assert normalize_answer("The, CAT!") == "cat"
    assert exact_match("The Harbor City", "Harbor City") == 1.0
    assert token_f1("Harbor City museum", "Harbor City") == pytest.approx(0.8)
    assert containment_accuracy("It is Harbor City.", "Harbor City") == 1.0
    assert containment_accuracy("Harbor", "Harbor City") == 0.0


def test_answer_aliases_choose_one_reference_consistently() -> None:
    metrics = evaluate_answer(
        "Bellhaven City",
        "Bellhaven",
        aliases=("Bellhaven City",),
    )
    assert metrics.exact_match == 1.0
    assert metrics.f1 == 1.0
    assert metrics.matched_reference == "Bellhaven City"


def test_answer_generator_receives_only_question_and_retrieved_evidence() -> None:
    evidence = EvidenceUnit(
        evidence_id="p1_s0",
        paragraph_id="p1",
        sentence_id="p1_s0",
        title="Northstar Labs",
        text="The company is headquartered in Harbor City.",
    )
    request = AnswerRequest(
        payload=InferencePayload(
            dataset="hotpotqa",
            question="Where is Northstar Labs headquartered?",
        ),
        evidence=[evidence],
    )
    assert StaticAnswerGenerator("Harbor City").generate(request) == "Harbor City"
    assert request.payload.model_dump(mode="json") == {
        "dataset": "hotpotqa",
        "question": "Where is Northstar Labs headquartered?",
    }


def test_openai_answer_generator_parses_and_audits_without_raw_prompts() -> None:
    class FakeClient:
        def complete(self, messages):
            assert len(messages) == 2
            return LLMResponse(
                content="Thought: The evidence names the city.\nAnswer: Harbor City",
                model="fixture-model",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=20, completion_tokens=7, total_tokens=27),
                request_id="request-1",
                response_sha256="a" * 64,
                attempts=1,
                elapsed_seconds=0.25,
            )

    generator = OpenAIAnswerGenerator(client=FakeClient(), system_prompt="Answer from evidence.")
    generation = generator.generate_audited(
        AnswerRequest(
            payload=InferencePayload(dataset="2wiki", question="Where is it?"),
            evidence=(
                EvidenceUnit(
                    evidence_id="e1",
                    paragraph_id="p1",
                    sentence_id="s1",
                    title="Northstar Labs",
                    text="It is headquartered in Harbor City.",
                ),
            ),
        )
    )

    assert generation.answer == "Harbor City"
    assert generation.llm_calls == 1
    assert generation.input_tokens == 20
    assert generation.output_tokens == 7
    assert "Where is it?" not in str(generation.call_record)
    assert "Harbor City" not in str(generation.call_record)
    assert parse_answer_text("plain answer") == "plain answer"
    assert parse_answer_text(
        "Thought: The evidence resolves it. Answer: Harbor City"
    ) == "Harbor City"


def test_openai_answer_generator_receives_planned_final_operation() -> None:
    class FakeClient:
        def complete(self, messages):
            user_prompt = messages[1].content
            assert "Planned final operation:" in user_prompt
            assert "Type: compare" in user_prompt
            assert "Inputs: step_2, step_4" in user_prompt
            assert "Return the film whose director was born earlier." in user_prompt
            assert "Resolved intermediate answers:" in user_prompt
            assert "step_2: 1990" in user_prompt
            assert "step_4: 1980" in user_prompt
            return LLMResponse(
                content="Answer: Film A",
                model="fixture-model",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
                request_id="request-operation",
                response_sha256="e" * 64,
                attempts=1,
                elapsed_seconds=0.1,
            )

    request = AnswerRequest(
        payload=InferencePayload(
            dataset="2wiki",
            question="Which film has the older director?",
        ),
        evidence=(),
        final_operation=FinalOperation(
            type="compare",
            inputs=("step_2", "step_4"),
            instruction="Return the film whose director was born earlier.",
        ),
        step_answers=(
            StepAnswer(step_id="step_2", answer="1990"),
            StepAnswer(step_id="step_4", answer="1980"),
        ),
    )

    assert OpenAIAnswerGenerator(
        client=FakeClient(),
        system_prompt="Answer from evidence.",
    ).generate(request) == "Film A"


def test_openai_answer_generator_retries_empty_final_answer_and_audits_both_calls() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            return LLMResponse(
                content=(
                    "Thought: Evidence is inconclusive. Answer:"
                    if self.calls == 1
                    else "Thought: The evidence names it. Answer: Harbor City"
                ),
                model="fixture-model",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                request_id=f"request-{self.calls}",
                response_sha256=str(self.calls) * 64,
                attempts=1,
                elapsed_seconds=0.1,
            )

    generator = OpenAIAnswerGenerator(client=FakeClient(), system_prompt="Answer from evidence.")
    generation = generator.generate_audited(
        AnswerRequest(
            payload=InferencePayload(dataset="2wiki", question="Where is it?"),
            evidence=(),
        )
    )

    assert generation.answer == "Harbor City"
    assert generation.llm_calls == 2
    assert generation.input_tokens == 20
    assert generation.output_tokens == 4
    assert len(generation.call_records) == 2
    assert [record["format_retry"] for record in generation.call_records] == [False, True]


def test_answer_parser_rejects_retry_placeholder() -> None:
    with pytest.raises(ValueError, match="non-empty answer"):
        parse_answer_text("Thought: I cannot resolve it. Answer: <non-empty answer>")


def test_openai_answer_generator_preserves_audit_when_format_retries_fail() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            return LLMResponse(
                content="Thought: Evidence is inconclusive. Answer:",
                model="fixture-model",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                request_id=f"request-{self.calls}",
                response_sha256=str(self.calls) * 64,
                attempts=1,
                elapsed_seconds=0.1,
            )

    generator = OpenAIAnswerGenerator(client=FakeClient(), system_prompt="Answer from evidence.")
    request = AnswerRequest(
        payload=InferencePayload(dataset="2wiki", question="Where is it?"),
        evidence=(),
    )

    with pytest.raises(AnswerGenerationError) as captured:
        generator.generate_audited(request)

    error = captured.value
    assert error.llm_calls == 2
    assert error.input_tokens == 20
    assert error.output_tokens == 4
    assert len(error.call_records) == 2


def test_gpt_answer_judge_is_audited_and_evaluator_only() -> None:
    class FakeClient:
        def complete(self, messages, **kwargs):
            assert kwargs == {"temperature": 0.0, "max_completion_tokens": 200}
            assert "Gold answer: Harbor City" in messages[1].content
            return LLMResponse(
                content="correct",
                model="fixture-judge",
                finish_reason="stop",
                usage=LLMUsage(prompt_tokens=30, completion_tokens=1, total_tokens=31),
                request_id="judge-1",
                response_sha256="b" * 64,
                attempts=1,
                elapsed_seconds=0.1,
            )

    handle = QuestionHandle.create(dataset="2wiki", split="fixture", raw_dataset_id="q1")
    evaluation = GPTAnswerJudge(client=FakeClient(), system_prompt="You are an evaluator.").judge_audited(
        prediction="Harbor City",
        annotations=EvaluationAnnotations(
            handle_id=handle.handle_id,
            answer="Harbor City",
        ),
    )

    assert evaluation.correct is True
    assert evaluation.llm_calls == 1
    assert evaluation.input_tokens == 30
    assert "Harbor City" not in str(evaluation.call_record)
