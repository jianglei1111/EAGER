from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eager.content_identity import make_paragraph_id
from eager.indexing import CorpusRecord, CorpusSnapshot, build_tiny_index
from eager.pipeline import OnlineAutoRunner, RetrievalEngine
from eager.retrieval import RetrievalKernel, RetrievalKernelConfig, TinySemanticIndex
from eager.runtime.llm_client import LLMResponse, LLMUsage
from eager.schemas import (
    BudgetContract,
    EvidenceCandidate,
    EvidenceUnit,
    FinalOperation,
    GeneratedSignalStep,
    InferencePayload,
    RetrievalState,
)
from eager.signals import (
    ChainGeneration,
    HybridRouter,
    LLMRouter,
    OpenAIChainGenerator,
    OpenAIIntermediateAnswerer,
    OpenAIRetrievalQuestionGenerator,
    RetrievalQuestionController,
    RuleRouter,
    StaticChainGenerator,
    StaticIntermediateAnswerer,
    SubquestionChainController,
    TraceReplayChainGenerator,
    materialize_step,
    parse_router_output,
)


QUESTION = "Who is the mother of the director of The Silent Map?"


def test_trace_replay_chain_uses_only_question_and_generated_plan() -> None:
    generator = TraceReplayChainGenerator(
        [
            {
                "handle_id": "opaque",
                "gold_answer": "must be ignored",
                "run": {
                    "payload": {"dataset": "2wiki", "question": QUESTION},
                    "controller_state": {
                        "generated_chain": [
                            {
                                "step_id": "step_1",
                                "text": "Who directed The Silent Map?",
                                "depends_on": [],
                            },
                            {
                                "step_id": "step_2",
                                "text": "Who is the mother of #1?",
                                "depends_on": ["step_1"],
                            },
                        ],
                        "step_answers": [
                            {"step_id": "step_1", "answer": "must be ignored"}
                        ],
                    },
                },
            }
        ]
    )

    steps = generator.generate(
        InferencePayload(dataset="2wiki", question=f"  {QUESTION}  "),
        max_steps=2,
    )

    assert steps == _chain()


class FakeLLMClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.messages: list[tuple[Any, ...]] = []

    def complete(self, messages, **_kwargs) -> LLMResponse:
        self.messages.append(tuple(messages))
        content = self.outputs.pop(0)
        return LLMResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
            request_id=f"request-{len(self.messages)}",
            response_sha256=str(len(self.messages)) * 64,
            attempts=1,
            elapsed_seconds=0.01,
        )


class StaticDAGGenerator:
    def __init__(
        self,
        steps: tuple[GeneratedSignalStep, ...],
        final_operation: FinalOperation,
    ) -> None:
        self.generation = ChainGeneration(
            steps=steps,
            final_operation=final_operation,
        )

    def generate(self, _payload, max_steps: int) -> ChainGeneration:
        assert len(self.generation.steps) <= max_steps
        return self.generation


def _corpus() -> CorpusSnapshot:
    records = []
    for title, sentences in (
        ("The Silent Map", ("The Silent Map was directed by Elian Voss.",)),
        ("Elian Voss", ("Elian Voss is the son of Rhea Voss.",)),
    ):
        records.append(
            CorpusRecord(
                paragraph_id=make_paragraph_id(
                    dataset="2wiki",
                    title=title,
                    content=sentences,
                ),
                title=title,
                text=" ".join(sentences),
                sentences=sentences,
                source_dataset="2wiki",
                source_split="synthetic_chain",
            )
        )
    return CorpusSnapshot.from_records(records)


def _engine(tmp_path: Path) -> RetrievalEngine:
    corpus = _corpus()
    index_dir = tmp_path / "index"
    build_tiny_index(corpus, index_dir, dimensions=32)
    return RetrievalEngine(
        RetrievalKernel(
            corpus=corpus,
            semantic_index=TinySemanticIndex.load(index_dir, corpus=corpus),
            config=RetrievalKernelConfig(
                candidate_limit=5,
                paragraph_limit=2,
                rerank_limit=5,
                evidence_limit=4,
            ),
        )
    )


def _budget() -> BudgetContract:
    return BudgetContract(
        max_iterations=2,
        max_candidates_per_step=5,
        max_candidates_per_question=10,
        max_parent_units_per_step=2,
        max_parent_units_per_question=4,
        max_rerank_pairs_per_step=5,
        max_rerank_pairs_per_question=10,
        max_retained_evidence=4,
        max_llm_calls_per_step=0,
        max_llm_calls_per_question=0,
        max_input_tokens_per_question=0,
        max_output_tokens_per_question=0,
    )


def _llm_budget() -> BudgetContract:
    return _budget().model_copy(
        update={
            "max_llm_calls_per_step": 2,
            "max_llm_calls_per_question": 4,
            "max_input_tokens_per_question": 100,
            "max_output_tokens_per_question": 100,
        }
    )


def _chain() -> tuple[GeneratedSignalStep, ...]:
    return (
        GeneratedSignalStep(
            step_id="step_1",
            text="Who directed The Silent Map?",
        ),
        GeneratedSignalStep(
            step_id="step_2",
            text="Who is the mother of #1?",
            depends_on=("step_1",),
        ),
    )


def test_chain_materializes_dependency_and_injects_intermediate_answer(
    tmp_path: Path,
) -> None:
    controller = SubquestionChainController(
        generator=StaticChainGenerator(_chain()),
        answerer=StaticIntermediateAnswerer(
            {"step_1": "Elian Voss", "step_2": "Rhea Voss"}
        ),
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=controller,
        budget=_budget(),
    )

    assert [step.decision.signal_text for step in result.steps] == [
        "Who directed The Silent Map?",
        "Who is the mother of Elian Voss?",
    ]
    assert [answer.answer for answer in result.controller_state.step_answers] == [
        "Elian Voss",
    ]
    assert [step.outcome.used_fallback for step in result.steps] == [False, False]
    assert {item.title for item in result.retrieval_state.retained_evidence} == {
        "The Silent Map",
        "Elian Voss",
    }


def test_chain_falls_back_when_dependency_answer_is_missing(tmp_path: Path) -> None:
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=SubquestionChainController(
            generator=StaticChainGenerator(_chain()),
            answerer=StaticIntermediateAnswerer({}),
        ),
        budget=_budget(),
    )

    assert len(result.steps) == 1
    assert result.steps[0].outcome.step_status == "failed"
    assert result.steps[0].outcome.used_fallback is False
    assert result.controller_state.failed_step_ids == ("step_1",)
    assert result.controller_state.blocked_step_ids == ("step_2",)
    materialized, answers, unresolved = materialize_step(
        _chain()[1],
        chain=_chain(),
        answers={},
    )
    assert materialized == "Who is the mother of #1?"
    assert answers == ()
    assert unresolved is True


def test_openai_chain_records_decomposition_and_intermediate_answer_usage(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient(
        [
            '{"target_slot":"mother","subquestions":['
            '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]},'
            '{"id":2,"question":"Who is the mother of #1?","depends_on":[1]}]}',
            '{"answer":"Elian Voss"}',
        ]
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=SubquestionChainController(
            generator=OpenAIChainGenerator(client=client, system_prompt="decompose"),
            answerer=OpenAIIntermediateAnswerer(client=client, system_prompt="answer"),
        ),
        budget=_llm_budget(),
    )

    assert [step.decision.signal_text for step in result.steps] == [
        "Who directed The Silent Map?",
        "Who is the mother of Elian Voss?",
    ]
    assert result.retrieval_state.total_usage.llm_calls == 2
    assert result.retrieval_state.total_usage.input_tokens == 22
    assert result.retrieval_state.total_usage.output_tokens == 6
    assert [record["kind"] for record in result.steps[0].decision.call_records] == [
        "subquestion_decomposition"
    ]
    assert [record["kind"] for record in result.steps[0].outcome.call_records] == [
        "intermediate_answer_generation"
    ]
    assert result.steps[1].outcome.call_records == ()
    decomposition_input = client.messages[0][1].content
    assert set(json.loads(decomposition_input)) == {"original_question"}


def test_intermediate_answer_retries_same_evidence_and_recovers_plain_text() -> None:
    state = RetrievalState.initial(_llm_budget())
    client = FakeLLMClient([
        "The answer is Elian Voss.",
        '{"answer":"Elian Voss"}',
    ])
    answerer = OpenAIIntermediateAnswerer(
        client=client,
        system_prompt="answer",
        same_evidence_retry_limit=1,
    )

    recovered = answerer.answer(
        InferencePayload(dataset="2wiki", question=QUESTION),
        GeneratedSignalStep(step_id="step_1", text="Who directed The Silent Map?"),
        state,
    )

    assert recovered.answer == "The answer is Elian Voss."
    assert recovered.llm_calls == 1
    assert recovered.call_records[0]["format_attempt"] == 1

    client = FakeLLMClient(["", '{"answer":"Elian Voss"}'])
    answerer = OpenAIIntermediateAnswerer(
        client=client,
        system_prompt="answer",
        same_evidence_retry_limit=1,
    )
    retried = answerer.answer(
        InferencePayload(dataset="2wiki", question=QUESTION),
        GeneratedSignalStep(step_id="step_1", text="Who directed The Silent Map?"),
        state,
    )

    assert retried.answer == "Elian Voss"
    assert retried.llm_calls == 2
    assert retried.call_records[1]["same_evidence_retry"] is True


def test_openai_chain_retries_structurally_invalid_plan_without_metadata(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient(
        [
            '{"target_slot":"mother","subquestions":['
            '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]},'
            '{"id":2,"question":"Who is the mother of #2?","depends_on":[1]}]}',
            '{"target_slot":"mother","subquestions":['
            '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]},'
            '{"id":2,"question":"Who is the mother of #1?","depends_on":[1]}]}',
            '{"answer":"Elian Voss"}',
        ]
    )
    retry_budget = _llm_budget().model_copy(
        update={
            "max_llm_calls_per_step": 3,
            "max_llm_calls_per_question": 5,
        }
    )

    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=SubquestionChainController(
            generator=OpenAIChainGenerator(client=client, system_prompt="decompose"),
            answerer=OpenAIIntermediateAnswerer(client=client, system_prompt="answer"),
        ),
        budget=retry_budget,
    )

    assert result.steps[1].decision.signal_text == "Who is the mother of Elian Voss?"
    assert result.retrieval_state.total_usage.llm_calls == 3
    assert len(result.steps[0].decision.call_records) == 2
    assert result.steps[0].decision.call_records[0]["validation_error"]
    assert result.steps[0].decision.call_records[1]["validation_error"] is None
    first_request = json.loads(client.messages[0][1].content)
    retry_request = json.loads(client.messages[1][1].content)
    assert set(first_request) == {"original_question"}
    assert set(retry_request) == {
        "original_question",
        "previous_invalid_plan",
        "retry_instruction",
        "validation_error",
    }
    assert "MINIMAL EDIT task" in retry_request["retry_instruction"]
    assert "Do not regenerate" in retry_request["retry_instruction"]
    assert not ({"question_id", "hop", "hop_count", "type"} & set(retry_request))


def test_dag_parser_rejects_declared_dependency_without_placeholder() -> None:
    raw = (
        '{"steps":['
        '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]},'
        '{"id":2,"question":"When was the director born?","depends_on":[1]}],'
        '"final_operation":{"type":"return","inputs":[2],'
        '"instruction":"Return the birth date."}}'
    )

    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(raw, 2)

    assert steps == ()
    assert operation is None
    assert "placeholders do not match" in (error or "")


def test_simple_steps_parser_derives_dependencies_from_placeholders() -> None:
    raw = (
        '{"steps":['
        '{"id":1,"question":"Who directed The Silent Map?"},'
        '{"id":2,"question":"Who is the mother of #1?"}]}'
    )

    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(raw, 2)

    assert error is None
    assert [step.depends_on for step in steps] == [(), ("step_1",)]
    assert operation is None


def test_simple_steps_parser_accepts_independent_terminal_branches() -> None:
    raw = (
        '{"steps":['
        '{"id":1,"question":"What is the population of City A?"},'
        '{"id":2,"question":"What is the population of City B?"}]}'
    )

    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(raw, 2)

    assert error is None
    assert [step.depends_on for step in steps] == [(), ()]
    assert operation is None


def test_simple_steps_parser_rejects_textual_step_references() -> None:
    raw = (
        '{"steps":['
        '{"id":1,"question":"Who directed Film A?"},'
        '{"id":2,"question":"When was step 1 born?"}]}'
    )

    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(raw, 2)

    assert steps == ()
    assert operation is None
    assert "#n placeholders" in (error or "")


def test_dag_parser_rejects_return_operation_with_multiple_inputs() -> None:
    raw = (
        '{"steps":['
        '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]},'
        '{"id":2,"question":"Who wrote The Silent Map?","depends_on":[]}],'
        '"final_operation":{"type":"return","inputs":[1,2],'
        '"instruction":"Return the answer."}}'
    )

    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(raw, 2)

    assert steps == ()
    assert operation is None
    assert "exactly one input" in (error or "")


def test_dag_parser_rejects_compare_operation_with_one_input() -> None:
    raw = (
        '{"steps":['
        '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]}],'
        '"final_operation":{"type":"compare","inputs":[1],'
        '"instruction":"Compare the result."}}'
    )

    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(raw, 1)

    assert steps == ()
    assert operation is None
    assert "at least two inputs" in (error or "")


def test_failed_branch_blocks_dependents_but_keeps_independent_branch_running(
    tmp_path: Path,
) -> None:
    chain = (
        GeneratedSignalStep(step_id="step_1", text="Who directed The Silent Map?"),
        GeneratedSignalStep(
            step_id="step_2",
            text="Who is the mother of #1?",
            depends_on=("step_1",),
        ),
        GeneratedSignalStep(step_id="step_3", text="What genre is The Silent Map?"),
    )
    budget = _budget().model_copy(
        update={
            "max_iterations": 4,
            "max_candidates_per_question": 15,
            "max_parent_units_per_question": 6,
            "max_rerank_pairs_per_question": 15,
        }
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=SubquestionChainController(
            generator=StaticChainGenerator(chain),
            answerer=StaticIntermediateAnswerer({}),
            max_step_retries=1,
        ),
        budget=budget,
    )

    assert result.steps[0].decision.signal_text == "Who directed The Silent Map?"
    assert "Current relation: Who directed The Silent Map?" in result.steps[1].decision.signal_text
    assert "Original question:" in result.steps[1].decision.signal_text
    assert "#" not in result.steps[1].decision.signal_text
    assert result.steps[2].decision.signal_text == "What genre is The Silent Map?"
    assert [step.outcome.step_status for step in result.steps] == [
        "retrying",
        "failed",
        "succeeded",
    ]
    assert result.controller_state.failed_step_ids == ("step_1",)
    assert result.controller_state.blocked_step_ids == ("step_2",)
    assert result.controller_state.completed_steps == 3


def test_openai_chain_reviews_valid_plan_and_keeps_dependency_complete_revision() -> None:
    client = FakeLLMClient(
        [
            '{"target_slot":"mother","subquestions":['
            '{"id":1,"question":"Who directed The Silent Map?","depends_on":[]},'
            '{"id":2,"question":"Who is the mother of #1?","depends_on":[1]}]}',
            '{"target_slot":"mother","subquestions":['
            '{"id":1,"question":"Who wrote The Silent Map?","depends_on":[]},'
            '{"id":2,"question":"Who directed the adaptation by #1?","depends_on":[1]},'
            '{"id":3,"question":"Who is the mother of #2?","depends_on":[2]}]}',
        ]
    )
    generation = OpenAIChainGenerator(
        client=client,
        system_prompt="decompose",
        review_valid_plan=True,
    ).generate(
        InferencePayload(dataset="2wiki", question=QUESTION),
        max_steps=4,
    )

    assert len(generation.steps) == 3
    assert generation.llm_calls == 2
    review_request = json.loads(client.messages[1][1].content)
    assert set(review_request) == {
        "original_question",
        "draft_plan",
        "review_instruction",
    }
    assert generation.call_records[1]["plan_review"] is True
    assert generation.call_records[1]["plan_accepted"] is True


def test_dag_parser_accepts_independent_roots_and_rejects_orphans() -> None:
    valid = (
        '{"steps":['
        '{"id":1,"question":"When was Film A released?","depends_on":[]},'
        '{"id":2,"question":"When was Film B released?","depends_on":[]}],'
        '"final_operation":{"type":"compare","inputs":[1,2],'
        '"instruction":"Return the film released earlier."}}'
    )
    steps, operation, error = OpenAIChainGenerator._parse_plan_with_error(valid, 2)

    assert error is None
    assert [step.depends_on for step in steps] == [(), ()]
    assert operation == FinalOperation(
        type="compare",
        inputs=("step_1", "step_2"),
        instruction="Return the film released earlier.",
    )

    orphan = valid.replace(
        '],"final_operation"',
        ',{"id":3,"question":"Who directed Film C?","depends_on":[]}],'
        '"final_operation"',
    )
    orphan_steps, orphan_operation, orphan_error = (
        OpenAIChainGenerator._parse_plan_with_error(orphan, 3)
    )
    assert orphan_steps == ()
    assert orphan_operation is None
    assert "orphan" in (orphan_error or "")


def test_dag_review_keeps_draft_operation_when_revision_drops_a_branch() -> None:
    client = FakeLLMClient(
        [
            '{"steps":['
            '{"id":1,"question":"When was Film A released?","depends_on":[]},'
            '{"id":2,"question":"When was Film B released?","depends_on":[]}],'
            '"final_operation":{"type":"compare","inputs":[1,2],'
            '"instruction":"Return the film released earlier."}}',
            '{"steps":['
            '{"id":1,"question":"When was Film A released?","depends_on":[]}],'
            '"final_operation":{"type":"return","inputs":[1],'
            '"instruction":"Return step 1."}}',
        ]
    )

    generation = OpenAIChainGenerator(
        client=client,
        system_prompt="decompose",
        review_valid_plan=True,
    ).generate(
        InferencePayload(dataset="2wiki", question="Which film was released earlier?"),
        max_steps=2,
    )

    assert len(generation.steps) == 2
    assert generation.final_operation == FinalOperation(
        type="compare",
        inputs=("step_1", "step_2"),
        instruction="Return the film released earlier.",
    )
    assert generation.call_records[1]["plan_accepted"] is False


def test_dag_terminal_branches_do_not_call_intermediate_answerer(
    tmp_path: Path,
) -> None:
    class RejectingAnswerer:
        def answer(self, *_args) -> str:
            raise AssertionError("terminal DAG branches must not be answered early")

    operation = FinalOperation(
        type="compare",
        inputs=("step_1", "step_2"),
        instruction="Return the film released earlier.",
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(
            dataset="2wiki",
            question="Which was released earlier, The Silent Map or Elian Voss?",
        ),
        controller=SubquestionChainController(
            generator=StaticDAGGenerator(
                (
                    GeneratedSignalStep(
                        step_id="step_1",
                        text="When was The Silent Map released?",
                    ),
                    GeneratedSignalStep(
                        step_id="step_2",
                        text="When was Elian Voss released?",
                    ),
                ),
                operation,
            ),
            answerer=RejectingAnswerer(),
        ),
        budget=_budget(),
    )

    assert len(result.steps) == result.retrieval_state.iteration == 2
    assert result.controller_state.final_operation == operation
    assert result.controller_state.step_answers == ()


def test_dag_answers_only_nodes_consumed_by_later_retrievals(tmp_path: Path) -> None:
    class RecordingAnswerer:
        def __init__(self) -> None:
            self.step_ids: list[str] = []

        def answer(self, _payload, step, _retrieval_state) -> str:
            self.step_ids.append(step.step_id)
            return {"step_1": "Elian Voss", "step_3": "Rhea Voss"}[step.step_id]

    steps = (
        GeneratedSignalStep(step_id="step_1", text="Who directed The Silent Map?"),
        GeneratedSignalStep(
            step_id="step_2",
            text="When was #1 born?",
            depends_on=("step_1",),
        ),
        GeneratedSignalStep(step_id="step_3", text="Who directed Elian Voss?"),
        GeneratedSignalStep(
            step_id="step_4",
            text="When was #3 born?",
            depends_on=("step_3",),
        ),
    )
    answerer = RecordingAnswerer()
    budget = _budget().model_copy(
        update={
            "max_iterations": 4,
            "max_candidates_per_question": 20,
            "max_parent_units_per_question": 8,
            "max_rerank_pairs_per_question": 20,
        }
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(
            dataset="2wiki",
            question="Which film has the older director?",
        ),
        controller=SubquestionChainController(
            generator=StaticDAGGenerator(
                steps,
                FinalOperation(
                    type="compare",
                    inputs=("step_2", "step_4"),
                    instruction="Return the film whose director was born earlier.",
                ),
            ),
            answerer=answerer,
        ),
        budget=budget,
    )

    assert answerer.step_ids == ["step_1", "step_3"]
    assert [answer.step_id for answer in result.controller_state.step_answers] == [
        "step_1",
        "step_3",
    ]
    assert len(result.steps) == result.retrieval_state.iteration == 4


def test_chain_break_retries_the_same_step_before_advancing(tmp_path: Path) -> None:
    class SequentialAnswerer:
        def __init__(self) -> None:
            self.answers = iter(("", "Elian Voss"))

        def answer(self, *_args) -> str:
            return next(self.answers)

    retry_budget = _budget().model_copy(
        update={
            "max_iterations": 3,
            "max_candidates_per_question": 15,
            "max_parent_units_per_question": 6,
            "max_rerank_pairs_per_question": 15,
        }
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=SubquestionChainController(
            generator=StaticChainGenerator(_chain()),
            answerer=SequentialAnswerer(),
            max_chain_steps=2,
            max_step_retries=1,
        ),
        budget=retry_budget,
    )

    assert result.steps[0].decision.signal_text == "Who directed The Silent Map?"
    assert "Current relation: Who directed The Silent Map?" in result.steps[1].decision.signal_text
    assert "Original question:" in result.steps[1].decision.signal_text
    assert "#" not in result.steps[1].decision.signal_text
    assert result.steps[2].decision.signal_text == "Who is the mother of Elian Voss?"
    assert [step.decision.fallback for step in result.steps] == [False, True, False]
    assert result.steps[0].outcome.reason == (
        "empty intermediate answer; retry current chain step"
    )
    assert [answer.answer for answer in result.controller_state.step_answers] == [
        "Elian Voss"
    ]


def test_openai_retrieval_question_uses_evidence_state_and_records_usage(
    tmp_path: Path,
) -> None:
    client = FakeLLMClient(
        ['{"retrieval_question":"Who is Elian Voss\'s mother?"}']
    )
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        controller=RetrievalQuestionController(
            generator=OpenAIRetrievalQuestionGenerator(
                client=client,
                system_prompt="next question",
            ),
            max_steps=2,
        ),
        budget=_llm_budget(),
    )

    assert result.steps[0].decision.signal_text == QUESTION
    assert result.steps[0].decision.call_records == ()
    assert result.steps[1].decision.signal_text == "Who is Elian Voss's mother?"
    assert result.steps[1].decision.selected_by == "OpenAIRetrievalQuestionGenerator"
    assert result.retrieval_state.total_usage.llm_calls == 1
    request = json.loads(client.messages[0][1].content)
    assert set(request) == {
        "original_question",
        "iteration",
        "previous_retrieval_questions",
        "current_evidence",
    }
    assert request["current_evidence"]


def test_retrieval_question_default_window_keeps_legacy_oldest_evidence() -> None:
    history = tuple(
        EvidenceUnit(
            evidence_id=f"history_{index}",
            paragraph_id=f"history_p{index}",
            sentence_id=f"history_{index}",
            title=f"History {index}",
            text=f"Historical fact {index}.",
        )
        for index in range(10)
    )
    current = tuple(
        EvidenceUnit(
            evidence_id=f"current_{index}",
            paragraph_id=f"current_p{index}",
            sentence_id=f"current_{index}",
            title=f"Current {index}",
            text=f"Current fact {index}.",
        )
        for index in range(6)
    )
    state = RetrievalState.initial(
        _llm_budget().model_copy(update={"max_retained_evidence": 20})
    ).model_copy(
        update={
            "retained_evidence": (*history, *current[:3]),
            "current_evidence": current,
        }
    )
    generator = OpenAIRetrievalQuestionGenerator(
        client=FakeLLMClient([]),
        system_prompt="next question",
        evidence_limit=10,
    )

    request = json.loads(
        generator._user_prompt(
            InferencePayload(dataset="musique", question=QUESTION),
            state,
            2,
        )
    )

    assert [item["title"] for item in request["current_evidence"]] == [
        f"History {index}" for index in range(10)
    ]


def test_retrieval_question_balanced_window_prioritizes_current_and_keeps_history() -> None:
    history = tuple(
        EvidenceUnit(
            evidence_id=f"history_{index}",
            paragraph_id=f"history_p{index}",
            sentence_id=f"history_{index}",
            title=f"History {index}",
            text=f"Historical fact {index}.",
        )
        for index in range(10)
    )
    current = tuple(
        EvidenceUnit(
            evidence_id=f"current_{index}",
            paragraph_id=f"current_p{index}",
            sentence_id=f"current_{index}",
            title=f"Current {index}",
            text=f"Current fact {index}.",
        )
        for index in range(6)
    )
    state = RetrievalState.initial(
        _llm_budget().model_copy(update={"max_retained_evidence": 20})
    ).model_copy(
        update={
            "retained_evidence": (*history, *current[:3]),
            "current_evidence": current,
        }
    )
    generator = OpenAIRetrievalQuestionGenerator(
        client=FakeLLMClient([]),
        system_prompt="next question",
        evidence_limit=10,
        evidence_window_mode="latest_balanced",
    )

    request = json.loads(
        generator._user_prompt(
            InferencePayload(dataset="musique", question=QUESTION),
            state,
            2,
        )
    )
    titles = [item["title"] for item in request["current_evidence"]]

    assert titles == [
        *(f"Current {index}" for index in range(5)),
        *(f"History {index}" for index in range(5)),
    ]
    assert len(titles) == len(set(titles)) == 10


def test_intermediate_answer_prompt_prioritizes_latest_step_evidence() -> None:
    old = EvidenceUnit(
        evidence_id="old_s0",
        paragraph_id="old",
        sentence_id="old_s0",
        title="Old evidence",
        text="An earlier retrieval fact.",
    )
    latest = EvidenceUnit(
        evidence_id="latest_s0",
        paragraph_id="latest",
        sentence_id="latest_s0",
        title="Latest evidence",
        text="The newest bridge fact.",
    )
    state = RetrievalState.initial(_llm_budget()).model_copy(
        update={
            "candidate_pool": (
                EvidenceCandidate(
                    evidence=latest,
                    sources=("test",),
                    coarse_score=1.0,
                ),
            ),
            "retained_evidence": (old, latest),
            "current_evidence": (latest,),
        }
    )
    client = FakeLLMClient(['{"answer":"bridge"}'])
    answerer = OpenAIIntermediateAnswerer(
        client=client,
        system_prompt="answer",
    )

    answerer.answer(
        InferencePayload(dataset="2wiki", question=QUESTION),
        GeneratedSignalStep(step_id="step_2", text="What is the bridge?"),
        state,
    )
    request = json.loads(client.messages[0][1].content)

    assert [item["title"] for item in request["retained_evidence"]] == [
        "Latest evidence",
        "Old evidence",
    ]


def test_rule_llm_and_hybrid_router_are_question_only_and_reproducible() -> None:
    rule = RuleRouter()
    assert rule.route(InferencePayload(dataset="2wiki", question=QUESTION)).mode == (
        "subquestion_chain"
    )
    comparison = InferencePayload(
        dataset="2wiki",
        question="Which film has the earlier-born director, Gendernauts or I Am Guilty?",
    )
    assert rule.route(comparison).mode == "retrieval_question"
    assert parse_router_output('{"signal_type":"chain","reason":"nested"}') == (
        "subquestion_chain",
        "nested",
        False,
    )

    calls: list[InferencePayload] = []

    def classify(payload: InferencePayload) -> str:
        calls.append(payload)
        return '{"mode":"subquestion_chain","reason":"LLM gray-zone decision"}'

    hybrid = HybridRouter(LLMRouter(classify))
    decision = hybrid.route(
        InferencePayload(dataset="hotpotqa", question="Where was Atlas written?")
    )
    assert decision.mode == "subquestion_chain"
    assert decision.selected_by == "hybrid_router"
    assert len(calls) == 1
    assert calls[0].model_dump(mode="json") == {
        "dataset": "hotpotqa",
        "question": "Where was Atlas written?",
    }


def test_auto_runner_never_constructs_non_selected_controller(tmp_path: Path) -> None:
    constructed = {"retrieval_question": 0, "subquestion_chain": 0}

    def retrieval_factory() -> RetrievalQuestionController:
        constructed["retrieval_question"] += 1
        return RetrievalQuestionController(max_steps=2)

    def chain_factory() -> SubquestionChainController:
        constructed["subquestion_chain"] += 1
        return SubquestionChainController(
            generator=StaticChainGenerator(_chain()),
            answerer=StaticIntermediateAnswerer(
                {"step_1": "Elian Voss", "step_2": "Rhea Voss"}
            ),
        )

    result = OnlineAutoRunner(
        engine=_engine(tmp_path),
        router=RuleRouter(),
        controller_factories={
            "retrieval_question": retrieval_factory,
            "subquestion_chain": chain_factory,
        },
    ).run(
        payload=InferencePayload(dataset="2wiki", question=QUESTION),
        budget=_budget(),
    )

    assert result.router.mode == "subquestion_chain"
    assert result.run.controller_type == "subquestion_chain"
    assert constructed == {"retrieval_question": 0, "subquestion_chain": 1}
