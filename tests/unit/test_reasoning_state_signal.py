from __future__ import annotations

import json
from pathlib import Path

from eager.content_identity import make_paragraph_id
from eager.indexing import CorpusRecord, CorpusSnapshot, build_tiny_index
from eager.pipeline import RetrievalEngine
from eager.retrieval import RetrievalKernel, RetrievalKernelConfig, TinySemanticIndex
from eager.runtime.llm_client import LLMResponse, LLMUsage
from eager.schemas import (
    BudgetContract,
    EvidenceUnit,
    InferencePayload,
    RetrievalState,
    SignalDecision,
)
from eager.signals import (
    OpenAIReasoningStateGenerator,
    ReasoningStateController,
    compose_reasoning_signal,
    extract_reasoning_state,
)


QUESTION = "Where was the author of The Old Man and the Sea born?"
THOUGHT = "The Old Man and the Sea was written by Ernest Hemingway."


class FakeLLMClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.messages: list[tuple[object, ...]] = []

    def complete(self, messages, **_kwargs) -> LLMResponse:
        self.messages.append(tuple(messages))
        content = self.outputs.pop(0)
        return LLMResponse(
            content=content,
            model="fake-model",
            finish_reason="stop",
            usage=LLMUsage(prompt_tokens=17, completion_tokens=5, total_tokens=22),
            request_id=f"request-{len(self.messages)}",
            response_sha256=str(len(self.messages)) * 64,
            attempts=1,
            elapsed_seconds=0.01,
        )


def _corpus() -> CorpusSnapshot:
    records = []
    for title, sentence in (
        (
            "The Old Man and the Sea",
            "The Old Man and the Sea was written by Ernest Hemingway.",
        ),
        ("Ernest Hemingway", "Ernest Hemingway was born in Oak Park, Illinois."),
    ):
        records.append(
            CorpusRecord(
                paragraph_id=make_paragraph_id(
                    dataset="hotpotqa",
                    title=title,
                    content=(sentence,),
                ),
                title=title,
                text=sentence,
                sentences=(sentence,),
                source_dataset="hotpotqa",
                source_split="synthetic_reasoning_state",
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
                candidate_limit=2,
                paragraph_limit=2,
                rerank_limit=2,
                evidence_limit=2,
            ),
        )
    )


def _budget() -> BudgetContract:
    return BudgetContract(
        max_iterations=2,
        max_candidates_per_step=2,
        max_candidates_per_question=4,
        max_parent_units_per_step=2,
        max_parent_units_per_question=4,
        max_rerank_pairs_per_step=2,
        max_rerank_pairs_per_question=4,
        max_retained_evidence=2,
        max_llm_calls_per_step=1,
        max_llm_calls_per_question=2,
        max_input_tokens_per_question=100,
        max_output_tokens_per_question=100,
    )


def test_reasoning_signal_is_reversible_without_synthetic_labels() -> None:
    signal = compose_reasoning_signal(QUESTION, THOUGHT)

    assert signal == f"{QUESTION}\n{THOUGHT}"
    assert extract_reasoning_state(signal, question=QUESTION) == THOUGHT
    assert extract_reasoning_state(
        f"A different question?\n{THOUGHT}",
        question=QUESTION,
    ) == ""
    assert "Original question:" not in signal
    assert "Reasoning state:" not in signal


def test_openai_reasoning_state_uses_only_inference_visible_fields() -> None:
    client = FakeLLMClient([json.dumps({"reasoning_state": THOUGHT})])
    generator = OpenAIReasoningStateGenerator(
        client=client,
        system_prompt="generate a reasoning state",
    )
    payload = InferencePayload(dataset="hotpotqa", question=QUESTION)
    evidence = EvidenceUnit(
        evidence_id="book_s0",
        paragraph_id="book",
        sentence_id="book_s0",
        title="The Old Man and the Sea",
        text=THOUGHT,
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={
            "retained_evidence": (evidence,),
            "current_evidence": (evidence,),
        }
    )

    generated = generator.generate(payload, retrieval_state, 2)

    request = json.loads(client.messages[0][1].content)
    assert set(request) == {
        "original_question",
        "iteration",
        "previous_reasoning_states",
        "retained_evidence",
    }
    assert "answer" not in request
    assert "supporting" not in request
    assert generated.text == THOUGHT
    assert generated.llm_calls == 1
    assert generated.input_tokens == 17
    assert generated.output_tokens == 5


def test_reasoning_state_parser_rejects_extra_fields_and_answer_markers() -> None:
    parse = OpenAIReasoningStateGenerator._parse_text

    assert parse(json.dumps({"reasoning_state": THOUGHT, "answer": "Oak Park"})) == ""
    assert parse(json.dumps({"reasoning_state": "The final answer is Oak Park."})) == ""
    assert parse(json.dumps({"reasoning_state": [THOUGHT]})) == ""
    assert parse(f"Result: {json.dumps({'reasoning_state': THOUGHT})}") == ""
    assert OpenAIReasoningStateGenerator._is_explicit_empty(
        json.dumps({"reasoning_state": ""})
    ) is True
    assert OpenAIReasoningStateGenerator._is_explicit_empty("not json") is False


def test_reasoning_state_prompt_recovers_only_prior_rs_history() -> None:
    client = FakeLLMClient([json.dumps({"reasoning_state": THOUGHT})])
    generator = OpenAIReasoningStateGenerator(
        client=client,
        system_prompt="generate a reasoning state",
    )
    prior_state = "The Old Man and the Sea is a novel by Ernest Hemingway."
    history = (
        SignalDecision(
            decision_id="rs-history",
            controller_type="reasoning_state",
            iteration=2,
            action="retrieve",
            reason="test reasoning state",
            signal_text=compose_reasoning_signal(QUESTION, prior_state),
        ),
        SignalDecision(
            decision_id="rq-history",
            controller_type="retrieval_question",
            iteration=1,
            action="retrieve",
            reason="unrelated controller",
            signal_text="Where was Ernest Hemingway born?",
        ),
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={"signal_history": history}
    )

    request = json.loads(generator._user_prompt(
        InferencePayload(dataset="hotpotqa", question=QUESTION),
        retrieval_state,
        3,
    ))

    assert request["previous_reasoning_states"] == [prior_state]


def test_reasoning_state_rejects_unsupported_comparison() -> None:
    unsupported = (
        "Ernest Hemingway was born earlier than another author."
    )
    client = FakeLLMClient([json.dumps({"reasoning_state": unsupported})])
    generator = OpenAIReasoningStateGenerator(
        client=client,
        system_prompt="generate a reasoning state",
    )
    evidence = EvidenceUnit(
        evidence_id="book_s0",
        paragraph_id="book",
        sentence_id="book_s0",
        title="The Old Man and the Sea",
        text=THOUGHT,
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={"retained_evidence": (evidence,), "current_evidence": (evidence,)}
    )

    generated = generator.generate(
        InferencePayload(dataset="hotpotqa", question=QUESTION),
        retrieval_state,
        2,
    )

    assert generated.text == ""
    assert generated.fallback is True
    assert generated.call_records[0]["validation_status"] == "ungrounded_or_unsupported_operation"


def test_reasoning_state_rejects_repeated_history() -> None:
    client = FakeLLMClient([json.dumps({"reasoning_state": THOUGHT})])
    generator = OpenAIReasoningStateGenerator(
        client=client,
        system_prompt="generate a reasoning state",
    )
    evidence = EvidenceUnit(
        evidence_id="book_s0",
        paragraph_id="book",
        sentence_id="book_s0",
        title="The Old Man and the Sea",
        text=THOUGHT,
    )
    prior = SignalDecision(
        decision_id="rs-history",
        controller_type="reasoning_state",
        iteration=2,
        action="retrieve",
        reason="test reasoning state",
        signal_text=compose_reasoning_signal(QUESTION, THOUGHT),
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={
            "retained_evidence": (evidence,),
            "current_evidence": (evidence,),
            "signal_history": (prior,),
        }
    )

    generated = generator.generate(
        InferencePayload(dataset="hotpotqa", question=QUESTION),
        retrieval_state,
        3,
    )

    assert generated.text == ""
    assert generated.fallback is True


def test_reasoning_state_rejects_direct_answer_slot() -> None:
    question = "Who is Ada Lovelace's mother?"
    direct_answer = "Ada Lovelace's mother was Anne Isabella Milbanke."
    client = FakeLLMClient([json.dumps({"reasoning_state": direct_answer})])
    generator = OpenAIReasoningStateGenerator(
        client=client,
        system_prompt="generate a reasoning state",
    )
    evidence = EvidenceUnit(
        evidence_id="ada_s0",
        paragraph_id="ada",
        sentence_id="ada_s0",
        title="Ada Lovelace",
        text=direct_answer,
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={"retained_evidence": (evidence,), "current_evidence": (evidence,)}
    )

    generated = generator.generate(
        InferencePayload(dataset="hotpotqa", question=question),
        retrieval_state,
        2,
    )

    assert generated.text == ""
    assert generated.fallback is True
    assert generated.call_records[0]["validation_status"] == "fills_answer_slot"


def test_reasoning_state_prompt_prioritizes_exact_question_title() -> None:
    question = "Which film has the earlier director, Film A or Film B?"
    exact = EvidenceUnit(
        evidence_id="exact_s0",
        paragraph_id="exact",
        sentence_id="exact_s0",
        title="Film A",
        text="Film A was directed by Director A.",
    )
    variant = EvidenceUnit(
        evidence_id="variant_s0",
        paragraph_id="variant",
        sentence_id="variant_s0",
        title="Film A (1921 film)",
        text="Film A is a 1921 film directed by Director B.",
    )
    generator = OpenAIReasoningStateGenerator(
        client=FakeLLMClient([json.dumps({"reasoning_state": exact.text})]),
        system_prompt="generate a reasoning state",
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={"retained_evidence": (variant, exact), "current_evidence": (variant, exact)}
    )

    request = json.loads(generator._user_prompt(
        InferencePayload(dataset="hotpotqa", question=question),
        retrieval_state,
        2,
    ))

    assert [item["title"] for item in request["retained_evidence"]] == [
        "Film A",
        "Film A (1921 film)",
    ]


def test_possessive_film_title_is_not_treated_as_answer_slot() -> None:
    question = "Which film came out first, Bhavni Bhavai or Matthew's Days?"
    state = "Matthew's Days is a 1968 Polish drama film."
    client = FakeLLMClient([json.dumps({"reasoning_state": state})])
    generator = OpenAIReasoningStateGenerator(
        client=client,
        system_prompt="generate a reasoning state",
    )
    evidence = EvidenceUnit(
        evidence_id="matthew_s0",
        paragraph_id="matthew",
        sentence_id="matthew_s0",
        title="Matthew's Days",
        text=state,
    )
    retrieval_state = RetrievalState.initial(_budget()).model_copy(
        update={"retained_evidence": (evidence,), "current_evidence": (evidence,)}
    )

    generated = generator.generate(
        InferencePayload(dataset="2wiki", question=question),
        retrieval_state,
        2,
    )

    assert generated.text == state
    assert generated.fallback is False


def test_reasoning_state_controller_composes_question_and_thought(tmp_path: Path) -> None:
    client = FakeLLMClient([json.dumps({"reasoning_state": THOUGHT})])
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="hotpotqa", question=QUESTION),
        controller=ReasoningStateController(
            generator=OpenAIReasoningStateGenerator(
                client=client,
                system_prompt="generate a reasoning state",
            ),
            max_steps=2,
        ),
        budget=_budget(),
    )

    assert result.steps[0].decision.signal_text == QUESTION
    assert result.steps[0].decision.call_records == ()
    assert result.steps[1].decision.signal_text == f"{QUESTION}\n{THOUGHT}"
    assert result.steps[1].decision.controller_type == "reasoning_state"
    assert result.steps[1].decision.selected_by == "OpenAIReasoningStateGenerator"
    assert result.steps[1].decision.fallback is False
    assert result.retrieval_state.total_usage.llm_calls == 1
    assert result.steps[-1].outcome.reason == "reasoning-state step limit reached"
    assert result.controller_state.stop_reason == "reasoning-state step limit reached"


def test_reasoning_state_invalid_json_falls_back_to_original_question(
    tmp_path: Path,
) -> None:
    result = _engine(tmp_path).run(
        payload=InferencePayload(dataset="hotpotqa", question=QUESTION),
        controller=ReasoningStateController(
            generator=OpenAIReasoningStateGenerator(
                client=FakeLLMClient(["not json"]),
                system_prompt="generate a reasoning state",
            ),
            max_steps=2,
        ),
        budget=_budget(),
    )

    assert result.steps[1].decision.signal_text == QUESTION
    assert result.steps[1].decision.fallback is True
