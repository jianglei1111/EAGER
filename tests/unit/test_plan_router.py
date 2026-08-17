from __future__ import annotations

import pytest

from eager.signals.plan_router import (
    PlanRouterError,
    QuestionPlan,
    parse_question_plan,
    route_question_plan,
)


def test_parallel_branch_plan_routes_to_retrieval_question() -> None:
    plan = QuestionPlan.model_validate(
        {
            "steps": [
                {"step_id": "s1", "operation": "lookup", "query": "director A"},
                {"step_id": "s2", "operation": "lookup", "query": "director B"},
                {
                    "step_id": "s3",
                    "operation": "lookup",
                    "query": "birth date A",
                    "depends_on": ["s1"],
                },
                {
                    "step_id": "s4",
                    "operation": "lookup",
                    "query": "birth date B",
                    "depends_on": ["s2"],
                },
                {
                    "step_id": "s5",
                    "operation": "temporal_compare",
                    "query": "compare birth dates",
                    "depends_on": ["s3", "s4"],
                },
            ],
            "final_step": "s5",
        }
    )

    decision = route_question_plan(plan)

    assert decision.mode == "retrieval_question"
    assert decision.maximum_dependency_depth == 3
    assert decision.final_dependency_count == 2


def test_single_answer_dependent_path_routes_to_subquestion_chain() -> None:
    plan = QuestionPlan.model_validate(
        {
            "steps": [
                {"step_id": "s1", "operation": "lookup", "query": "find creator"},
                {
                    "step_id": "s2",
                    "operation": "lookup",
                    "query": "find creator spouse",
                    "depends_on": ["s1"],
                },
            ],
            "final_step": "s2",
        }
    )

    assert route_question_plan(plan).mode == "subquestion_chain"


def test_plan_parser_accepts_fenced_json_and_normalizes_operation_alias() -> None:
    plan = parse_question_plan(
        """```json
        {"steps":[{"step_id":"s1","operation":"find","query":"find entity"}],
         "final_step":"s1"}
        ```"""
    )

    assert plan.steps[0].operation == "lookup"
    assert route_question_plan(plan).mode == "retrieval_question"


def test_plan_parser_rejects_forward_dependency() -> None:
    with pytest.raises(PlanRouterError, match="non-prior dependencies"):
        parse_question_plan(
            '{"steps":['
            '{"step_id":"s1","operation":"lookup","query":"x","depends_on":["s2"]},'
            '{"step_id":"s2","operation":"lookup","query":"y"}],'
            '"final_step":"s2"}'
        )
