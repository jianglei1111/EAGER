from __future__ import annotations

import pytest

from eager.signals.dependency_router import (
    DependencyRouterError,
    QuestionDependencyPlan,
    parse_dependency_plan,
    route_dependency_plan,
    route_dependency_plan_v2,
)


def test_independent_lookups_route_to_retrieval_question() -> None:
    plan = QuestionDependencyPlan.model_validate(
        {
            "steps": [
                {"id": 1, "question": "When was Person A born?"},
                {"id": 2, "question": "When was Person B born?"},
            ]
        }
    )

    decision = route_dependency_plan(plan)

    assert decision.mode == "retrieval_question"
    assert decision.dependency_edge_count == 0
    assert decision.maximum_dependency_depth == 1


def test_answer_dependent_lookup_routes_to_subquestion_chain() -> None:
    plan = QuestionDependencyPlan.model_validate(
        {
            "steps": [
                {"id": 1, "question": "Who created Work A?"},
                {
                    "id": 2,
                    "question": "Where was #1 born?",
                    "depends_on": [1],
                },
            ]
        }
    )

    decision = route_dependency_plan(plan)

    assert decision.mode == "subquestion_chain"
    assert decision.dependency_edge_count == 1
    assert decision.maximum_dependency_depth == 2


def test_v2_parallel_roots_route_to_retrieval_question() -> None:
    plan = QuestionDependencyPlan.model_validate(
        {
            "steps": [
                {"id": 1, "question": "Who made Product A?"},
                {"id": 2, "question": "Who made Product B?"},
                {
                    "id": 3,
                    "question": "When did #1 and #2 open a shared facility?",
                    "depends_on": [1, 2],
                },
            ]
        }
    )

    decision = route_dependency_plan_v2(plan)

    assert decision.mode == "retrieval_question"
    assert decision.independent_root_count == 2
    assert decision.maximum_dependency_depth == 2


def test_v2_single_root_candidate_filter_routes_to_subquestion_chain() -> None:
    plan = QuestionDependencyPlan.model_validate(
        {
            "steps": [
                {"id": 1, "question": "Who graduated from Academy A?"},
                {
                    "id": 2,
                    "question": "Which member of #1 was born on Date B?",
                    "depends_on": [1],
                },
            ]
        }
    )

    decision = route_dependency_plan_v2(plan)

    assert decision.mode == "subquestion_chain"
    assert decision.independent_root_count == 1


def test_v2_independent_properties_route_to_retrieval_question() -> None:
    plan = QuestionDependencyPlan.model_validate(
        {
            "steps": [
                {"id": 1, "question": "What is Person A's occupation?"},
                {"id": 2, "question": "What is Person B's occupation?"},
            ]
        }
    )

    assert route_dependency_plan_v2(plan).mode == "retrieval_question"


def test_parser_accepts_fenced_json_and_normalizes_prefixed_ids() -> None:
    plan = parse_dependency_plan(
        """```json
        {"steps":[
          {"id":"s1","question":"find creator","depends_on":[]},
          {"id":"s2","question":"find #1's birthplace","depends_on":["s1"]}
        ]}
        ```"""
    )

    assert plan.steps[0].id == 1
    assert plan.steps[1].depends_on == (1,)
    assert route_dependency_plan(plan).mode == "subquestion_chain"


def test_parser_rejects_forward_dependency() -> None:
    with pytest.raises(DependencyRouterError, match="non-prior dependencies"):
        parse_dependency_plan(
            '{"steps":['
            '{"id":1,"question":"x","depends_on":[2]},'
            '{"id":2,"question":"y","depends_on":[]}]}'
        )


def test_parser_rejects_operation_fields_from_old_schema() -> None:
    with pytest.raises(DependencyRouterError, match="dependency schema"):
        parse_dependency_plan(
            '{"steps":[{"id":1,"operation":"lookup",'
            '"question":"x","depends_on":[]}]}'
        )
