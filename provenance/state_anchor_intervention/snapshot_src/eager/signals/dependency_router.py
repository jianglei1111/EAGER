from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SignalMode = Literal["retrieval_question", "subquestion_chain"]


class DependencyRouterError(ValueError):
    pass


class DependencyStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    id: int = Field(ge=1)
    question: str = Field(min_length=1)
    depends_on: tuple[int, ...] = ()


class QuestionDependencyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    steps: tuple[DependencyStep, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_dependencies(self) -> "QuestionDependencyPlan":
        step_ids = [step.id for step in self.steps]
        expected = list(range(1, len(self.steps) + 1))
        if step_ids != expected:
            raise ValueError(f"step IDs must be consecutive and ordered: {expected}")

        prior: set[int] = set()
        for step in self.steps:
            if len(step.depends_on) != len(set(step.depends_on)):
                raise ValueError(f"duplicate dependency in step {step.id}")
            missing = set(step.depends_on) - prior
            if missing:
                raise ValueError(
                    f"step {step.id} has non-prior dependencies: {sorted(missing)}"
                )
            prior.add(step.id)
        return self


class DependencyRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    mode: SignalMode
    reason: str
    dependency_edge_count: int = Field(ge=0)
    maximum_dependency_depth: int = Field(ge=1)
    independent_root_count: int = Field(default=1, ge=1)


def _json_object(raw_output: str) -> dict[str, object]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise DependencyRouterError("planner response does not contain a JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise DependencyRouterError(f"planner response is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise DependencyRouterError("planner response root must be an object")
    return payload


def _step_id(value: object) -> object:
    if isinstance(value, str):
        match = re.fullmatch(r"s?(\d+)", value.strip(), flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return value


def parse_dependency_plan(raw_output: str) -> QuestionDependencyPlan:
    payload = _json_object(raw_output)
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, list):
        normalized_steps = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                normalized_steps.append(raw_step)
                continue
            step = dict(raw_step)
            step["id"] = _step_id(step.get("id"))
            dependencies = step.get("depends_on", [])
            if isinstance(dependencies, list):
                step["depends_on"] = [_step_id(value) for value in dependencies]
            normalized_steps.append(step)
        payload["steps"] = normalized_steps
    try:
        return QuestionDependencyPlan.model_validate(payload)
    except ValueError as exc:
        raise DependencyRouterError(
            f"planner response violates the dependency schema: {exc}"
        ) from exc


def route_dependency_plan(plan: QuestionDependencyPlan) -> DependencyRouteDecision:
    depths: dict[int, int] = {}
    edge_count = 0
    root_count = 0
    for step in plan.steps:
        if not step.depends_on:
            root_count += 1
        edge_count += len(step.depends_on)
        depths[step.id] = 1 + max(
            (depths[dependency] for dependency in step.depends_on),
            default=0,
        )

    maximum_depth = max(depths.values())
    if edge_count:
        return DependencyRouteDecision(
            mode="subquestion_chain",
            reason=(
                "at least one retrieval question requires a factual answer from an "
                "earlier retrieval step"
            ),
            dependency_edge_count=edge_count,
            maximum_dependency_depth=maximum_depth,
            independent_root_count=root_count,
        )
    return DependencyRouteDecision(
        mode="retrieval_question",
        reason="all retrieval questions can be issued independently from the original question",
        dependency_edge_count=0,
        maximum_dependency_depth=maximum_depth,
        independent_root_count=root_count,
    )


def route_dependency_plan_v2(
    plan: QuestionDependencyPlan,
) -> DependencyRouteDecision:
    depths: dict[int, int] = {}
    edge_count = 0
    root_count = 0
    for step in plan.steps:
        if not step.depends_on:
            root_count += 1
        edge_count += len(step.depends_on)
        depths[step.id] = 1 + max(
            (depths[dependency] for dependency in step.depends_on),
            default=0,
        )

    maximum_depth = max(depths.values())
    if edge_count == 0:
        return DependencyRouteDecision(
            mode="retrieval_question",
            reason="the plan contains no answer-dependent retrieval step",
            dependency_edge_count=0,
            maximum_dependency_depth=maximum_depth,
            independent_root_count=root_count,
        )
    if root_count >= 2:
        return DependencyRouteDecision(
            mode="retrieval_question",
            reason=(
                "the answer requires multiple independently retrievable evidence branches"
            ),
            dependency_edge_count=edge_count,
            maximum_dependency_depth=maximum_depth,
            independent_root_count=root_count,
        )
    return DependencyRouteDecision(
        mode="subquestion_chain",
        reason="one evidence branch must resolve factual answers in sequence",
        dependency_edge_count=edge_count,
        maximum_dependency_depth=maximum_depth,
        independent_root_count=1,
    )


__all__ = [
    "DependencyRouteDecision",
    "DependencyRouterError",
    "DependencyStep",
    "QuestionDependencyPlan",
    "parse_dependency_plan",
    "route_dependency_plan",
    "route_dependency_plan_v2",
]
