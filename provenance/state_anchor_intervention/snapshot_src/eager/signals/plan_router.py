from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PlanOperation = Literal[
    "lookup",
    "compare",
    "intersection",
    "shared_property",
    "temporal_compare",
    "set_operation",
    "aggregate",
]
SignalMode = Literal["retrieval_question", "subquestion_chain"]


class PlanRouterError(ValueError):
    pass


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    step_id: str = Field(pattern=r"^s[1-9][0-9]*$")
    operation: PlanOperation
    query: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()


class QuestionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=8)
    final_step: str

    @model_validator(mode="after")
    def validate_dag(self) -> "QuestionPlan":
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step IDs must be unique")
        if self.final_step not in step_ids:
            raise ValueError("final_step must identify a plan step")
        prior: set[str] = set()
        for step in self.steps:
            if len(step.depends_on) != len(set(step.depends_on)):
                raise ValueError(f"duplicate dependency in {step.step_id}")
            missing = set(step.depends_on) - prior
            if missing:
                raise ValueError(
                    f"step {step.step_id} has non-prior dependencies: {sorted(missing)}"
                )
            prior.add(step.step_id)
        return self


class PlanTopologyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    mode: SignalMode
    reason: str
    final_operation: PlanOperation
    maximum_dependency_depth: int = Field(ge=1)
    final_dependency_count: int = Field(ge=0)


_OPERATION_ALIASES = {
    "find": "lookup",
    "search": "lookup",
    "comparison": "compare",
    "temporal": "temporal_compare",
    "time_compare": "temporal_compare",
    "set": "set_operation",
}
_PARALLEL_OPERATIONS = {
    "compare",
    "intersection",
    "shared_property",
    "temporal_compare",
    "set_operation",
    "aggregate",
}


def _json_object(raw_output: str) -> dict[str, object]:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        raise PlanRouterError("planner response does not contain a JSON object")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise PlanRouterError(f"planner response is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlanRouterError("planner response root must be an object")
    return payload


def parse_question_plan(raw_output: str) -> QuestionPlan:
    payload = _json_object(raw_output)
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, list):
        normalized_steps = []
        for raw_step in raw_steps:
            if not isinstance(raw_step, dict):
                normalized_steps.append(raw_step)
                continue
            step = dict(raw_step)
            operation = str(step.get("operation", "")).strip().casefold()
            step["operation"] = _OPERATION_ALIASES.get(operation, operation)
            normalized_steps.append(step)
        payload["steps"] = normalized_steps
    try:
        return QuestionPlan.model_validate(payload)
    except ValueError as exc:
        raise PlanRouterError(f"planner response violates the plan schema: {exc}") from exc


def route_question_plan(plan: QuestionPlan) -> PlanTopologyDecision:
    steps = {step.step_id: step for step in plan.steps}
    depths: dict[str, int] = {}
    for step in plan.steps:
        depths[step.step_id] = 1 + max(
            (depths[dependency] for dependency in step.depends_on),
            default=0,
        )
    final = steps[plan.final_step]
    maximum_depth = max(depths.values())
    final_dependency_count = len(final.depends_on)

    if final.operation in _PARALLEL_OPERATIONS and final_dependency_count >= 2:
        return PlanTopologyDecision(
            mode="retrieval_question",
            reason=(
                "the final operation combines multiple independently planned evidence branches"
            ),
            final_operation=final.operation,
            maximum_dependency_depth=maximum_depth,
            final_dependency_count=final_dependency_count,
        )
    if maximum_depth >= 2:
        return PlanTopologyDecision(
            mode="subquestion_chain",
            reason="at least one later lookup requires the output of a prior planned step",
            final_operation=final.operation,
            maximum_dependency_depth=maximum_depth,
            final_dependency_count=final_dependency_count,
        )
    return PlanTopologyDecision(
        mode="retrieval_question",
        reason="the plan contains no answer-dependent lookup edge",
        final_operation=final.operation,
        maximum_dependency_depth=maximum_depth,
        final_dependency_count=final_dependency_count,
    )


__all__ = [
    "PlanRouterError",
    "PlanStep",
    "PlanTopologyDecision",
    "QuestionPlan",
    "parse_question_plan",
    "route_question_plan",
]
