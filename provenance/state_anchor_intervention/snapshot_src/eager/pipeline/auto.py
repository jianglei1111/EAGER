from __future__ import annotations

from collections.abc import Callable, Mapping

from pydantic import BaseModel, ConfigDict

from eager.pipeline.engine import RetrievalEngine, RetrievalRunResult
from eager.schemas import BudgetContract, InferencePayload
from eager.signals import RouterDecision, SignalController, SignalRouter


class RoutedRetrievalRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    router: RouterDecision
    run: RetrievalRunResult


class OnlineAutoRunner:
    """Route before construction and execute exactly one controller."""

    def __init__(
        self,
        *,
        engine: RetrievalEngine,
        router: SignalRouter,
        controller_factories: Mapping[str, Callable[[], SignalController]],
    ) -> None:
        self.engine = engine
        self.router = router
        self.controller_factories = dict(controller_factories)

    def run(
        self,
        *,
        payload: InferencePayload,
        budget: BudgetContract,
    ) -> RoutedRetrievalRun:
        decision = self.router.route(payload)
        factory = self.controller_factories.get(decision.mode)
        if factory is None:
            raise ValueError(f"no controller factory registered for {decision.mode}")
        controller = factory()
        result = self.engine.run(payload=payload, controller=controller, budget=budget)
        if result.controller_type != decision.mode:
            raise ValueError("router mode and constructed controller type differ")
        return RoutedRetrievalRun(router=decision, run=result)


__all__ = ["OnlineAutoRunner", "RoutedRetrievalRun"]
