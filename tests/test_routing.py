from dataclasses import dataclass

import pytest

from agent_runtime import ModelCapabilities, ModelDeployment, ModelRouter, RoutingRequirements


@dataclass
class StubProvider:
    name: str

    def chat(self, model, request):  # pragma: no cover
        raise NotImplementedError

    def close(self):
        pass


def deployment(
    name: str,
    *,
    context: int,
    tools: bool,
    cost: float | None,
    priority: int = 100,
) -> ModelDeployment:
    return ModelDeployment(
        provider=StubProvider(name),
        model=name,
        capabilities=ModelCapabilities(
            context_window=context,
            supports_tools=tools,
            input_cost_per_million=cost,
            output_cost_per_million=cost,
        ),
        priority=priority,
    )


def test_router_filters_capabilities_then_chooses_lowest_known_cost() -> None:
    router = ModelRouter(
        (
            deployment("small", context=8_000, tools=True, cost=0.1),
            deployment("large", context=128_000, tools=True, cost=1.0),
            deployment("local", context=128_000, tools=False, cost=0.0),
        )
    )

    selected = router.select(RoutingRequirements(minimum_context_window=32_000, tools=True))

    assert selected.model == "large"


def test_router_uses_priority_when_cost_is_unknown() -> None:
    router = ModelRouter(
        (
            deployment("fallback", context=32_000, tools=False, cost=None, priority=20),
            deployment("preferred", context=32_000, tools=False, cost=None, priority=10),
        )
    )

    assert router.select().model == "preferred"


def test_router_reports_when_no_deployment_matches() -> None:
    router = ModelRouter((deployment("chat-only", context=8_000, tools=False, cost=0.1),))

    with pytest.raises(LookupError, match="no model deployment"):
        router.select(RoutingRequirements(tools=True))
