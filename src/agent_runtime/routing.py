from __future__ import annotations

from dataclasses import dataclass

from agent_runtime.models import ModelCapabilities
from agent_runtime.providers.base import Provider


@dataclass(frozen=True, slots=True)
class ModelDeployment:
    provider: Provider
    model: str
    capabilities: ModelCapabilities
    priority: int = 100


@dataclass(frozen=True, slots=True)
class RoutingRequirements:
    minimum_context_window: int = 0
    tools: bool = False
    streaming: bool = False


class ModelRouter:
    def __init__(self, deployments: tuple[ModelDeployment, ...]) -> None:
        if not deployments:
            raise ValueError("at least one model deployment is required")
        self._deployments = deployments

    def select(self, requirements: RoutingRequirements | None = None) -> ModelDeployment:
        requirements = requirements or RoutingRequirements()
        candidates = [
            deployment
            for deployment in self._deployments
            if deployment.capabilities.context_window >= requirements.minimum_context_window
            and (not requirements.tools or deployment.capabilities.supports_tools)
            and (not requirements.streaming or deployment.capabilities.supports_streaming)
        ]
        if not candidates:
            raise LookupError("no model deployment satisfies the routing requirements")
        return min(candidates, key=self._sort_key)

    @staticmethod
    def _sort_key(deployment: ModelDeployment) -> tuple[float, int]:
        capabilities = deployment.capabilities
        known_cost = (
            capabilities.input_cost_per_million is not None
            and capabilities.output_cost_per_million is not None
        )
        average_cost = (
            (capabilities.input_cost_per_million + capabilities.output_cost_per_million) / 2
            if known_cost
            else float("inf")
        )
        return average_cost, deployment.priority
