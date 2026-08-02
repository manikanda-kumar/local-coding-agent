"""Opt-in live contract checks; never run or spend provider credits by default."""

import os

import pytest

from agent_runtime import (
    AgentRunner,
    CapabilityGateway,
    InMemoryAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    StaticPolicyEngine,
    fixture_read_capability,
)
from agent_runtime.providers import OpenAICompatibleProvider


def test_openrouter_target_models_complete_a_real_gateway_loop():
    if os.getenv("RUN_OPENROUTER_LIVE_TESTS") != "1" or not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("live OpenRouter contract test is not enabled")
    models = tuple(
        filter(
            None,
            os.getenv(
                "OPENROUTER_LIVE_MODELS",
                "google/gemma-4-31b-it,minimax/minimax-m2.7,z-ai/glm-5.2,moonshotai/kimi-k3",
            ).split(","),
        )
    )
    prompt = (
        "Use the gateway tools to search for, describe, and invoke the available read capability "
        'with key="README.md". After receiving its result, answer exactly DONE. '
        "Do not invent the result."
    )
    with OpenAICompatibleProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=300,
    ) as provider:
        for index, model in enumerate(models):
            audit = InMemoryAuditSink()
            calls = []
            fixture = fixture_read_capability()
            capability = type(fixture)(
                fixture.descriptor,
                lambda arguments, context, calls=calls, fixture=fixture: (
                    calls.append(dict(arguments)) or fixture.handler(arguments, context)
                ),
            )
            gateway = CapabilityGateway(
                InMemoryCapabilityCatalog((capability,)),
                StaticPolicyEngine(frozenset({"fixture.read"})),
                audit,
                InvocationContext(
                    "live-contract",
                    f"live-{index}",
                    "ANALYZE",
                    "story",
                    "repository",
                    "workspace",
                ),
            )
            answer = AgentRunner(provider, model, gateway, max_turns=8, max_invocations=3).run(
                prompt
            )
            assert answer.strip() == "DONE", model
            assert calls == [{"key": "README.md"}], model
            successful = [event for event in audit.events if event.outcome == "success"]
            assert {event.operation for event in successful} >= {"search", "describe", "invoke"}
            invoke = [event for event in successful if event.operation == "invoke"]
            assert len(invoke) == 1 and invoke[0].execution_id, model
