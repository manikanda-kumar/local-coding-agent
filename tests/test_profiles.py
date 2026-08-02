import pytest

from agent_runtime import (
    MODEL_PROFILES,
    AgentRunner,
    CapabilityGateway,
    ChatResponse,
    ContextBudget,
    InMemoryAuditSink,
    InMemoryCapabilityCatalog,
    InvocationContext,
    ModelRequestProfile,
    StaticPolicyEngine,
)


def test_builtin_local_profiles_are_explicit_and_immutable() -> None:
    assert set(MODEL_PROFILES) == {
        "minimax-m2.7-vllm",
        "gemma-4-31b-it-vllm",
        "glm-5.2-vllm",
        "kimi-k3-vllm-agentic",
    }
    assert all(profile.max_output_tokens for profile in MODEL_PROFILES.values())
    assert MODEL_PROFILES["minimax-m2.7-vllm"].top_k == 40
    assert MODEL_PROFILES["glm-5.2-vllm"].tool_parser == "glm47"
    assert MODEL_PROFILES["kimi-k3-vllm-agentic"].request_extensions()["reasoning_effort"] == "max"
    assert MODEL_PROFILES["gemma-4-31b-it-vllm"].reasoning_parser == "gemma4"
    with pytest.raises(TypeError):
        MODEL_PROFILES["other"] = MODEL_PROFILES["kimi-k3-vllm-agentic"]


def test_profile_bounds_and_freezes_allowlisted_extensions() -> None:
    source = {"chat_template_kwargs": {"enable_thinking": True}}
    profile = ModelRequestProfile(0.1, 1024, 0.9, extensions=source)
    source["chat_template_kwargs"]["enable_thinking"] = False
    assert profile.request_extensions()["chat_template_kwargs"]["enable_thinking"] is True
    with pytest.raises(ValueError, match="unsupported request extensions"):
        ModelRequestProfile(0.1, 1024, 0.9, extensions={"model": "override"})
    with pytest.raises(ValueError, match="max_tokens"):
        ModelRequestProfile(0.1, 1.5, 0.9)
    with pytest.raises(ValueError, match="context window"):
        ModelRequestProfile(0.1, 1024, 0.9, context_window_tokens=1024)


def test_runner_applies_profile_to_every_model_request() -> None:
    class CapturingProvider:
        name = "fixture"

        def __init__(self):
            self.requests = []

        def chat(self, model, request):
            self.requests.append((model, request))
            return ChatResponse("done", model, self.name)

        def close(self):
            pass

    provider = CapturingProvider()
    gateway = CapabilityGateway(
        InMemoryCapabilityCatalog(()),
        StaticPolicyEngine(),
        InMemoryAuditSink(),
        InvocationContext("principal", "run", "ANALYZE"),
    )
    profile = MODEL_PROFILES["minimax-m2.7-vllm"]
    assert AgentRunner(provider, "minimax", gateway, profile=profile).run("task") == "done"
    request = provider.requests[0][1]
    assert (request.temperature, request.top_p, request.top_k, request.timeout) == (
        1.0,
        0.95,
        40,
        300,
    )
    assert request.reasoning_mode == "structured"
    with pytest.raises(ValueError, match="exceeds the model profile"):
        AgentRunner(
            provider,
            "minimax",
            gateway,
            profile=profile,
            context_budget=ContextBudget(
                profile.context_window_tokens + 1,
                profile.max_output_tokens,
            ),
        )
    with pytest.raises(ValueError, match="exceeds the model profile"):
        AgentRunner(
            provider,
            "minimax",
            gateway,
            profile=profile,
            context_budget=ContextBudget(
                profile.context_window_tokens,
                profile.max_output_tokens - 1,
            ),
        )
