from __future__ import annotations


class ModelOutputError(RuntimeError):
    def __init__(self, message: str, *, code: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class IncompleteModelOutputError(ModelOutputError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="incomplete_output", retryable=True)


class InvalidModelOutputError(ModelOutputError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_output", retryable=True)


class ToolArgumentsError(ModelOutputError):
    def __init__(self, tool_call_id: str, message: str) -> None:
        super().__init__(message, code="invalid_tool_arguments", retryable=True)
        self.tool_call_id = tool_call_id


class ModelRoutingError(LookupError):
    code = "unsupported_model_features"
