from unittest import mock

import pytest
import ouroboros.pricing as pricing_module
from ouroboros.llm import LLMClient


def test_resolve_openai_target(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    target = LLMClient()._resolve_remote_target("openai::gpt-4.1")

    assert target["provider"] == "openai"
    assert target["resolved_model"] == "gpt-4.1"
    assert target["usage_model"] == "openai/gpt-4.1"
    assert target["base_url"] == "https://api.openai.com/v1"


def test_build_remote_kwargs_uses_max_completion_tokens_for_openai_gpt5(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs


def test_build_remote_kwargs_keeps_max_tokens_for_openai_gpt41(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-4.1")
    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        None,
    )

    assert kwargs["max_tokens"] == 512
    assert "max_completion_tokens" not in kwargs


def test_build_remote_kwargs_normalizes_tool_descriptions_for_openrouter():
    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")

    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        [{
            "type": "function",
            "function": {
                "name": "bad_tool",
                "description": ("first half ", "second half"),
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )

    assert kwargs["tools"][0]["function"]["description"] == "first half second half"


def test_build_remote_kwargs_deduplicates_tool_names_for_openrouter():
    client = LLMClient()
    target = client._resolve_remote_target("anthropic/claude-sonnet-4.6")

    kwargs = client._build_remote_kwargs(
        target,
        [{"role": "user", "content": "hi"}],
        "high",
        512,
        "auto",
        None,
        [
            {
                "type": "function",
                "function": {
                    "name": "dup_tool",
                    "description": "first",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dup_tool",
                    "description": "second",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    )

    assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["dup_tool"]
    assert kwargs["tools"][0]["function"]["description"] == "first"


def test_build_anthropic_tools_deduplicates_tool_names():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "dup_tool",
                "description": "first",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "dup_tool",
                "description": "second",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    anthropic_tools = LLMClient._build_anthropic_tools(tools)

    assert anthropic_tools == [
        {
            "name": "dup_tool",
            "description": "first",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_resolve_anthropic_target_normalizes_direct_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    target = LLMClient()._resolve_remote_target("anthropic::claude-sonnet-4.6")

    assert target["provider"] == "anthropic"
    assert target["resolved_model"] == "claude-sonnet-4-6"
    assert target["usage_model"] == "anthropic/claude-sonnet-4-6"
    assert target["base_url"] == "https://api.anthropic.com/v1"


def test_normalize_anthropic_response_maps_tool_use(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    client = LLMClient()
    target = client._resolve_remote_target("anthropic::claude-sonnet-4-6")
    message, usage = client._normalize_anthropic_response(
        {
            "content": [
                {"type": "text", "text": "Working on it."},
                {"type": "tool_use", "id": "toolu_1", "name": "echo_tool", "input": {"text": "hello"}},
            ],
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
        },
        target,
    )

    assert message["content"] == "Working on it."
    assert message["tool_calls"][0]["function"]["name"] == "echo_tool"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"text": "hello"}'
    assert usage["provider"] == "anthropic"
    assert usage["resolved_model"] == "anthropic/claude-sonnet-4-6"
    assert usage["cached_tokens"] == 3
    assert usage["cache_write_tokens"] == 2


def test_resolve_openai_compatible_target_prefers_dedicated_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "compat-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://compat.example/v1")

    target = LLMClient()._resolve_remote_target("openai-compatible::meta-llama/compatible")

    assert target["provider"] == "openai-compatible"
    assert target["api_key"] == "compat-key"
    assert target["base_url"] == "https://compat.example/v1"
    assert target["usage_model"] == "openai-compatible/meta-llama/compatible"


def test_resolve_cloudru_target_uses_default_base_url(monkeypatch):
    monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "cloudru-key")
    monkeypatch.delenv("CLOUDRU_FOUNDATION_MODELS_BASE_URL", raising=False)

    target = LLMClient()._resolve_remote_target("cloudru::giga-model")

    assert target["provider"] == "cloudru"
    assert target["api_key"] == "cloudru-key"
    assert target["base_url"] == "https://foundation-models.api.cloud.ru/v1"
    assert target["usage_model"] == "cloudru/giga-model"


def test_normalize_remote_response_estimates_cost_for_direct_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")
    seen = {}

    def fake_estimate_cost(model, prompt_tokens, completion_tokens, cached_tokens=0, cache_write_tokens=0):
        seen["args"] = (model, prompt_tokens, completion_tokens, cached_tokens, cache_write_tokens)
        return 0.123456

    monkeypatch.setattr(pricing_module, "estimate_cost", fake_estimate_cost)

    message, usage = client._normalize_remote_response(
        {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        },
        target,
    )

    assert message["content"] == "ok"
    assert usage["provider"] == "openai"
    assert usage["resolved_model"] == "openai/gpt-5.2"
    assert usage["cached_tokens"] == 10
    assert usage["cost"] == 0.123456
    assert seen["args"] == ("openai/gpt-5.2", 100, 40, 10, 0)


def test_get_remote_client_sets_explicit_retry_and_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")

    with mock.patch("openai.OpenAI") as mock_openai_cls:
        created = mock.Mock()
        mock_openai_cls.return_value = created

        result = client._get_remote_client(target)

    assert result is created
    kwargs = mock_openai_cls.call_args.kwargs
    assert kwargs["max_retries"] == 2
    timeout = kwargs["timeout"]
    assert timeout.connect == 15.0
    assert timeout.read == 900.0
    assert timeout.write == 60.0
    assert timeout.pool == 15.0


def test_chat_remote_retries_timeout_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    class APITimeoutError(Exception):
        pass

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")
    response = mock.Mock()
    response.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    mock_sdk_client = mock.Mock()
    mock_sdk_client.chat.completions.create.side_effect = [APITimeoutError("gateway timeout"), response]

    with mock.patch.object(client, "_get_remote_client", return_value=mock_sdk_client):
        with mock.patch("ouroboros.llm.time.sleep") as mock_sleep:
            message, usage = client._chat_remote(
                target,
                [{"role": "user", "content": "hi"}],
                None,
                "medium",
                128,
                "auto",
            )

    assert message["content"] == "ok"
    assert usage["provider"] == "openai"
    assert mock_sdk_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


def test_chat_remote_does_not_retry_non_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    client = LLMClient()
    target = client._resolve_remote_target("openai::gpt-5.2")
    mock_sdk_client = mock.Mock()
    mock_sdk_client.chat.completions.create.side_effect = RuntimeError("boom")

    with mock.patch.object(client, "_get_remote_client", return_value=mock_sdk_client):
        with mock.patch("ouroboros.llm.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="boom"):
                client._chat_remote(
                    target,
                    [{"role": "user", "content": "hi"}],
                    None,
                    "medium",
                    128,
                    "auto",
                )

    assert mock_sdk_client.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()


def test_chat_async_retries_timeout_then_succeeds(monkeypatch):
    import asyncio

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    class APITimeoutError(Exception):
        pass

    client = LLMClient()
    response = mock.Mock()
    response.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok-async"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    mock_sdk_client = mock.Mock()
    mock_sdk_client.chat.completions.create = mock.AsyncMock(
        side_effect=[APITimeoutError("gateway timeout"), response]
    )

    with mock.patch.object(client, "_get_async_remote_client", return_value=mock_sdk_client):
        with mock.patch("ouroboros.llm.asyncio.sleep", new=mock.AsyncMock()) as mock_sleep:
            message, usage = asyncio.run(
                client.chat_async(
                    messages=[{"role": "user", "content": "hi"}],
                    model="openai::gpt-5.2",
                    reasoning_effort="medium",
                    max_tokens=128,
                )
            )

    assert message["content"] == "ok-async"
    assert usage["provider"] == "openai"
    assert mock_sdk_client.chat.completions.create.await_count == 2
    mock_sleep.assert_awaited_once_with(1.0)


def test_build_anthropic_messages_rejects_tool_result_without_tool_call_id(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    client = LLMClient()

    with pytest.raises(ValueError, match="tool_call_id"):
        client._build_anthropic_messages([
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "done"},
        ])
