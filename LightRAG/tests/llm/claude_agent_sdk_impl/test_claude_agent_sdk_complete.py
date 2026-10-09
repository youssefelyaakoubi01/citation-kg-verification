"""Offline tests for the Claude Agent SDK binding (`lightrag/llm/claude_agent_sdk.py`)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock  # noqa: E402
from claude_agent_sdk.types import StreamEvent  # noqa: E402

from lightrag.llm import claude_agent_sdk as binding  # noqa: E402
from lightrag.utils import TokenTracker, TruncatedResponse  # noqa: E402

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

MODULE = "lightrag.llm.claude_agent_sdk"


def _result(text="hello", *, is_error=False, usage=None, **extra) -> ResultMessage:
    return ResultMessage(
        subtype="success" if not is_error else "error_during_execution",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id="s",
        result=text,
        usage=usage,
        **extra,
    )


def _assistant(text="hello", stop_reason=None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)], model="sonnet", stop_reason=stop_reason
    )


class FakeQuery:
    """Stand-in for ``claude_agent_sdk.query`` recording prompt and options."""

    def __init__(self, messages):
        self.messages = messages
        self.calls: list[dict] = []

    def __call__(self, *, prompt, options=None, **_):
        self.calls.append({"prompt": prompt, "options": options})

        async def gen():
            for message in self.messages:
                yield message

        return gen()


async def _call(fake, *args, **kwargs):
    with patch(f"{MODULE}.query", fake):
        return await binding.claude_agent_sdk_complete_if_cache.__wrapped__(
            *args, **kwargs
        )


async def test_returns_result_text_and_builds_pure_chat_options(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_AGENT_SDK_CWD", str(tmp_path / "cwd"))
    monkeypatch.delenv("CLAUDE_AGENT_SDK_EFFORT", raising=False)
    fake = FakeQuery([_assistant("hello"), _result("hello")])

    out = await _call(
        fake,
        "sonnet",
        "Say hello",
        system_prompt="Be terse.",
        hashing_kv=object(),
        keyword_extraction=True,
        entity_extraction=True,
        max_tokens=100,
    )

    assert out == "hello"
    assert not isinstance(out, TruncatedResponse)
    call = fake.calls[0]
    assert call["prompt"] == "Say hello"
    options = call["options"]
    assert options.model == "sonnet"
    assert options.system_prompt == "Be terse."
    assert options.tools == []
    assert options.max_turns == 1
    assert options.setting_sources == []
    assert options.strict_mcp_config is True
    assert options.mcp_servers == {}
    assert options.cwd == str(tmp_path / "cwd")
    assert (tmp_path / "cwd").is_dir()
    assert options.include_partial_messages is False
    # Default effort is medium; thinking stays at the CLI's adaptive default
    # because Opus 5.5 rejects ``thinking: disabled``.
    assert options.effort == "medium"
    assert options.thinking is None
    assert options.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    # Subscription login must win over any inherited API key.
    assert options.env["ANTHROPIC_API_KEY"] == ""
    assert options.env["ANTHROPIC_AUTH_TOKEN"] == ""


async def test_api_key_is_forwarded_when_given():
    fake = FakeQuery([_result("ok")])
    await _call(fake, "sonnet", "hi", api_key="sk-ant-test")
    env = fake.calls[0]["options"].env
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert "ANTHROPIC_AUTH_TOKEN" not in env


async def test_effort_env_overrides_default(monkeypatch):
    monkeypatch.setenv("CLAUDE_AGENT_SDK_EFFORT", "High")
    fake = FakeQuery([_result("ok")])
    await _call(fake, "claude-opus-5-5", "hi")
    options = fake.calls[0]["options"]
    assert options.effort == "high"
    assert options.thinking is None


async def test_invalid_effort_falls_back_to_medium_with_warning(monkeypatch):
    monkeypatch.setenv("CLAUDE_AGENT_SDK_EFFORT", "ultra")
    fake = FakeQuery([_result("ok")])
    with patch(f"{MODULE}.logger") as fake_logger:
        await _call(fake, "sonnet", "hi")
    assert fake.calls[0]["options"].effort == "medium"
    assert fake_logger.warning.called


async def test_empty_effort_restores_disabled_thinking(monkeypatch):
    monkeypatch.setenv("CLAUDE_AGENT_SDK_EFFORT", "")
    fake = FakeQuery([_result("ok")])
    await _call(fake, "sonnet", "hi")
    options = fake.calls[0]["options"]
    assert options.effort is None
    assert options.thinking == {"type": "disabled"}


async def test_enable_cot_leaves_thinking_to_cli_default(monkeypatch):
    monkeypatch.setenv("CLAUDE_AGENT_SDK_EFFORT", "")
    fake = FakeQuery([_result("ok")])
    await _call(fake, "sonnet", "hi", enable_cot=True)
    options = fake.calls[0]["options"]
    assert options.thinking is None
    assert options.effort is None


async def test_history_is_folded_into_prompt():
    fake = FakeQuery([_result("ok")])
    history = [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Salut"},
    ]
    await _call(fake, "sonnet", "Et maintenant ?", history_messages=history)
    prompt = fake.calls[0]["prompt"]
    assert prompt.startswith("<conversation_history>")
    assert "<user>\nBonjour\n</user>" in prompt
    assert "<assistant>\nSalut\n</assistant>" in prompt
    assert prompt.endswith("</conversation_history>\n\nEt maintenant ?")


async def test_json_response_format_appends_instruction_and_is_not_forwarded():
    fake = FakeQuery([_result('{"a": 1}')])
    out = await _call(
        fake,
        "sonnet",
        "give json",
        system_prompt="Base.",
        response_format={"type": "json_object"},
    )
    assert out == '{"a": 1}'
    system_prompt = fake.calls[0]["options"].system_prompt
    assert system_prompt.startswith("Base.")
    assert "JSON" in system_prompt


async def test_truncated_response_when_stop_reason_is_max_tokens():
    fake = FakeQuery(
        [_assistant("partial", stop_reason="max_tokens"), _result("partial")]
    )
    out = await _call(fake, "sonnet", "long")
    assert isinstance(out, TruncatedResponse)
    assert out == "partial"


async def test_token_usage_is_recorded_including_cache_tokens():
    tracker = TokenTracker()
    usage = {
        "input_tokens": 2,
        "cache_creation_input_tokens": 500,
        "cache_read_input_tokens": 100,
        "output_tokens": 7,
    }
    fake = FakeQuery([_result("ok", usage=usage)])
    await _call(fake, "sonnet", "hi", token_tracker=tracker)
    assert tracker.get_usage() == {
        "prompt_tokens": 602,
        "completion_tokens": 7,
        "total_tokens": 609,
        "call_count": 1,
    }


async def test_error_result_raises_instead_of_returning_text():
    fake = FakeQuery([_result("boom", is_error=True, errors=["Login expired"])])
    with pytest.raises(binding.ClaudeAgentSDKError, match="Login expired"):
        await _call(fake, "sonnet", "hi")


async def test_rate_limited_result_raises_retryable_error():
    fake = FakeQuery([_result(None, is_error=True, api_error_status=429)])
    with pytest.raises(binding.ClaudeAgentSDKRetryableError):
        await _call(fake, "sonnet", "hi")


async def test_image_inputs_are_rejected_before_any_call():
    fake = FakeQuery([_result("ok")])
    with pytest.raises(ValueError, match="image_inputs"):
        await _call(fake, "sonnet", "hi", image_inputs=[b"png"])
    assert fake.calls == []


async def test_stream_yields_text_deltas_only():
    def event(payload):
        return StreamEvent(uuid="u", session_id="s", event=payload)

    fake = FakeQuery(
        [
            event({"type": "message_start"}),
            event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "Hel"},
                }
            ),
            event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "..."},
                }
            ),
            event(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "lo"},
                }
            ),
            _assistant("Hello"),
            _result("Hello"),
        ]
    )
    # The generator body runs lazily, so the patch must outlive its consumption.
    with patch(f"{MODULE}.query", fake):
        gen = await binding.claude_agent_sdk_complete_if_cache.__wrapped__(
            "sonnet", "hi", stream=True
        )
        chunks = [chunk async for chunk in gen]
    assert chunks == ["Hel", "lo"]
    assert fake.calls[0]["options"].include_partial_messages is True


async def test_stream_error_result_raises():
    fake = FakeQuery([_result(None, is_error=True, errors=["nope"])])
    with patch(f"{MODULE}.query", fake):
        gen = await binding.claude_agent_sdk_complete_if_cache.__wrapped__(
            "sonnet", "hi", stream=True
        )
        with pytest.raises(binding.ClaudeAgentSDKError):
            async for _ in gen:
                pass


async def test_complete_reads_model_from_hashing_kv():
    fake = FakeQuery([_result("ok")])
    hashing_kv = SimpleNamespace(global_config={"llm_model_name": "opus"})
    with patch(f"{MODULE}.query", fake):
        out = await binding.claude_agent_sdk_complete("hi", hashing_kv=hashing_kv)
    assert out == "ok"
    assert fake.calls[0]["options"].model == "opus"


async def test_cli_path_prefers_env_then_system_claude(monkeypatch):
    fake = FakeQuery([_result("ok")])
    monkeypatch.setenv("CLAUDE_AGENT_SDK_CLI_PATH", "/opt/claude/bin/claude")
    await _call(fake, "sonnet", "hi")
    assert fake.calls[0]["options"].cli_path == "/opt/claude/bin/claude"

    monkeypatch.delenv("CLAUDE_AGENT_SDK_CLI_PATH")
    monkeypatch.setattr(binding.shutil, "which", lambda name: "/usr/local/bin/claude")
    await _call(fake, "sonnet", "hi")
    assert fake.calls[1]["options"].cli_path == "/usr/local/bin/claude"

    monkeypatch.setattr(binding.shutil, "which", lambda name: None)
    await _call(fake, "sonnet", "hi")
    assert fake.calls[2]["options"].cli_path is None
