"""Claude Agent SDK binding: Claude through a Claude Code login.

Rules for callers:

- Authentication comes from the ``claude`` CLI session (``claude /login``) or
  from ``CLAUDE_CODE_OAUTH_TOKEN``; no API key is required. Leave
  ``LLM_BINDING_API_KEY`` empty for a subscription login. When ``api_key`` IS
  given it is forwarded as ``ANTHROPIC_API_KEY`` and billed to that key.
- Every call spawns one ``claude`` subprocess, so keep ``MAX_ASYNC`` small
  (4 is a good start) and expect roughly 1 GiB RAM per concurrent call.
- Image inputs are not supported: do not use this binding for the VLM role.
- Reasoning depth is set with ``CLAUDE_AGENT_SDK_EFFORT`` (``low`` | ``medium`` |
  ``high`` | ``xhigh`` | ``max``, default ``medium``). With an effort level the
  CLI keeps adaptive thinking on; Opus 5.5 / Sonnet 5.5 reject
  ``thinking: disabled``. Set the variable to an empty string to send no
  effort and fall back to ``thinking: disabled`` (older models only).
- Prior conversation turns are folded into the prompt because the SDK cannot
  replay assistant turns; this costs input tokens on long chats.
- The Claude Code subscription login is meant for personal / internal use.
  Anthropic's terms reserve third-party products for API keys.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pipmaster as pm  # Pipmaster for dynamic library install

# Install the Claude Agent SDK if not present
if not pm.is_installed("claude-agent-sdk"):
    pm.install("claude-agent-sdk")

from claude_agent_sdk import (  # noqa: E402
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    ProcessError,
    ResultMessage,
    TextBlock,
    query,
)
from claude_agent_sdk.types import StreamEvent  # noqa: E402
from tenacity import (  # noqa: E402
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import TruncatedResponse, logger  # noqa: E402

__all__ = [
    "ClaudeAgentSDKError",
    "claude_agent_sdk_complete_if_cache",
    "claude_agent_sdk_complete",
]

_JSON_ONLY_INSTRUCTION = (
    "Respond with a single valid JSON object only. "
    "Do not wrap it in markdown fences and do not add any text before or after it."
)

_DEFAULT_MODEL = "sonnet"

_SCRATCH_CWD_ENV = "CLAUDE_AGENT_SDK_CWD"
_CLI_PATH_ENV = "CLAUDE_AGENT_SDK_CLI_PATH"
_EFFORT_ENV = "CLAUDE_AGENT_SDK_EFFORT"
_DEFAULT_EFFORT = "medium"
_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})


class ClaudeAgentSDKError(RuntimeError):
    """The CLI reported an error result (``ResultMessage.is_error``)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ClaudeAgentSDKRetryableError(ClaudeAgentSDKError):
    """Transient CLI error (rate limit / overloaded / server error)."""


def _scratch_cwd() -> str:
    """Return an empty working directory so no CLAUDE.md / .claude is picked up."""
    path = Path(
        os.getenv(_SCRATCH_CWD_ENV)
        or Path.home() / ".lightrag" / "claude_agent_sdk_cwd"
    )
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _cli_path() -> str | None:
    """Prefer the system ``claude`` over the SDK's bundled binary.

    The wheel bundles a fixed CLI version that lags behind ``claude update``;
    a newer model can be refused by the bundled binary ("version X or newer is
    required") while the installed CLI, which also holds the login, accepts it.
    ``CLAUDE_AGENT_SDK_CLI_PATH`` forces a path; ``None`` falls back to bundled.
    """
    forced = os.getenv(_CLI_PATH_ENV)
    if forced:
        return forced
    return shutil.which("claude")


def _effort() -> str | None:
    """Resolve ``CLAUDE_AGENT_SDK_EFFORT``; ``None`` when explicitly emptied."""
    raw = os.getenv(_EFFORT_ENV)
    if raw is None:
        return _DEFAULT_EFFORT
    level = raw.strip().lower()
    if not level:
        return None
    if level not in _EFFORT_LEVELS:
        logger.warning(
            "%s=%r is not one of %s; using %s",
            _EFFORT_ENV,
            raw,
            "/".join(sorted(_EFFORT_LEVELS)),
            _DEFAULT_EFFORT,
        )
        return _DEFAULT_EFFORT
    return level


def _build_prompt(prompt: str, history_messages: list[dict[str, Any]] | None) -> str:
    """Fold prior turns into the prompt; the SDK only accepts the current user turn."""
    if not history_messages:
        return prompt
    turns = []
    for message in history_messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        turns.append(f"<{role}>\n{content}\n</{role}>")
    transcript = "\n\n".join(turns)
    return f"<conversation_history>\n{transcript}\n</conversation_history>\n\n{prompt}"


def _build_options(
    *,
    model: str,
    system_prompt: str | None,
    api_key: str | None,
    stream: bool,
    enable_cot: bool,
) -> ClaudeAgentOptions:
    env: dict[str, str] = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    if api_key:
        env["ANTHROPIC_API_KEY"] = api_key
    else:
        # The CLI prefers an API key over the login session; an empty value
        # makes the inherited key (e.g. from a loaded .env) inert.
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_AUTH_TOKEN"] = ""
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt or "You are a helpful assistant.",
        tools=[],  # pure chat completion: no built-in tools
        # Without strict_mcp_config the CLI loads every MCP server from the
        # user's ~/.claude.json and ships their tool schemas on every call
        # (measured: ~40k input tokens instead of ~600).
        mcp_servers={},
        strict_mcp_config=True,
        max_turns=1,
        setting_sources=[],  # do not load CLAUDE.md / settings from any directory
        cwd=_scratch_cwd(),
        cli_path=_cli_path(),
        env=env,
        include_partial_messages=stream,
    )
    effort = _effort()
    if effort is not None:
        # Effort governs thinking depth; leave thinking at the CLI's adaptive
        # default because Opus 5.5 / Sonnet 5.5 reject ``disabled``.
        options.effort = effort
    elif not enable_cot:
        options.thinking = {"type": "disabled"}
    return options


def _record_usage(token_tracker: Any | None, usage: dict[str, Any] | None) -> None:
    if token_tracker is None or not usage:
        return
    prompt_tokens = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    )
    completion_tokens = usage.get("output_tokens") or 0
    token_tracker.add_usage(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    )


def _raise_for_result(result: ResultMessage) -> None:
    if not result.is_error:
        return
    detail = "; ".join(result.errors or []) or result.result or result.subtype
    message = f"Claude Agent SDK error ({result.subtype}): {detail}"
    if result.api_error_status in (429, 500, 502, 503, 529):
        raise ClaudeAgentSDKRetryableError(message, result.api_error_status)
    raise ClaudeAgentSDKError(message, result.api_error_status)


async def _collect(
    prompt: str, options: ClaudeAgentOptions, token_tracker: Any | None
) -> str:
    parts: list[str] = []
    final: str | None = None
    truncated = False
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            parts.extend(
                block.text for block in message.content if isinstance(block, TextBlock)
            )
            if message.stop_reason == "max_tokens":
                truncated = True
        elif isinstance(message, ResultMessage):
            _raise_for_result(message)
            _record_usage(token_tracker, message.usage)
            final = message.result
    content = final if final is not None else "".join(parts)
    if truncated:
        logger.warning("Claude Agent SDK response truncated (stop_reason=max_tokens)")
        return TruncatedResponse(content)
    return content


async def _stream(
    prompt: str, options: ClaudeAgentOptions, token_tracker: Any | None
) -> AsyncIterator[str]:
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                event = message.event
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield delta["text"]
            elif isinstance(message, ResultMessage):
                _raise_for_result(message)
                _record_usage(token_tracker, message.usage)
    except Exception as e:
        logger.error(f"Error in Claude Agent SDK stream: {e}")
        raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (ClaudeAgentSDKRetryableError, CLIConnectionError, ProcessError)
    ),
)
async def claude_agent_sdk_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    image_inputs: list[Any] | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """Complete ``prompt`` with Claude through the Claude Agent SDK.

    ``base_url`` is accepted for signature parity and ignored. ``image_inputs``
    raises because the SDK prompt channel is text only. Returns an async
    iterator of text deltas when ``stream=True`` and a ``str`` otherwise
    (``TruncatedResponse`` when the model hit its output limit).
    """
    if image_inputs:
        raise ValueError(
            "claude_agent_sdk binding does not support image_inputs; "
            "use another binding for the VLM role"
        )
    if base_url:
        logger.debug("claude_agent_sdk ignores base_url=%s", base_url)

    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("entity_extraction", None)
    kwargs.pop("max_tokens", None)
    timeout = kwargs.pop("timeout", None)
    stream = bool(kwargs.pop("stream", False))
    response_format = kwargs.pop("response_format", None)

    effective_system_prompt = system_prompt
    if (
        isinstance(response_format, dict)
        and response_format.get("type") == "json_object"
    ):
        effective_system_prompt = (
            f"{system_prompt}\n\n{_JSON_ONLY_INSTRUCTION}"
            if system_prompt
            else _JSON_ONLY_INSTRUCTION
        )

    full_prompt = _build_prompt(prompt, history_messages)
    options = _build_options(
        model=model or _DEFAULT_MODEL,
        system_prompt=effective_system_prompt,
        api_key=api_key or None,
        stream=stream,
        enable_cot=enable_cot,
    )

    if stream:
        return _stream(full_prompt, options, token_tracker)

    coro = _collect(full_prompt, options, token_tracker)
    if timeout:
        return await asyncio.wait_for(coro, timeout=float(timeout))
    return await coro


async def claude_agent_sdk_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """Model-from-config variant: reads ``llm_model_name`` from ``hashing_kv``."""
    hashing_kv = kwargs.get("hashing_kv")
    model_name = _DEFAULT_MODEL
    if hashing_kv is not None:
        model_name = hashing_kv.global_config.get("llm_model_name") or _DEFAULT_MODEL
    return await claude_agent_sdk_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )
