"""Offline tests: `claude_agent_sdk` LLM binding needs no API key and rejects VLM."""

from __future__ import annotations

import pytest

from lightrag.api import config as api_config

pytestmark = pytest.mark.offline

_ENV_KEYS = [
    "LLM_BINDING",
    "LLM_MODEL",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "EMBEDDING_BINDING",
    "EMBEDDING_BINDING_API_KEY",
    "VLM_PROCESS_ENABLE",
    "VLM_LLM_BINDING",
    "QUERY_LLM_BINDING",
    "QUERY_LLM_MODEL",
    "QUERY_LLM_BINDING_API_KEY",
    "ANTHROPIC_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(api_config, "_parsed_args_cache", None, raising=False)
    monkeypatch.setattr("sys.argv", ["lightrag-server"])
    yield monkeypatch


def test_claude_agent_sdk_binding_parses_without_api_key(clean_env):
    clean_env.setenv("LLM_BINDING", "claude_agent_sdk")
    clean_env.setenv("LLM_MODEL", "sonnet")
    clean_env.setenv("EMBEDDING_BINDING", "openai")
    clean_env.setenv("EMBEDDING_BINDING_API_KEY", "sk-openai")

    args = api_config.parse_args()

    assert args.llm_binding == "claude_agent_sdk"
    assert args.llm_model == "sonnet"
    assert args.llm_binding_api_key is None
    assert api_config.get_default_host("claude_agent_sdk") == ""


def test_claude_agent_sdk_role_binding_needs_no_api_key(clean_env):
    clean_env.setenv("LLM_BINDING", "openai")
    clean_env.setenv("LLM_BINDING_API_KEY", "sk-openai")
    clean_env.setenv("QUERY_LLM_BINDING", "claude_agent_sdk")
    clean_env.setenv("QUERY_LLM_MODEL", "opus")

    args = api_config.parse_args()

    assert args.query_llm_binding == "claude_agent_sdk"
    assert args.query_llm_model == "opus"


def test_claude_agent_sdk_is_rejected_for_vlm_role(clean_env):
    clean_env.setenv("LLM_BINDING", "claude_agent_sdk")
    clean_env.setenv("VLM_PROCESS_ENABLE", "true")

    with pytest.raises(SystemExit, match="does not support image inputs"):
        api_config.parse_args()


def test_claude_agent_sdk_ignores_template_placeholder_api_key(clean_env):
    clean_env.setenv("LLM_BINDING", "claude_agent_sdk")
    clean_env.setenv("LLM_BINDING_API_KEY", "your_api_key")

    args = api_config.parse_args()

    assert args.llm_binding_api_key is None


def test_claude_agent_sdk_keeps_real_api_key(clean_env):
    clean_env.setenv("LLM_BINDING", "claude_agent_sdk")
    clean_env.setenv("LLM_BINDING_API_KEY", "sk-ant-real")

    args = api_config.parse_args()

    assert args.llm_binding_api_key == "sk-ant-real"
