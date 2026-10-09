"""Tests for the scientific-citation entity extraction profile and validator.

The profile lives at ``prompts/samples/scientific_citation.yml`` and the
enforcement half (``kg_extraction_validator``) in
``examples/scientific_citation_kg.py``. Both are repo artifacts the core never
imports, so these tests pin that they stay loadable through the real profile
loader and that the validator honours the schema the profile declares.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "prompts" / "samples" / "scientific_citation.yml"
EXAMPLE_PATH = REPO_ROOT / "examples" / "scientific_citation_kg.py"

pytestmark = pytest.mark.offline


def _require_yaml():
    pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def example_module():
    spec = importlib.util.spec_from_file_location(
        "scientific_citation_kg", EXAMPLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile():
    _require_yaml()
    from lightrag.prompt import load_entity_extraction_prompt_profile

    return load_entity_extraction_prompt_profile(PROFILE_PATH)


def test_profile_declares_every_type_and_keyword(profile, example_module):
    guidance = profile["entity_types_guidance"]
    for stored_type in example_module.ENTITY_TYPES:
        # Guidance uses the display form ("Paper"); the validator the stored
        # form ("paper"). Both must agree on the vocabulary.
        assert stored_type in guidance.casefold(), stored_type
    for keyword in example_module.RELATION_KEYWORDS:
        assert f"- {keyword}:" in guidance, keyword


def test_text_examples_survive_format(profile):
    from lightrag.prompt import PROMPTS

    rendered = "\n".join(profile["entity_extraction_examples"]).format(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types_guidance="guidance",
        language="English",
    )
    assert PROMPTS["DEFAULT_COMPLETION_DELIMITER"] in rendered
    assert "cites" in rendered and "supports" in rendered
    rows = [
        line
        for line in rendered.splitlines()
        if line.startswith(("entity", "relation"))
    ]
    for row in rows:
        fields = row.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
        expected = 4 if fields[0] == "entity" else 5
        assert len(fields) == expected, row


def test_json_examples_are_valid_json(profile):
    blob = "\n".join(profile["entity_extraction_json_examples"])
    payload = json.loads(blob[blob.index("{") :])
    assert {e["type"] for e in payload["entities"]} >= {"Paper", "Claim", "Author"}
    for rel in payload["relationships"]:
        assert set(rel) == {"source", "target", "keywords", "description"}
    keywords = {rel["keywords"].split(",")[0] for rel in payload["relationships"]}
    assert {"cites", "supports"} <= keywords


@pytest.mark.parametrize("use_json", [True, False])
def test_profile_resolves_through_loader_in_both_modes(monkeypatch, tmp_path, use_json):
    _require_yaml()
    from lightrag.prompt import resolve_entity_extraction_prompt_profile

    entity_type_dir = tmp_path / "entity_type"
    entity_type_dir.mkdir()
    (entity_type_dir / "scientific_citation.yml").write_bytes(PROFILE_PATH.read_bytes())
    monkeypatch.setenv("PROMPT_DIR", str(tmp_path))

    resolved = resolve_entity_extraction_prompt_profile(
        {"entity_type_prompt_file": "scientific_citation.yml"}, use_json=use_json
    )
    assert "- Claim:" in resolved["entity_types_guidance"]
    key = (
        "entity_extraction_json_examples" if use_json else "entity_extraction_examples"
    )
    assert resolved[key] and "Attention Is All You Need" in resolved[key][0]


def _node(entity_type: str) -> dict:
    return {"entity_type": entity_type, "description": "d", "source_id": "c1"}


def _edge(keywords: str) -> dict:
    return {"keywords": keywords, "description": "d", "weight": 1.0, "source_id": "c1"}


def test_validator_drops_off_schema_types_and_relations(example_module):
    validate = example_module.make_scientific_validator()
    nodes = {
        "Paper A": [_node("paper")],
        "Alice": [_node("author")],
        "Bob": [_node("person")],  # not in schema
        "Mixed": [_node("location"), _node("concept")],  # one allowed record
    }
    edges = {
        ("Paper A", "Alice"): [_edge("Authored_By")],
        ("Paper A", "Bob"): [_edge("authored_by")],  # endpoint dropped
        ("Paper A", "Mixed"): [_edge("mentions")],  # keyword off-vocabulary
        ("Alice", "Mixed"): [_edge("research topic, addresses")],
    }

    out_nodes, out_edges = validate("c1", "text", nodes, edges)

    assert set(out_nodes) == {"Paper A", "Alice", "Mixed"}
    assert [r["entity_type"] for r in out_nodes["Mixed"]] == ["concept"]
    assert set(out_edges) == {("Paper A", "Alice"), ("Alice", "Mixed")}
    assert out_edges[("Paper A", "Alice")][0]["keywords"] == "authored_by"
    # Canonical keyword is moved to the front, secondary keyword kept.
    assert out_edges[("Alice", "Mixed")][0]["keywords"] == "addresses, research_topic"


def test_validator_non_strict_keeps_everything_but_orphans(example_module):
    validate = example_module.make_scientific_validator(
        strict_types=False, strict_relations=False
    )
    nodes = {"X": [_node("person")], "Y": [_node("paper")]}
    edges = {("X", "Y"): [_edge("mentions")], ("Y", "Z"): [_edge("cites")]}

    out_nodes, out_edges = validate("c1", "text", nodes, edges)

    assert set(out_nodes) == {"X", "Y"}
    assert set(out_edges) == {("X", "Y")}
    assert out_edges[("X", "Y")][0]["keywords"] == "mentions"


def test_ensure_profile_installed_copies_sample(example_module, tmp_path):
    target = example_module.ensure_profile_installed(prompt_dir=tmp_path)
    assert target == tmp_path / "entity_type" / "scientific_citation.yml"
    assert target.read_bytes() == PROFILE_PATH.read_bytes()
    # Second call is a no-op on an existing file.
    target.write_text("custom")
    example_module.ensure_profile_installed(prompt_dir=tmp_path)
    assert target.read_text() == "custom"


def test_rag_kwargs_graph_only_disables_vectors(example_module, monkeypatch):
    monkeypatch.delenv("LIGHTRAG_VECTOR_STORAGE", raising=False)
    kwargs = example_module.rag_kwargs("/tmp/wd", "ws", graph_only=True)
    assert kwargs["vector_storage"] == "NoopVectorDBStorage"
    assert kwargs["embedding_func"] is None
    assert (
        kwargs["addon_params"]["entity_type_prompt_file"] == "scientific_citation.yml"
    )
    assert callable(kwargs["kg_extraction_validator"])


def test_rag_kwargs_default_keeps_vectors(example_module, monkeypatch):
    from lightrag.utils import EmbeddingFunc

    monkeypatch.delenv("LIGHTRAG_VECTOR_STORAGE", raising=False)
    monkeypatch.setenv("EMBEDDING_DIM", "8")
    kwargs = example_module.rag_kwargs("/tmp/wd", "ws", graph_only=False)
    assert kwargs["vector_storage"] == "NanoVectorDBStorage"
    assert isinstance(kwargs["embedding_func"], EmbeddingFunc)
    assert kwargs["embedding_func"].embedding_dim == 8
