"""``KG_ENTITY_LOOKUP_METHOD=LABEL``: resolving keywords by entity name.

With ``NoopVectorDBStorage`` every KG mode used to fail at
``entities_vdb.query``. LABEL lookup is the explicit opt-in that routes
``local`` / ``global`` / ``hybrid`` / ``mix`` through
``BaseGraphStorage.search_labels`` instead. These tests pin:

- the keyword splitting and the bounded token fallback,
- the shared ``_expand_entity_hits`` tail producing the same shapes as the
  VECTOR path,
- ``_perform_kg_search`` never touching a vector index nor the embedding
  function under LABEL, while VECTOR + a non-persisting storage still fails
  closed with a message that names the knob.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.base import QueryParam
from lightrag.exceptions import StorageCapabilityError
from lightrag.operate import (
    _get_node_data,
    _get_node_data_by_label,
    _perform_kg_search,
    _search_entity_labels,
    _split_keyword_terms,
)

pytestmark = pytest.mark.offline


class _FakeGraph:
    """Minimal graph: nodes by name, undirected edges, substring label search."""

    def __init__(self, nodes: dict[str, dict], edges: dict[tuple, dict]):
        self._nodes = nodes
        self._edges = {tuple(sorted(k)): v for k, v in edges.items()}
        self.search_calls: list[str] = []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        self.search_calls.append(query)
        q = query.lower().strip()
        if not q:
            return []
        hits = [n for n in self._nodes if q in n.lower()]
        hits.sort(key=lambda n: (n.lower() != q, not n.lower().startswith(q), n))
        return hits[:limit]

    async def get_nodes_batch(self, names):
        return {n: dict(self._nodes[n]) for n in names if n in self._nodes}

    async def node_degrees_batch(self, names):
        return {
            n: sum(1 for e in self._edges if n in e) for n in names if n in self._nodes
        }

    async def get_nodes_edges_batch(self, names):
        return {n: [e for e in self._edges if n in e] for n in names}

    async def get_edges_batch(self, pairs):
        return {
            (p["src"], p["tgt"]): dict(self._edges[tuple(sorted((p["src"], p["tgt"])))])
            for p in pairs
            if tuple(sorted((p["src"], p["tgt"]))) in self._edges
        }

    async def edge_degrees_batch(self, pairs):
        return {tuple(p): 2 for p in pairs}


def _graph() -> _FakeGraph:
    return _FakeGraph(
        nodes={
            "Boukrouh, Ikhlass": {"entity_type": "person", "description": "author"},
            "Paper A": {"entity_type": "article", "description": "a paper"},
            "Alice": {"entity_type": "person", "description": "someone"},
            "Alice Smith": {"entity_type": "person", "description": "someone else"},
        },
        edges={
            ("Boukrouh, Ikhlass", "Paper A"): {"weight": 1.0, "description": "wrote"},
        },
    )


# --- _split_keyword_terms ----------------------------------------------------


def test_split_keyword_terms_strips_dedupes_and_keeps_order():
    assert _split_keyword_terms("Ikhlass Boukrouh, articles , ,ARTICLES") == [
        "Ikhlass Boukrouh",
        "articles",
    ]
    assert _split_keyword_terms("") == []
    assert _split_keyword_terms(" , ") == []


# --- _search_entity_labels ---------------------------------------------------


@pytest.mark.asyncio
async def test_full_term_hit_skips_token_fallback():
    graph = _graph()
    hits = await _search_entity_labels("Alice", graph, top_k=10)
    assert hits == ["Alice", "Alice Smith"]
    assert graph.search_calls == ["Alice"]


@pytest.mark.asyncio
async def test_token_fallback_only_on_miss_and_only_long_tokens():
    graph = _graph()
    hits = await _search_entity_labels("Ikhlass de Boukrouh", graph, top_k=10)
    assert hits == ["Boukrouh, Ikhlass"]
    # Full term first, then tokens >= 3 chars; "de" is never searched.
    assert graph.search_calls == ["Ikhlass de Boukrouh", "Ikhlass", "Boukrouh"]


@pytest.mark.asyncio
async def test_round_robin_across_terms_dedupes_and_caps():
    graph = _graph()
    hits = await _search_entity_labels("Alice, Boukrouh, alice", graph, top_k=2)
    # First hit of each term, then capped at top_k; the duplicate term is dropped.
    assert hits == ["Alice", "Boukrouh, Ikhlass"]
    assert graph.search_calls == ["Alice", "Boukrouh"]


@pytest.mark.asyncio
async def test_blank_keywords_or_zero_top_k_return_nothing():
    graph = _graph()
    assert await _search_entity_labels("", graph, top_k=5) == []
    assert await _search_entity_labels("Alice", graph, top_k=0) == []
    assert graph.search_calls == []


# --- _get_node_data_by_label -------------------------------------------------


@pytest.mark.asyncio
async def test_label_lookup_returns_vector_path_shapes():
    graph = _graph()
    entities, relations = await _get_node_data_by_label(
        "Ikhlass Boukrouh", graph, QueryParam(mode="local", top_k=5)
    )
    assert [e["entity_name"] for e in entities] == ["Boukrouh, Ikhlass"]
    assert entities[0]["rank"] == 1
    assert entities[0]["created_at"] is None
    assert entities[0]["entity_type"] == "person"
    assert len(relations) == 1
    assert relations[0]["src_tgt"] == ("Boukrouh, Ikhlass", "Paper A")
    assert relations[0]["weight"] == 1.0
    assert relations[0]["rank"] == 2


@pytest.mark.asyncio
async def test_label_lookup_no_match_is_empty_not_error():
    graph = _graph()
    assert await _get_node_data_by_label(
        "nobody", graph, QueryParam(mode="local", top_k=5)
    ) == ([], [])


# --- _get_node_data regression after the shared-tail refactor ---------------


@pytest.mark.asyncio
async def test_vector_node_data_still_forwards_embedding_and_shapes():
    graph = _graph()
    vdb = AsyncMock()
    vdb.cosine_better_than_threshold = 0.2
    vdb.query = AsyncMock(return_value=[{"entity_name": "Alice", "created_at": 123}])
    embedding = [0.1, 0.2]
    entities, relations = await _get_node_data(
        "alice",
        graph,
        vdb,
        QueryParam(mode="local", top_k=3),
        query_embedding=embedding,
    )
    vdb.query.assert_awaited_once_with("alice", top_k=3, query_embedding=embedding)
    assert entities[0]["entity_name"] == "Alice"
    assert entities[0]["created_at"] == 123
    assert entities[0]["rank"] == 0
    assert relations == []


# --- _perform_kg_search ------------------------------------------------------


def _embedding_mock():
    async def _embed(texts, **_):
        return np.ones((len(texts), 4), dtype=np.float32)

    return AsyncMock(side_effect=_embed)


def _text_chunks(embedding, lookup="LABEL"):
    kv = MagicMock()
    kv.embedding_func = embedding
    kv.global_config = {
        "kg_chunk_pick_method": "VECTOR",
        "kg_entity_lookup_method": lookup,
    }
    return kv


def _vdb(persists=True):
    vdb = AsyncMock()
    vdb.query = AsyncMock(return_value=[])
    vdb.cosine_better_than_threshold = 0.2
    if not persists:
        vdb.persists_vectors = False
    return vdb


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
async def test_label_lookup_never_embeds_or_queries_vector_indexes(mode):
    embedding = _embedding_mock()
    entities_vdb = _vdb(persists=False)
    relationships_vdb = _vdb(persists=False)
    chunks_vdb = _vdb(persists=False)

    result = await _perform_kg_search(
        query="articles de Ikhlass Boukrouh",
        ll_keywords="Ikhlass Boukrouh",
        hl_keywords="articles",
        knowledge_graph_inst=_graph(),
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=_text_chunks(embedding),
        query_param=QueryParam(mode=mode, top_k=5),
        chunks_vdb=chunks_vdb,
    )

    assert embedding.await_count == 0
    entities_vdb.query.assert_not_awaited()
    relationships_vdb.query.assert_not_awaited()
    chunks_vdb.query.assert_not_awaited()
    assert result["chunks_vdb_queryable"] is False
    assert result["vector_chunks"] == []
    names = {e["entity_name"] for e in result["final_entities"]}
    if mode == "global":
        # "articles" matches nothing by name: explicit empty result.
        assert names == set()
    else:
        assert "Boukrouh, Ikhlass" in names
        assert any(
            tuple(r["src_tgt"]) == ("Boukrouh, Ikhlass", "Paper A")
            for r in result["final_relations"]
        )


@pytest.mark.asyncio
async def test_label_lookup_with_persistent_chunks_vdb_still_fetches_vector_chunks():
    embedding = _embedding_mock()
    chunks_vdb = _vdb()
    chunks_vdb.query = AsyncMock(
        return_value=[{"id": "chunk-1", "content": "text", "file_path": "f"}]
    )

    result = await _perform_kg_search(
        query="who",
        ll_keywords="Alice",
        hl_keywords="",
        knowledge_graph_inst=_graph(),
        entities_vdb=_vdb(persists=False),
        relationships_vdb=_vdb(persists=False),
        text_chunks_db=_text_chunks(embedding),
        query_param=QueryParam(mode="mix", top_k=5),
        chunks_vdb=chunks_vdb,
    )

    # Only the query is embedded (for the chunks VDB); keywords never are.
    assert embedding.await_count == 1
    assert embedding.await_args[0][0] == ["who"]
    chunks_vdb.query.assert_awaited_once()
    assert result["chunks_vdb_queryable"] is True
    assert [c["chunk_id"] for c in result["vector_chunks"]] == ["chunk-1"]


@pytest.mark.asyncio
async def test_vector_lookup_fails_closed_and_names_the_knob():
    entities_vdb = _vdb(persists=False)
    with pytest.raises(StorageCapabilityError) as excinfo:
        await _perform_kg_search(
            query="q",
            ll_keywords="Alice",
            hl_keywords="",
            knowledge_graph_inst=_graph(),
            entities_vdb=entities_vdb,
            relationships_vdb=_vdb(persists=False),
            text_chunks_db=_text_chunks(_embedding_mock(), lookup="VECTOR"),
            query_param=QueryParam(mode="local", top_k=5),
        )
    message = str(excinfo.value)
    assert "KG_ENTITY_LOOKUP_METHOD=LABEL" in message
    assert "lightrag-rebuild-vdb" in message
    assert "'local'" in message
    entities_vdb.query.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_lookup_default_path_unchanged_with_persistent_vdb():
    embedding = _embedding_mock()
    entities_vdb = _vdb()
    text_chunks = _text_chunks(embedding, lookup="VECTOR")

    result = await _perform_kg_search(
        query="q",
        ll_keywords="Alice",
        hl_keywords="",
        knowledge_graph_inst=_graph(),
        entities_vdb=entities_vdb,
        relationships_vdb=_vdb(),
        text_chunks_db=text_chunks,
        query_param=QueryParam(mode="local", top_k=5),
    )

    entities_vdb.query.assert_awaited_once()
    assert embedding.await_count == 1
    assert result["chunks_vdb_queryable"] is True


# --- chunk selection without a chunk index -----------------------------------


@pytest.mark.asyncio
async def test_label_workspace_entities_still_deliver_chunks_without_chunk_index():
    """Regression for ``Final context: N entities, M relations, 0 chunks``.

    ``_build_query_context`` hands ``chunks_vdb=None`` to the chunk pickers
    whenever the chunk store does not persist vectors (the LABEL / Noop
    setup). With the default VECTOR pick method the picker must fall back to
    weighted polling instead of selecting nothing.
    """
    from lightrag.constants import GRAPH_FIELD_SEP
    from lightrag.operate import _find_related_text_unit_from_entities

    text_chunks_db = MagicMock()
    text_chunks_db.global_config = {
        "kg_chunk_pick_method": "VECTOR",
        "related_chunk_number": 5,
    }
    text_chunks_db.embedding_func = None
    text_chunks_db.get_by_ids = AsyncMock(
        side_effect=lambda ids: [{"content": f"text of {i}"} for i in ids]
    )
    node_datas = [
        {"entity_name": "Paper A", "source_id": GRAPH_FIELD_SEP.join(["c-1", "c-2"])}
    ]

    chunks = await _find_related_text_unit_from_entities(
        node_datas,
        QueryParam(mode="local"),
        text_chunks_db,
        MagicMock(),
        query="who wrote Paper A",
        chunks_vdb=None,
    )

    assert [c["chunk_id"] for c in chunks] == ["c-1", "c-2"]
