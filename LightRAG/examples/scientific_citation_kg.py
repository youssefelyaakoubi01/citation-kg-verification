"""Build a citation-oriented knowledge graph from scientific articles.

Companion to the ``scientific_citation.yml`` entity extraction profile
(``prompts/samples/scientific_citation.yml``). The profile tells the LLM which
entity types (Paper, Author, Venue, Identifier, Methodology, Dataset, Concept,
Claim) and which relationship keywords (``cites``, ``supports``,
``authored_by``, ...) to use. The core keeps whatever the model emits, so this
script adds the enforcement half: a ``kg_extraction_validator`` that drops
off-schema entities and relations before they reach the graph.

Usage (reads LLM/embedding settings from ``.env``)::

    python examples/scientific_citation_kg.py --input ./papers_txt
    python examples/scientific_citation_kg.py --query "Which claims cite Vaswani et al.?"

The ingest step accepts a directory (or single file) of ``.txt`` / ``.md``
files whose text was already extracted from the articles.

Graph-only ingestion (no embeddings)
------------------------------------
Embeddings are computed at each document's flush for every chunk, entity and
relation it touched, and an entity merged again by a later document is
re-embedded. For a corpus backfill, skip that cost and build the vectors once
at the end:

1. ``python examples/scientific_citation_kg.py --input ./papers_txt --graph-only``
   uses ``NoopVectorDBStorage`` and no embedding function: graph, chunks,
   doc status and LLM cache are written normally, vectors are not.
2. Stop every writer, then run ``lightrag-rebuild-vdb`` ("Rebuild ALL") with
   the same ``WORKING_DIR`` / ``WORKSPACE`` and
   ``LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage`` plus the ``EMBEDDING_*``
   settings from ``.env``.
3. Query as usual (``--query`` / ``--demo-queries``) without ``--graph-only``.

While the vector store is the no-op backend only ``bypass`` queries work, so
this script refuses ``--query`` together with ``--graph-only``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, logger

load_dotenv(dotenv_path=".env", override=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_NAME = "scientific_citation.yml"
PROFILE_SOURCE = REPO_ROOT / "prompts" / "samples" / PROFILE_NAME

# Stored forms of the entity types declared in the profile: the core lower-cases
# the type and removes spaces before the validator sees it ("Paper" -> "paper").
ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "paper",
        "author",
        "organization",
        "venue",
        "identifier",
        "methodology",
        "dataset",
        "concept",
        "claim",
        "other",
    }
)

# Controlled relationship vocabulary. The first comma-separated keyword of a
# relation must be one of these; secondary keywords are free text.
RELATION_KEYWORDS: frozenset[str] = frozenset(
    {
        "cites",
        "supports",
        "contradicts",
        "authored_by",
        "affiliated_with",
        "published_in",
        "has_identifier",
        "proposes",
        "uses_method",
        "uses_dataset",
        "extends",
        "compares_with",
        "addresses",
        "related_to",
    }
)


def _canonical_keyword(keyword: str) -> str:
    return "_".join(keyword.strip().casefold().split())


def split_relation_keywords(keywords: str) -> list[str]:
    """Split a relation ``keywords`` field into canonical tokens.

    ``"Cites, evaluation data"`` becomes ``["cites", "evaluation_data"]``.
    """

    return [
        _canonical_keyword(part)
        for part in keywords.replace("，", ",").split(",")
        if part.strip()
    ]


def make_scientific_validator(
    *,
    strict_types: bool = True,
    strict_relations: bool = True,
    entity_types: frozenset[str] = ENTITY_TYPES,
    relation_keywords: frozenset[str] = RELATION_KEYWORDS,
):
    """Return a ``kg_extraction_validator`` enforcing the citation schema.

    - ``strict_types``: an entity whose every extracted record carries a type
      outside ``entity_types`` is dropped. Records of an allowed type keep the
      entity alive, so a name the model typed both ways survives.
    - ``strict_relations``: a relation is kept only when at least one of its
      keywords is in ``relation_keywords``; the first matching keyword is moved
      to the front so that ``keywords`` always starts with the canonical verb.
    - Relations whose endpoint was dropped are removed too; the merge would
      otherwise materialize the missing endpoint as an ``UNKNOWN`` node.
    """

    def validate(
        chunk_key: str,
        chunk_text: str,
        maybe_nodes: dict[str, list[dict[str, Any]]],
        maybe_edges: dict[tuple[str, str], list[dict[str, Any]]],
    ) -> tuple[
        dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[dict[str, Any]]]
    ]:
        if strict_types:
            for name in list(maybe_nodes):
                records = maybe_nodes[name]
                kept = [
                    record
                    for record in records
                    if str(record.get("entity_type", "")).casefold() in entity_types
                ]
                if not kept:
                    types = sorted({str(r.get("entity_type", "")) for r in records})
                    logger.info(
                        "scientific validator: rejected entity %r (type %s) from %s",
                        name,
                        ", ".join(types),
                        chunk_key,
                    )
                    del maybe_nodes[name]
                elif len(kept) != len(records):
                    maybe_nodes[name] = kept

        for key in list(maybe_edges):
            src, tgt = key
            if src not in maybe_nodes or tgt not in maybe_nodes:
                logger.info(
                    "scientific validator: rejected relation %r -> %r from %s "
                    "(endpoint not extracted as an allowed entity)",
                    src,
                    tgt,
                    chunk_key,
                )
                del maybe_edges[key]
                continue

            if not strict_relations:
                continue

            kept_records: list[dict[str, Any]] = []
            for record in maybe_edges[key]:
                tokens = split_relation_keywords(str(record.get("keywords", "")))
                canonical = next((t for t in tokens if t in relation_keywords), None)
                if canonical is None:
                    logger.info(
                        "scientific validator: rejected relation %r -> %r from %s "
                        "(keywords %r not in vocabulary)",
                        src,
                        tgt,
                        chunk_key,
                        record.get("keywords", ""),
                    )
                    continue
                ordered = [canonical] + [t for t in tokens if t != canonical]
                record["keywords"] = ", ".join(ordered)
                kept_records.append(record)
            if kept_records:
                maybe_edges[key] = kept_records
            else:
                del maybe_edges[key]

        return maybe_nodes, maybe_edges

    return validate


def ensure_profile_installed(prompt_dir: Path | None = None) -> Path:
    """Copy the versioned profile into ``PROMPT_DIR/entity_type`` if missing.

    ``prompts/entity_type/`` is gitignored (it is the operator's folder), so
    the canonical copy lives in ``prompts/samples/``. The loader only reads
    from ``PROMPT_DIR/entity_type``, hence the copy.
    """

    base = prompt_dir or Path(os.getenv("PROMPT_DIR", "").strip() or "./prompts")
    target_dir = base / "entity_type"
    target = target_dir / PROFILE_NAME
    if not target.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROFILE_SOURCE, target)
        logger.info("Installed entity extraction profile at %s", target)
    return target


def _build_embedding_func() -> EmbeddingFunc:
    from lightrag.llm.openai import openai_embed

    return EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
        max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
        func=partial(
            openai_embed.func,  # unwrapped to avoid double EmbeddingFunc wrapping
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            base_url=os.getenv("EMBEDDING_BINDING_HOST") or None,
            api_key=os.getenv("EMBEDDING_BINDING_API_KEY")
            or os.getenv("OPENAI_API_KEY"),
        ),
    )


def rag_kwargs(
    working_dir: str, workspace: str = "", graph_only: bool = False
) -> dict[str, Any]:
    """Build the ``LightRAG`` constructor arguments for this schema.

    ``graph_only`` swaps the vector backend for ``NoopVectorDBStorage`` and
    drops the embedding function, so ingestion never embeds anything. Rebuild
    the vectors afterwards with ``lightrag-rebuild-vdb``.
    """

    kwargs: dict[str, Any] = dict(
        working_dir=working_dir,
        workspace=workspace,
        llm_model_name=os.getenv("LLM_MODEL", "sonnet"),
        llm_model_max_async=int(os.getenv("MAX_ASYNC", "4")),
        addon_params={
            "entity_type_prompt_file": PROFILE_NAME,
            "language": os.getenv("SUMMARY_LANGUAGE", "English"),
        },
        kg_extraction_validator=make_scientific_validator(),
    )
    if graph_only:
        kwargs["vector_storage"] = "NoopVectorDBStorage"
        kwargs["embedding_func"] = None
    else:
        kwargs["vector_storage"] = os.getenv(
            "LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"
        )
        kwargs["embedding_func"] = _build_embedding_func()
    return kwargs


async def build_rag(
    working_dir: str, workspace: str = "", graph_only: bool = False
) -> LightRAG:
    """Create and initialize a LightRAG instance wired for citation graphs."""

    from lightrag.llm.claude_agent_sdk import claude_agent_sdk_complete

    ensure_profile_installed()
    os.makedirs(working_dir, exist_ok=True)

    rag = LightRAG(
        llm_model_func=claude_agent_sdk_complete,
        **rag_kwargs(working_dir, workspace, graph_only),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status(workspace=workspace)
    return rag


def _collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".md"})


async def ingest(rag: LightRAG, input_path: Path) -> int:
    files = _collect_inputs(input_path)
    if not files:
        logger.warning("No .txt/.md files found under %s", input_path)
        return 0
    texts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    names = [
        str(f.relative_to(input_path if input_path.is_dir() else f.parent))
        for f in files
    ]
    await rag.ainsert(texts, file_paths=names)
    return len(files)


DEMO_QUERIES = [
    "Which claims in the citing papers are supported by cited references, and which references are cited for them?",
    "List the papers that use a methodology proposed by another paper in the corpus.",
    "For each cited paper, summarize the citation context in which it is cited.",
]


async def run_queries(rag: LightRAG, queries: list[str]) -> None:
    for question in queries:
        print("\n" + "=" * 80)
        print(question)
        print("=" * 80)
        answer = await rag.aquery(question, param=QueryParam(mode="mix"))
        print(answer)


async def _amain(args: argparse.Namespace) -> None:
    rag = await build_rag(args.working_dir, args.workspace, args.graph_only)
    try:
        if args.input:
            started = time.perf_counter()
            count = await ingest(rag, Path(args.input))
            elapsed = time.perf_counter() - started
            mode = "graph-only" if args.graph_only else "graph + vectors"
            print(
                f"Ingested {count} file(s) from {args.input} in {elapsed:.1f}s ({mode})"
            )
            if args.graph_only:
                print(
                    "Vectors were not built. Run `lightrag-rebuild-vdb` with "
                    "LIGHTRAG_VECTOR_STORAGE=NanoVectorDBStorage before querying."
                )
        if args.query:
            await run_queries(rag, [args.query])
        elif args.demo_queries:
            await run_queries(rag, DEMO_QUERIES)
    finally:
        await rag.finalize_storages()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", help="Directory or file of .txt/.md article texts")
    parser.add_argument("--working-dir", default="./rag_storage_scientific")
    parser.add_argument("--workspace", default="")
    parser.add_argument("--query", help="Run a single query against the graph")
    parser.add_argument(
        "--demo-queries",
        action="store_true",
        help="Run a few thesis-oriented sample queries after ingestion",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Ingest without embeddings (NoopVectorDBStorage); rebuild vectors "
        "later with lightrag-rebuild-vdb",
    )
    args = parser.parse_args()
    if not (args.input or args.query or args.demo_queries):
        parser.error("nothing to do: pass --input, --query or --demo-queries")
    if args.graph_only and (args.query or args.demo_queries):
        parser.error(
            "--graph-only cannot answer queries (no vectors); rebuild them with "
            "lightrag-rebuild-vdb, then query without --graph-only"
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
