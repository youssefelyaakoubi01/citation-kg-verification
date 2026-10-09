"""Corpus, ingestion-run and knowledge-graph statistics for the article.

Reads ``rag_storage/`` (doc status, text chunks, pre-merge indexes, LLM cache,
GraphML) and ``lightrag.log`` read-only; writes ``data/graph_stats.json``,
``data/corpus.csv``, ``data/numbers_stats.tex`` and ``tables/tab_*.tex``.
Run with the repository venv: ``LightRAG/.venv/bin/python analysis/graph_stats.py``.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

from common import (
    ANALYSIS, DATA, ENV, HUB, LOG, REPO, SEP, TABLES, VOCAB, DOI_RE,
    cited_endpoint, fmt_float, fmt_int, fmt_pct, has_citing_sentence,
    is_placeholder, latex_escape, load_graph, load_kv, node_file_paths,
    read_env, split_keywords, write_macros, write_table,
)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_cutoff() -> dt.datetime | None:
    """Creation time after which cache entries are not server responses.

    A frozen copy of the storage carries a SNAPSHOT.json marker with the time
    of the copy; entries created later were added by the in-process pilot.
    The live server directory has no marker and every entry counts.
    """
    marker = STORAGE / "SNAPSHOT.json"
    if not marker.is_file():
        return None
    return dt.datetime.fromisoformat(json.loads(marker.read_text())["copied_at"])


def local(ts: int | float | None) -> str:
    if not ts:
        return ""
    return dt.datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M:%S")


def parse_log(processed_files: set[str]) -> dict:
    ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - (\S+) - (\w+) - (.*)$")
    chunk_re = re.compile(r"Chunk (\d+) of (\d+) extracted (\d+) Ent \+ (\d+) Rel (doc-\S+)")
    done_re = re.compile(r"Completed processing file (\d+)/(\d+): (.*)$")
    final_re = re.compile(r"Final context: (\d+) entities, (\d+) relations, (\d+) chunks")
    restart_re = re.compile(r"Application startup complete|Started server process")
    activity = []
    chunk_by_id: dict[str, tuple[int, int]] = {}
    chunk_lines = 0
    session_limit_lines = 0
    queries = []
    for line in LOG.read_text(errors="replace").splitlines():
        if "session limit" in line.lower():
            session_limit_lines += 1
        m = ts_re.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        msg = m.group(4)
        if restart_re.search(msg):
            activity.append((t, "restart", ""))
            continue
        cm = chunk_re.search(msg)
        if cm:
            chunk_lines += 1
            chunk_by_id[cm.group(5)] = (int(cm.group(3)), int(cm.group(4)))
            activity.append((t, "chunk", cm.group(5)))
            continue
        dm = done_re.search(msg)
        if dm:
            activity.append((t, "done", dm.group(3)))
            continue
        fm = final_re.search(msg)
        if fm:
            queries.append(dict(time=m.group(1), entities=int(fm.group(1)),
                                relations=int(fm.group(2)), chunks=int(fm.group(3))))
    sessions: list[dict] = []
    force_new = True
    for t, kind, payload in activity:
        if kind == "restart":
            force_new = True
            continue
        if sessions and not force_new and (t - sessions[-1]["end"]).total_seconds() <= 600:
            s = sessions[-1]
        else:
            s = dict(start=t, end=t, chunks=0, docs=[])
            sessions.append(s)
            force_new = False
        s["end"] = t
        if kind == "chunk":
            s["chunks"] += 1
        else:
            s["docs"].append(payload)
    for s in sessions:
        s["duration_s"] = (s["end"] - s["start"]).total_seconds()
        s["corpus_docs"] = [d for d in s["docs"] if d in processed_files]
        if s["corpus_docs"]:
            s["kind"] = "ingestion"
        elif s["chunks"]:
            s["kind"] = "aborted"
        else:
            s["kind"] = "other"
    return dict(sessions=sessions, chunk_by_id=chunk_by_id, chunk_lines=chunk_lines,
                session_limit_lines=session_limit_lines, queries=queries)


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}~h {m:02d}~min"
    if m:
        return f"{m}~min {s:02d}~s"
    return f"{s}~s"


def main() -> None:
    S: dict = {}
    M: dict[str, str] = {}

    # ------------------------------------------------------------ corpus --
    ds = load_kv("doc_status")
    tc = load_kv("text_chunks")
    titles = {r["file_path"]: r for r in csv.DictReader((ANALYSIS / "corpus_titles.csv").open())}
    tokens_by_doc: dict[str, int] = defaultdict(int)
    chunk_tokens = [int(c["tokens"]) for c in tc.values()]
    for c in tc.values():
        tokens_by_doc[c["full_doc_id"]] += int(c["tokens"])
    docs = []
    for did, d in ds.items():
        meta = d.get("metadata") or {}
        start, end = meta.get("process_start_time"), meta.get("process_end_time")
        t = titles.get(d["file_path"], {})
        docs.append(dict(
            doc_id=did, file=d["file_path"], status=d["status"],
            chunks=int(d.get("chunks_count") or 0), chars=int(d.get("content_length") or 0),
            tokens=tokens_by_doc.get(did, 0), wall_s=(end - start) if start and end else None,
            parse_engine=meta.get("parse_engine"), chunk_method=meta.get("chunk_method"),
            chunk_opts=meta.get("chunk_opts"), error=str(d.get("error_msg") or ""),
            title=t.get("title", ""), venue=t.get("venue", ""), year=t.get("year", ""),
            created_at=d.get("created_at"), updated_at=d.get("updated_at"),
        ))
    processed = sorted([d for d in docs if d["status"] == "processed"], key=lambda d: d["title"].lower())
    failed = [d for d in docs if d["status"] != "processed"]
    for i, d in enumerate(processed, 1):
        d["id"] = f"D{i:02d}"
    inputs = REPO / "inputs"
    inventory_path = DATA / "inputs_inventory.json"
    pdf_files = sorted(inputs.rglob("*.pdf")) if inputs.is_dir() else []
    if pdf_files:
        inventory = [dict(path=str(f.relative_to(inputs)), bytes=f.stat().st_size, md5=md5(f)) for f in pdf_files]
        inventory_path.write_text(json.dumps(inventory, indent=1))
    else:  # published copy: the corpus PDFs are not redistributed, only their inventory
        inventory = json.loads(inventory_path.read_text())
    pdf_distinct = len({x["md5"] for x in inventory})
    failed_font = sum(1 for d in failed if "DescendantFonts" in d["error"])
    failed_dup = sum(1 for d in failed if "Identical content" in d["error"])
    S["corpus"] = dict(
        pdf_files=len(inventory), pdf_distinct=pdf_distinct, queued=len(docs),
        processed=len(processed), failed=len(failed), failed_font=failed_font,
        failed_duplicate=failed_dup, chunks=len(tc), tokens=sum(chunk_tokens),
        tokens_mean=statistics.mean(chunk_tokens), tokens_min=min(chunk_tokens),
        tokens_max=max(chunk_tokens), chars=sum(d["chars"] for d in processed),
        chunks_per_doc_mean=len(tc) / len(processed),
        short_docs=sum(1 for d in processed if d["chunks"] <= 2),
        parse_engines=Counter(d["parse_engine"] for d in processed),
        chunk_methods=Counter(d["chunk_method"] for d in processed),
        chunk_opts=Counter(d["chunk_opts"] for d in processed),
        queued_first=min(d["created_at"] for d in docs), queued_last=max(d["created_at"] for d in docs),
        finished_last=max(d["updated_at"] for d in processed),
        documents=processed, failed_documents=failed,
    )
    with (DATA / "corpus.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "file", "title", "venue", "year", "status", "chunks", "tokens", "chars", "wall_s", "error"])
        w.writeheader()
        for d in processed + failed:
            w.writerow({k: d.get(k, "") for k in w.fieldnames})

    wall = [d["wall_s"] for d in processed if d["wall_s"] is not None]
    S["per_doc_wall"] = dict(sum_s=sum(wall), mean_s=statistics.mean(wall), median_s=statistics.median(wall), min_s=min(wall), max_s=max(wall))

    # ------------------------------------------------------- pre-merge --
    fe = load_kv("full_entities")
    fr = load_kv("full_relations")
    S["pre_merge"] = dict(
        docs_with_entities=len(fe), entity_names=sum(len(v["entity_names"]) for v in fe.values()),
        docs_with_relations=len(fr), relation_pairs=sum(len(v["relation_pairs"]) for v in fr.values()),
        entity_chunks_rows=len(load_kv("entity_chunks")), relation_chunks_rows=len(load_kv("relation_chunks")),
        full_docs=len(load_kv("full_docs")),
    )

    # ------------------------------------------------------------ cache --
    cache = load_kv("llm_response_cache")
    cutoff = cache_cutoff()
    if cutoff is not None:
        cache = {k: r for k, r in cache.items() if dt.datetime.fromtimestamp(int(r["create_time"])) <= cutoff}
    by_type = Counter(r.get("cache_type") for r in cache.values())
    extract_chunks = {r.get("chunk_id") for r in cache.values() if r.get("cache_type") == "extract"}
    S["cache"] = dict(total=len(cache), by_type=dict(by_type), extract_chunks=len(extract_chunks),
                      create_first=local(min(int(r["create_time"]) for r in cache.values())),
                      create_last=local(max(int(r["create_time"]) for r in cache.values())))

    # -------------------------------------------------------------- log --
    log = parse_log({d["file"] for d in processed})
    ingestion = [s for s in log["sessions"] if s["kind"] == "ingestion"]
    aborted = [s for s in log["sessions"] if s["kind"] == "aborted"]
    first_ingest = min(s["start"] for s in ingestion)
    extract_created = [int(r["create_time"]) for r in cache.values() if r.get("cache_type") == "extract"]
    reused = sum(1 for t in extract_created if dt.datetime.fromtimestamp(t) < first_ingest)
    ent_sum = sum(e for e, _ in log["chunk_by_id"].values())
    rel_sum = sum(r for _, r in log["chunk_by_id"].values())
    S["log"] = dict(
        sessions=[dict(start=s["start"].strftime("%Y-%m-%d %H:%M"), end=s["end"].strftime("%H:%M"),
                       duration_s=s["duration_s"], chunks=s["chunks"], docs=len(s["corpus_docs"]), kind=s["kind"])
                  for s in log["sessions"] if s["kind"] != "other"],
        ingestion_sessions=len(ingestion), ingestion_wall_s=sum(s["duration_s"] for s in ingestion),
        ingestion_first=first_ingest.strftime("%Y-%m-%d %H:%M"),
        ingestion_last=max(s["end"] for s in ingestion).strftime("%Y-%m-%d %H:%M"),
        aborted_sessions=len(aborted), aborted_wall_s=sum(s["duration_s"] for s in aborted),
        aborted_chunk_lines=sum(s["chunks"] for s in aborted),
        chunk_lines=log["chunk_lines"], chunks_logged=len(log["chunk_by_id"]),
        entities_logged=ent_sum, relations_logged=rel_sum,
        entities_per_chunk=ent_sum / len(log["chunk_by_id"]), relations_per_chunk=rel_sum / len(log["chunk_by_id"]),
        session_limit_lines=log["session_limit_lines"], extract_cache_reused_from_aborted=reused,
        queries=log["queries"],
    )

    # ------------------------------------------------------------ graph --
    g = load_graph()
    types = {n: str(d.get("entity_type", "")) for n, d in g.nodes(data=True)}
    type_counts = Counter(types.values())
    degrees = dict(g.degree())
    comps = sorted((len(c) for c in __import__("networkx").connected_components(g)), reverse=True)
    kw_contain = Counter()
    kw_first = Counter()
    edges_with_vocab = 0
    cites, supports, contradicts = [], [], []
    for u, v, d in g.edges(data=True):
        toks = split_keywords(d.get("keywords", ""))
        tokset = set(toks)
        hit = False
        for k in VOCAB:
            if k in tokset:
                kw_contain[k] += 1
                hit = True
        edges_with_vocab += hit
        if toks and toks[0] in VOCAB:
            kw_first[toks[0]] += 1
        if "cites" in tokset:
            cites.append((u, v, d))
        if "supports" in tokset:
            supports.append((u, v, d))
        if "contradicts" in tokset:
            contradicts.append((u, v, d))
    placeholder_nodes = [n for n in g.nodes if is_placeholder(n)]
    cites_pp = sum(1 for u, v, _ in cites if types[u] == "paper" and types[v] == "paper")
    papers_in_cites = {x for u, v, _ in cites for x in (u, v) if types[x] == "paper"}
    cites_hub = sum(1 for u, v, _ in cites if HUB in (u, v))
    cites_placeholder = sum(1 for u, v, _ in cites if is_placeholder(u) or is_placeholder(v))
    cites_sentence = sum(1 for _, _, d in cites if has_citing_sentence(d.get("description", "")))
    cites_sentence_strict = sum(1 for _, _, d in cites if has_citing_sentence(d.get("description", ""), strict=True))
    cited_nodes = {cited_endpoint(g, u, v) for u, v, _ in cites}
    cited_resolved = {n for n in cited_nodes if not is_placeholder(n) and n != HUB and types[n] == "paper"}
    claims = {n for n, t in types.items() if t == "claim"}
    sup_pc = [(u, v) for u, v, _ in supports if {types[u], types[v]} == {"paper", "claim"}]
    claims_supported = {x for u, v in sup_pc for x in (u, v) if types[x] == "claim"}
    papers_supporting = {x for u, v in sup_pc for x in (u, v) if types[x] == "paper"}
    cited_and_supporting = cited_resolved & papers_supporting
    verifiable_edges = [(u, v, d) for u, v, d in cites
                        if has_citing_sentence(d.get("description", ""), strict=True)
                        and cited_endpoint(g, u, v) in cited_and_supporting]
    identifiers = [n for n, t in types.items() if t == "identifier"]
    doi_like = sum(1 for n in identifiers if DOI_RE.match(n.strip()))
    hub_deg = degrees.get(HUB, 0)
    hub_files = len(node_file_paths(g.nodes[HUB])) if HUB in g else 0
    top = sorted(degrees.items(), key=lambda kv: -kv[1])[:10]
    S["graph"] = dict(
        nodes=g.number_of_nodes(), edges=g.number_of_edges(), density=__import__("networkx").density(g),
        degree_mean=statistics.mean(degrees.values()), degree_median=statistics.median(degrees.values()),
        degree_max=max(degrees.values()), components=len(comps), giant=comps[0], isolated=sum(1 for d in degrees.values() if d == 0),
        type_counts=dict(type_counts.most_common()), keyword_containment=dict(kw_contain.most_common()),
        keyword_first=dict(kw_first.most_common()), edges_with_vocab=edges_with_vocab,
        edges_multi_vocab=sum(kw_contain.values()) - edges_with_vocab,
        placeholder_nodes=len(placeholder_nodes), hub_degree=hub_deg, hub_files=hub_files,
        top_degree=[(n, degrees[n], types[n]) for n, _ in top],
        degree_hist=dict(Counter(degrees.values())),
        citation=dict(
            cites=len(cites), cites_paper_paper=cites_pp, papers_in_cites=len(papers_in_cites),
            cites_hub=cites_hub, cites_placeholder=cites_placeholder, cites_sentence=cites_sentence, cites_sentence_strict=cites_sentence_strict,
            cited_nodes=len(cited_nodes), cited_resolved=len(cited_resolved),
            supports=len(supports), supports_paper_claim=len(sup_pc), claims=len(claims),
            claims_supported=len(claims_supported), claims_unsupported=len(claims - claims_supported),
            papers_supporting=len(papers_supporting), cited_and_supporting=len(cited_and_supporting),
            verifiable_edges=len(verifiable_edges), contradicts=len(contradicts),
            identifiers=len(identifiers), identifiers_doi=doi_like,
        ),
    )

    # ----------------------------------------------------------- config --
    env = read_env(["LLM_BINDING", "LLM_MODEL", "CLAUDE_AGENT_SDK_EFFORT", "MAX_ASYNC_LLM", "MAX_PARALLEL_INSERT",
                    "ENTITY_EXTRACTION_USE_JSON", "ENTITY_TYPE_PROMPT_FILE", "LIGHTRAG_KV_STORAGE",
                    "LIGHTRAG_DOC_STATUS_STORAGE", "LIGHTRAG_GRAPH_STORAGE", "LIGHTRAG_VECTOR_STORAGE",
                    "KG_ENTITY_LOOKUP_METHOD", "SUMMARY_LANGUAGE", "ENABLE_LLM_CACHE", "RERANK_BINDING",
                    "EMBEDDING_BINDING", "EMBEDDING_MODEL", "EMBEDDING_DIM", "LIGHTRAG_PARSER"])
    sys.path.insert(0, str(REPO))
    import lightrag  # noqa: E402
    from lightrag import constants as C  # noqa: E402
    import networkx  # noqa: E402
    import importlib.metadata as im  # noqa: E402
    cli = subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    cpu = ""
    for line in subprocess.run(["lscpu"], capture_output=True, text=True).stdout.splitlines():
        if line.startswith("Model name:"):
            cpu = line.split(":", 1)[1].strip()
    ncpu = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    mem_mib = 0
    for line in subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines():
        if line.startswith("Mem:"):
            mem_mib = int(line.split()[1])
    S["config"] = dict(env=env, python=platform.python_version(), lightrag=lightrag.__version__,
                       networkx=networkx.__version__, claude_agent_sdk=im.version("claude-agent-sdk"),
                       claude_cli=cli, cpu=cpu, ncpu=ncpu, ram_gib=mem_mib / 1024,
                       chunk_size=C.DEFAULT_CHUNK_SIZE if hasattr(C, "DEFAULT_CHUNK_SIZE") else 1200,
                       chunk_overlap=C.DEFAULT_CHUNK_OVERLAP_SIZE if hasattr(C, "DEFAULT_CHUNK_OVERLAP_SIZE") else 100,
                       max_gleaning=C.DEFAULT_MAX_GLEANING, top_k=C.DEFAULT_TOP_K, chunk_top_k=C.DEFAULT_CHUNK_TOP_K,
                       max_entity_tokens=C.DEFAULT_MAX_ENTITY_TOKENS, max_relation_tokens=C.DEFAULT_MAX_RELATION_TOKENS,
                       max_total_tokens=C.DEFAULT_MAX_TOTAL_TOKENS, related_chunk_number=C.DEFAULT_RELATED_CHUNK_NUMBER,
                       summary_max_tokens=getattr(C, "DEFAULT_SUMMARY_MAX_TOKENS", None),
                       force_summary_on_merge=getattr(C, "DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE", None))

    (DATA / "graph_stats.json").write_text(json.dumps(S, indent=2, default=str))

    # ----------------------------------------------------------- tables --
    c, gr, ci, lg, cf, pm, ca = S["corpus"], S["graph"], S["graph"]["citation"], S["log"], S["config"], S["pre_merge"], S["cache"]
    rows = [[d["id"], latex_escape(d["title"]), latex_escape(d["venue"]), latex_escape(d["year"]),
             fmt_int(d["chunks"]), fmt_int(d["tokens"]), fmt_int(d["wall_s"] or 0)] for d in processed]
    write_table(TABLES / "tab_corpus.tex", wide=True, size=r"\scriptsize", placement="p",  # full float page: keeps Section VI next to Table III
                caption=f"The {len(processed)} documents of the corpus after ingestion (venue and year as printed on the first page; dashes where the manuscript carries none). Chunks and tokens are those produced by the chunker; time is the per-document wall-clock span recorded in the document status store.",
                label="tab:corpus", colspec="l p{7.9cm} p{3.1cm} c r r r",
                header=["ID", "Title", "Venue", "Year", "Chunks", "Tokens", "Time (s)"], rows=rows,
                note=f"Five further uploads failed: {failed_font} with a PDF font error (\\texttt{{/DescendantFonts}}) and {failed_dup} rejected as byte-identical duplicates of a document already queued.")
    write_table(TABLES / "tab_graph_summary.tex", caption="Size and shape of the merged knowledge graph.", label="tab:graph_summary",
                colspec="l r", header=["Measure", "Value"], rows=[
                    ["Nodes", fmt_int(gr["nodes"])], ["Edges (undirected)", fmt_int(gr["edges"])],
                    ["Density", f"{gr['density']:.2e}"], ["Mean degree", fmt_float(gr["degree_mean"], 2)],
                    ["Median degree", fmt_int(gr["degree_median"])], ["Maximum degree (``This Paper'')", fmt_int(gr["degree_max"])],
                    ["Connected components", fmt_int(gr["components"])], ["Giant component (nodes)", fmt_int(gr["giant"])],
                    ["Isolated nodes", fmt_int(gr["isolated"])], ["Entity types present", fmt_int(len(gr["type_counts"]))],
                    ["Edges with $\\geq 1$ vocabulary keyword", f"{fmt_int(gr['edges_with_vocab'])} ({fmt_pct(gr['edges_with_vocab'] / gr['edges'])})"]])
    tot = gr["nodes"]
    write_table(TABLES / "tab_entity_types.tex", caption="Nodes per entity type after merging. \\textsc{unknown} nodes are relation endpoints that were never extracted as entities.", label="tab:entity_types",
                colspec="l r r", header=["Type", "Nodes", "Share"],
                rows=[[latex_escape(t) if t != "UNKNOWN" else r"\textsc{unknown}", fmt_int(n), fmt_pct(n / tot)] for t, n in gr["type_counts"].items()])
    tote = gr["edges"]
    write_table(TABLES / "tab_relation_types.tex", caption="Edges whose keyword list contains each controlled relation keyword (an edge merged from several extractions can carry more than one).", label="tab:relation_types",
                colspec="l r r", header=["Keyword", "Edges", "Share of edges"],
                rows=[[r"\texttt{" + latex_escape(k) + "}", fmt_int(n), fmt_pct(n / tote)] for k, n in gr["keyword_containment"].items()])
    write_table(TABLES / "tab_citation_coverage.tex", caption="Citation structure captured by the graph (counts over the merged graph).", label="tab:citation_coverage",
                colspec="p{4.9cm} r", header=["Measure", "Value"], rows=[
                    [r"\texttt{cites} edges", fmt_int(ci["cites"])],
                    [r"\quad between two \texttt{paper} nodes", fmt_int(ci["cites_paper_paper"])],
                    [r"\quad touching the ``This Paper'' hub", fmt_int(ci["cites_hub"])],
                    [r"\quad with a placeholder endpoint (``Reference [$n$]'')", fmt_int(ci["cites_placeholder"])],
                    [r"\quad quoting the citing sentence or its marker (strict)", f"{fmt_int(ci['cites_sentence_strict'])} ({fmt_pct(ci['cites_sentence_strict'] / ci['cites'])})"],
                    [r"\quad with a sentence-length context ($\geq 8$ words, lenient)", f"{fmt_int(ci['cites_sentence'])} ({fmt_pct(ci['cites_sentence'] / ci['cites'])})"],
                    [r"\texttt{paper} nodes in a \texttt{cites} edge", fmt_int(ci["papers_in_cites"])],
                    [r"Distinct cited works (resolved \texttt{paper} nodes)", fmt_int(ci["cited_resolved"])],
                    [r"Placeholder nodes", fmt_int(gr["placeholder_nodes"])],
                    [r"\texttt{supports} edges", fmt_int(ci["supports"])],
                    [r"\quad \texttt{paper}$\rightarrow$\texttt{claim}", fmt_int(ci["supports_paper_claim"])],
                    [r"\texttt{claim} nodes", fmt_int(ci["claims"])],
                    [r"\quad with $\geq 1$ supporting paper", f"{fmt_int(ci['claims_supported'])} ({fmt_pct(ci['claims_supported'] / ci['claims'])})"],
                    [r"\quad without any \texttt{supports} edge", fmt_int(ci["claims_unsupported"])],
                    [r"Cited works that also support a claim", fmt_int(ci["cited_and_supporting"])],
                    [r"\texttt{cites} edges verifiable end to end (sentence + cited work + claim)", fmt_int(ci["verifiable_edges"])],
                    [r"\texttt{contradicts} edges", fmt_int(ci["contradicts"])],
                    [r"\texttt{identifier} nodes / DOI-shaped", f"{fmt_int(ci['identifiers'])} / {fmt_int(ci['identifiers_doi'])}"]])
    write_table(TABLES / "tab_ingestion.tex", caption="Ingestion run: inputs, LLM calls, extraction volume and wall-clock time (from the document status store, the LLM response cache and the server log).", label="tab:ingestion",
                colspec="p{5.0cm} r", header=["Measure", "Value"], rows=[
                    ["PDF files uploaded / distinct by hash", f"{fmt_int(c['pdf_files'])} / {fmt_int(c['pdf_distinct'])}"],
                    ["Documents queued / processed / failed", f"{c['queued']} / {c['processed']} / {c['failed']}"],
                    ["Text chunks (1\\,200 tokens, overlap 100)", fmt_int(c["chunks"])],
                    ["Tokens chunked (mean / max per chunk)", f"{fmt_int(c['tokens'])} ({fmt_int(c['tokens_mean'])} / {fmt_int(c['tokens_max'])})"],
                    ["Cached LLM calls: extraction / summary / keywords", f"{fmt_int(ca['by_type'].get('extract', 0))} / {fmt_int(ca['by_type'].get('summary', 0))} / {fmt_int(ca['by_type'].get('keywords', 0))}"],
                    ["Extraction calls per chunk", fmt_float(ca["by_type"].get("extract", 0) / c["chunks"], 1)],
                    ["Entities / relations emitted per chunk (mean)", f"{fmt_float(lg['entities_per_chunk'])} / {fmt_float(lg['relations_per_chunk'])}"],
                    ["Entity mentions / relation mentions before merging", f"{fmt_int(lg['entities_logged'])} / {fmt_int(lg['relations_logged'])}"],
                    ["Distinct entity names / relation pairs per document (sum)", f"{fmt_int(pm['entity_names'])} / {fmt_int(pm['relation_pairs'])}"],
                    ["Nodes / edges after merging", f"{fmt_int(gr['nodes'])} / {fmt_int(gr['edges'])}"],
                    ["Ingestion sessions (server restarts)", fmt_int(lg["ingestion_sessions"])],
                    ["Ingestion wall-clock (sum of sessions)", human_duration(lg["ingestion_wall_s"])],
                    ["Per-document span: mean / median / max", f"{fmt_int(S['per_doc_wall']['mean_s'])}~s / {fmt_int(S['per_doc_wall']['median_s'])}~s / {fmt_int(S['per_doc_wall']['max_s'])}~s"],
                    ["Aborted attempts before the run (wall-clock)", f"{lg['aborted_sessions']} ({human_duration(lg['aborted_wall_s'])})"],
                    ["Extraction responses reused from those attempts", fmt_int(lg["extract_cache_reused_from_aborted"])],
                    ["Log lines reporting the subscription session limit", fmt_int(lg["session_limit_lines"])]])
    e = cf["env"]
    write_table(TABLES / "tab_config.tex", caption="Runtime configuration of the ingestion and query runs (values read from the server \\texttt{.env}, the package defaults and the installed versions).", label="tab:config",
                colspec="p{3.0cm} p{4.9cm}", header=["Parameter", "Value"], rows=[
                    ["Framework", f"LightRAG {latex_escape(cf['lightrag'])} (fork with the citation profile and the \\texttt{{LABEL}} lookup)"],
                    ["Python / NetworkX", f"{latex_escape(cf['python'])} / {latex_escape(cf['networkx'])}"],
                    ["LLM binding", f"\\texttt{{{latex_escape(e.get('LLM_BINDING', ''))}}} (claude-agent-sdk {latex_escape(cf['claude_agent_sdk'])}, CLI {latex_escape(cf['claude_cli'].split()[0])})"],
                    ["LLM model / effort", f"\\texttt{{{latex_escape(e.get('LLM_MODEL', ''))}}} / {latex_escape(e.get('CLAUDE_AGENT_SDK_EFFORT', ''))}"],
                    ["LLM concurrency / timeout / retries", f"{latex_escape(e.get('MAX_ASYNC_LLM', ''))} / 240~s / 3"],
                    ["Parallel document inserts", latex_escape(e.get("MAX_PARALLEL_INSERT", ""))],
                    ["Parser / chunker", f"legacy PDF text extraction / recursive character, {cf['chunk_size']} tokens, overlap {cf['chunk_overlap']}"],
                    ["Extraction prompt", f"profile \\texttt{{{latex_escape(e.get('ENTITY_TYPE_PROMPT_FILE', ''))}}}, JSON mode, {cf['max_gleaning']} gleaning pass"],
                    ["Schema validator at ingestion", "not enabled (server path)"],
                    ["Graph storage", f"\\texttt{{{latex_escape(e.get('LIGHTRAG_GRAPH_STORAGE', ''))}}} (GraphML file)"],
                    ["KV and doc-status storage", f"\\texttt{{{latex_escape(e.get('LIGHTRAG_KV_STORAGE', ''))}}}, \\texttt{{{latex_escape(e.get('LIGHTRAG_DOC_STATUS_STORAGE', ''))}}}"],
                    ["Vector storage / embeddings", f"\\texttt{{{latex_escape(e.get('LIGHTRAG_VECTOR_STORAGE', ''))}}} / none (bge-m3 endpoint configured but unused)"],
                    ["Entity lookup at query time", f"\\texttt{{{latex_escape(e.get('KG_ENTITY_LOOKUP_METHOD', ''))}}} (name matching, no embeddings)"],
                    ["Retrieval budget", f"top-$k$ {cf['top_k']}, chunk top-$k$ {cf['chunk_top_k']}, {fmt_int(cf['max_entity_tokens'])} / {fmt_int(cf['max_relation_tokens'])} / {fmt_int(cf['max_total_tokens'])} tokens (entities / relations / total)"],
                    ["Reranker", "none"],
                    ["LLM response cache", latex_escape(e.get("ENABLE_LLM_CACHE", ""))],
                    ["Hardware", f"{latex_escape(cf['cpu'])}, {cf['ncpu']} threads, {fmt_float(cf['ram_gib'], 0)}~GiB RAM, no GPU"]])
    write_table(TABLES / "tab_sessions.tex", caption="Server sessions that extracted chunks, from the log timestamps (a session ends at a restart or after ten idle minutes). Aborted sessions used the vector-store configuration that was later abandoned; their cached extractions were reused.", label="tab:sessions",
                colspec="l l l r r r l", header=["\\#", "Start", "End", "Duration", "Chunks", "Docs", "Kind"],
                rows=[[str(i + 1), s["start"], s["end"], human_duration(s["duration_s"]), fmt_int(s["chunks"]), fmt_int(s["docs"]), s["kind"]] for i, s in enumerate(lg["sessions"])], size=r"\scriptsize")

    # ----------------------------------------------------------- macros --
    M.update(
        nPdfFiles=fmt_int(c["pdf_files"]), nPdfDistinct=fmt_int(c["pdf_distinct"]), nDocsQueued=fmt_int(c["queued"]),
        nDocsProcessed=fmt_int(c["processed"]), nDocsFailed=fmt_int(c["failed"]), nDocsFailedFont=fmt_int(c["failed_font"]),
        nDocsFailedDup=fmt_int(c["failed_duplicate"]), nShortDocs=fmt_int(c["short_docs"]),
        nChunks=fmt_int(c["chunks"]), nTokens=fmt_int(c["tokens"]), meanTokensChunk=fmt_int(c["tokens_mean"]),
        minTokensChunk=fmt_int(c["tokens_min"]), maxTokensChunk=fmt_int(c["tokens_max"]), meanChunksDoc=fmt_float(c["chunks_per_doc_mean"], 1),
        nChars=fmt_int(c["chars"]),
        nNodes=fmt_int(gr["nodes"]), nEdges=fmt_int(gr["edges"]), graphDensity=f"{gr['density']:.2e}",
        meanDegree=fmt_float(gr["degree_mean"], 2), maxDegree=fmt_int(gr["degree_max"]), nComponents=fmt_int(gr["components"]),
        giantComponent=fmt_int(gr["giant"]), nIsolated=fmt_int(gr["isolated"]), nTypes=fmt_int(len(gr["type_counts"])),
        pctEdgesVocab=fmt_pct(gr["edges_with_vocab"] / gr["edges"]), nEdgesMultiVocab=fmt_int(gr["edges_multi_vocab"]),
        nPlaceholderNodes=fmt_int(gr["placeholder_nodes"]), hubDegree=fmt_int(gr["hub_degree"]), hubFiles=fmt_int(gr["hub_files"]),
        nCites=fmt_int(ci["cites"]), nCitesPaperPaper=fmt_int(ci["cites_paper_paper"]), nPapersInCites=fmt_int(ci["papers_in_cites"]),
        nCitesHub=fmt_int(ci["cites_hub"]), pctCitesHub=fmt_pct(ci["cites_hub"] / ci["cites"]), nCitesPlaceholder=fmt_int(ci["cites_placeholder"]),
        pctCitesPlaceholder=fmt_pct(ci["cites_placeholder"] / ci["cites"]), nCitesSentence=fmt_int(ci["cites_sentence"]),
        pctCitesSentence=fmt_pct(ci["cites_sentence"] / ci["cites"]), nCitedResolved=fmt_int(ci["cited_resolved"]),
        nCitesSentenceStrict=fmt_int(ci["cites_sentence_strict"]), pctCitesSentenceStrict=fmt_pct(ci["cites_sentence_strict"] / ci["cites"]),
        nSupports=fmt_int(ci["supports"]), nSupportsPaperClaim=fmt_int(ci["supports_paper_claim"]), nClaims=fmt_int(ci["claims"]),
        nClaimsSupported=fmt_int(ci["claims_supported"]), pctClaimsSupported=fmt_pct(ci["claims_supported"] / ci["claims"]),
        nClaimsUnsupported=fmt_int(ci["claims_unsupported"]), nCitedAndSupporting=fmt_int(ci["cited_and_supporting"]),
        nVerifiableEdges=fmt_int(ci["verifiable_edges"]), nContradicts=fmt_int(ci["contradicts"]),
        nIdentifiers=fmt_int(ci["identifiers"]), nIdentifiersDoi=fmt_int(ci["identifiers_doi"]),
        nCacheTotal=fmt_int(ca["total"]), nCacheExtract=fmt_int(ca["by_type"].get("extract", 0)), nCacheSummary=fmt_int(ca["by_type"].get("summary", 0)),
        nCacheKeywords=fmt_int(ca["by_type"].get("keywords", 0)), nCacheExtractReused=fmt_int(lg["extract_cache_reused_from_aborted"]),
        nIngestionSessions=fmt_int(lg["ingestion_sessions"]), ingestionWall=human_duration(lg["ingestion_wall_s"]),
        nAbortedSessions=fmt_int(lg["aborted_sessions"]), abortedWall=human_duration(lg["aborted_wall_s"]),
        perDocWallSum=human_duration(S["per_doc_wall"]["sum_s"]), perDocWallMean=fmt_int(S["per_doc_wall"]["mean_s"]),
        perDocWallMedian=fmt_int(S["per_doc_wall"]["median_s"]), perDocWallMax=fmt_int(S["per_doc_wall"]["max_s"]),
        nSessionLimitLines=fmt_int(lg["session_limit_lines"]), entitiesLogged=fmt_int(lg["entities_logged"]), relationsLogged=fmt_int(lg["relations_logged"]),
        meanEntitiesChunk=fmt_float(lg["entities_per_chunk"]), meanRelationsChunk=fmt_float(lg["relations_per_chunk"]),
        nPreMergeEntityNames=fmt_int(pm["entity_names"]), nPreMergeRelationPairs=fmt_int(pm["relation_pairs"]),
        nQueriesLogged=fmt_int(len(lg["queries"])), nQueriesZeroChunks=fmt_int(sum(1 for q in lg["queries"] if q["chunks"] == 0)),
        llmModel=latex_escape(e.get("LLM_MODEL", "")), llmEffort=latex_escape(e.get("CLAUDE_AGENT_SDK_EFFORT", "")), maxAsync=latex_escape(e.get("MAX_ASYNC_LLM", "")),
        chunkSize=fmt_int(cf["chunk_size"]), chunkOverlap=fmt_int(cf["chunk_overlap"]), maxGleaning=fmt_int(cf["max_gleaning"]),
        topK=fmt_int(cf["top_k"]), pyVersion=latex_escape(cf["python"]), lightragVersion=latex_escape(cf["lightrag"]),
        networkxVersion=latex_escape(cf["networkx"]), sdkVersion=latex_escape(cf["claude_agent_sdk"]), cliVersion=latex_escape(cf["claude_cli"].split()[0]),
        cpuModel=latex_escape(cf["cpu"]), nCpu=latex_escape(cf["ncpu"]), ramGiB=fmt_float(cf["ram_gib"], 0),
        ingestionDate=lg["ingestion_first"].split()[0],
    )
    for t, n in gr["type_counts"].items():
        M["n" + ("Unknown" if t == "UNKNOWN" else t.capitalize()) + "Nodes"] = fmt_int(n)
    for k, n in gr["keyword_containment"].items():
        M["n" + "".join(p.capitalize() for p in k.split("_")) + "Edges"] = fmt_int(n)
    write_macros(DATA / "numbers_stats.tex", M)
    print(json.dumps({k: v for k, v in S["graph"].items() if k not in ("degree_hist", "top_degree")}, indent=1, default=str)[:1500])
    print("sessions:", json.dumps(lg["sessions"], indent=0))
    print("macros:", len(M))


if __name__ == "__main__":
    main()
