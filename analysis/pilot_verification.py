"""Pilot: KG-grounded versus LLM-only claim-support judgments on real
citation contexts from the corpus graph.

Runs IN PROCESS on a copy of the working directory
(``analysis/rag_storage_pilot``), so the server and the original storage are
never touched. Two conditions per sampled ``cites`` edge:

  A. KG-grounded: LightRAG ``hybrid`` retrieval in LABEL lookup mode (no
     embeddings) assembles entities, relationships and chunks; the LLM judges
     the pair from that evidence only.
  B. LLM-only: the same prompt without evidence (parametric memory only).

No gold labels exist; the script reports label distributions, A/B agreement,
retrieval coverage and verifiable grounding (evidence identifiers that really
occur in the retrieved context), and writes an annotation sheet for human
labelling. Resumable: results are appended per pair.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np

from common import (
    ANALYSIS, DATA, HUB, REPO, TABLES, cited_endpoint, fmt_ci_pct, fmt_float,
    fmt_int, fmt_pct, has_citing_sentence, is_placeholder, load_graph,
    node_file_paths, read_env, split_keywords, wilson_ci, write_macros,
    write_table,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

PILOT_DIR = ANALYSIS / "rag_storage_pilot"
RESULTS = DATA / "pilot_results.json"
LLM_CACHE = DATA / "pilot_llm_cache.json"
LABELS = ["supports", "partially_supports", "unrelated", "contradicts", "insufficient_evidence"]

SYSTEM_A = (
    "You audit citations in scientific articles. Decide whether a cited work supports the "
    "statement it is cited for, using ONLY the evidence provided (knowledge-graph entities, "
    "relationships and document passages). Be strict: if the evidence does not describe the "
    "cited work or the statement, answer insufficient_evidence."
)
SYSTEM_B = (
    "You audit citations in scientific articles. Decide whether a cited work supports the "
    "statement it is cited for, using only your own knowledge of the cited work. No evidence "
    "is provided. If you do not know the cited work well enough, answer insufficient_evidence."
)
USER_TMPL = """Citing article: "{citing}"
Citing sentence: "{sentence}"
Cited work: "{cited}"

{evidence}

Classify the relation between the cited work and the citing sentence with exactly one label:
- supports: the cited work substantiates the statement;
- partially_supports: it substantiates only part of it or a weaker form;
- unrelated: it does not address the statement;
- contradicts: it argues against the statement;
- insufficient_evidence: the available evidence does not allow a judgment.

Respond with a single JSON object:
{{"label": "<one of the five labels>", "rationale": "<at most two sentences>", "evidence_ids": ["<exact names of the entities, relationships or chunk reference ids you relied on; empty list if none>"]}}"""


def sample_pairs(g, n: int, per_doc: int, seed: int) -> list[dict]:
    types = {k: str(d.get("entity_type", "")) for k, d in g.nodes(data=True)}
    titles = {r["file_path"]: r["title"] for r in csv.DictReader((ANALYSIS / "corpus_titles.csv").open())}

    def kws(u, v):
        return set(split_keywords(g.edges[u, v].get("keywords", "")))

    eligible = []
    for u, v, d in g.edges(data=True):
        if "cites" not in set(split_keywords(d.get("keywords", ""))):
            continue
        c = cited_endpoint(g, u, v)
        o = v if c == u else u
        if types[c] != "paper" or c == HUB or is_placeholder(c):
            continue
        if not has_citing_sentence(d.get("description", ""), strict=True):
            continue
        claims = [q for q in g.neighbors(c) if types[q] == "claim" and "supports" in kws(c, q)]
        if not claims:
            continue
        files = sorted(node_file_paths(d))
        doc = files[0] if files else ""
        citing = o if o != HUB else f"the article {titles.get(doc, doc)}"
        eligible.append(dict(key=f"{u}|||{v}", citing=citing, citing_node=o, cited=c, doc=doc,
                             sentence=d.get("description", "").strip(), claims=claims))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    per = Counter()
    out = []
    for e in eligible:
        if per[e["doc"]] >= per_doc:
            continue
        per[e["doc"]] += 1
        out.append(e)
        if len(out) >= n:
            break
    return out


def parse_context(ctx: str) -> dict:
    """Parse LightRAG's kg_query_context: three headed sections, each a
    ```json fence holding one JSON object per line."""
    sections = {"entities": "Knowledge Graph Data (Entity)", "relations": "Knowledge Graph Data (Relationship)", "chunks": "Document Chunks"}
    out = {k: [] for k in sections}
    for key, heading in sections.items():
        i = ctx.find(heading)
        if i < 0:
            continue
        m = re.search(r"```json\s*(.*?)```", ctx[i:], flags=re.S)
        if not m:
            continue
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out[key].append(obj)
    return out


def parse_label(text: str) -> dict:
    try:
        obj = json.loads(text)
    except Exception:
        import json_repair
        try:
            obj = json_repair.loads(text)
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return dict(label="unparseable", rationale=text[:300], evidence_ids=[])
    lab = str(obj.get("label", "")).strip().lower().replace(" ", "_")
    if lab not in LABELS:
        lab = "unparseable"
    ev = obj.get("evidence_ids") or []
    if isinstance(ev, str):
        ev = [ev]
    return dict(label=lab, rationale=str(obj.get("rationale", ""))[:600], evidence_ids=[str(x) for x in ev][:20])


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(1) * m.sum(0)).sum() / (n * n))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def bootstrap_kappa_ci(a: list[str], b: list[str], resamples: int = BOOTSTRAP_RESAMPLES,
                       seed: int = BOOTSTRAP_SEED, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval for Cohen's kappa over the paired labels."""
    rng = np.random.default_rng(seed)
    n = len(a)
    ks = []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        ks.append(cohen_kappa([a[i] for i in idx], [b[i] for i in idx]))
    return (float(np.percentile(ks, 100 * alpha / 2)), float(np.percentile(ks, 100 * (1 - alpha / 2))))


async def build_rag():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.claude_agent_sdk import claude_agent_sdk_complete

    env = read_env(["LLM_MODEL", "MAX_ASYNC_LLM", "SUMMARY_LANGUAGE"])
    rag = LightRAG(
        working_dir=str(PILOT_DIR),
        llm_model_func=claude_agent_sdk_complete,
        llm_model_name=env.get("LLM_MODEL", "sonnet"),
        llm_model_max_async=int(env.get("MAX_ASYNC_LLM", "4")),
        embedding_func=None,
        vector_storage="NoopVectorDBStorage",
        graph_storage="NetworkXStorage",
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        kg_entity_lookup_method="LABEL",
        addon_params={"entity_type_prompt_file": "scientific_citation.yml", "language": env.get("SUMMARY_LANGUAGE", "English")},
        enable_llm_cache=True,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, env.get("LLM_MODEL", "sonnet")


async def llm_json(model: str, system: str, user: str, cache: dict) -> tuple[str, float, bool]:
    from lightrag.llm.claude_agent_sdk import claude_agent_sdk_complete_if_cache

    key = hashlib.sha256((model + "\n" + system + "\n" + user).encode()).hexdigest()
    if key in cache:
        return cache[key]["text"], cache[key]["latency_s"], True
    t0 = time.perf_counter()
    text = await claude_agent_sdk_complete_if_cache(model, user, system_prompt=system, response_format={"type": "json_object"})
    lat = time.perf_counter() - t0
    cache[key] = dict(text=text, latency_s=lat)
    LLM_CACHE.write_text(json.dumps(cache, indent=1))
    return text, lat, False


async def retrieve(rag, p: dict) -> tuple[str, dict]:
    from lightrag.base import QueryParam

    question = f'Does the work "{p["cited"]}" support the statement: "{p["sentence"]}"?'
    t0 = time.perf_counter()
    ctx = await rag.aquery(question, param=QueryParam(mode="hybrid", only_need_context=True))
    ret_lat = time.perf_counter() - t0
    if isinstance(ctx, dict):
        ctx = json.dumps(ctx, ensure_ascii=False)
    ctx = ctx or ""
    parsed = parse_context(ctx)
    low = ctx.casefold()
    probe = re.sub(r"\s+", " ", p["sentence"].split("<SEP>")[0])[:60].casefold()
    retrieval = dict(
        latency_s=ret_lat, entities=len(parsed["entities"]), relations=len(parsed["relations"]),
        chunks=len(parsed["chunks"]), context_chars=len(ctx),
        cited_in_context=p["cited"].casefold() in low,
        claim_in_context=any(c.casefold() in low for c in p["claims"]),
        sentence_in_chunks=any(probe in re.sub(r"\s+", " ", json.dumps(ch, ensure_ascii=False)).casefold() for ch in parsed["chunks"]),
    )
    return ctx, retrieval


async def recount(args):
    """Re-run retrieval only (keyword extraction is cached) and refresh the
    retrieval measures of stored results; no judgment call is repeated."""
    g = load_graph()
    pairs = {p["key"]: p for p in sample_pairs(g, args.n, args.per_doc, args.seed)}
    results = json.loads(RESULTS.read_text())
    rag, _ = await build_rag()
    for i, rec in enumerate(results, 1):
        p = pairs[rec["key"]]
        ctx, retrieval = await retrieve(rag, p)
        low = ctx.casefold()
        rec["retrieval"] = retrieval
        rec["A_kg"]["evidence_in_context"] = sum(1 for e in rec["A_kg"]["evidence_ids"] if e.strip() and e.strip().casefold() in low)
        print(f"[{i}/{len(results)}] {retrieval['entities']}e/{retrieval['relations']}r/{retrieval['chunks']}c cited={retrieval['cited_in_context']} claim={retrieval['claim_in_context']} sent={retrieval['sentence_in_chunks']}")
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    await rag.finalize_storages()
    summarize(results)


async def run(args):
    g = load_graph()
    pairs = sample_pairs(g, args.n, args.per_doc, args.seed)
    print(f"eligible sample: {len(pairs)} pairs from {len({p['doc'] for p in pairs})} documents")
    if args.dry_run:
        for p in pairs[: args.limit or 5]:
            print(json.dumps({k: p[k] for k in ("citing", "cited", "sentence", "doc")}, ensure_ascii=False)[:400])
        return
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    done = {r["key"] for r in results}
    cache = json.loads(LLM_CACHE.read_text()) if LLM_CACHE.exists() else {}
    from lightrag.base import QueryParam

    logging.getLogger("lightrag").setLevel(logging.INFO)
    fh = logging.FileHandler(ANALYSIS / "pilot.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("lightrag").addHandler(fh)
    rag, model = await build_rag()
    todo = [p for p in pairs if p["key"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    for i, p in enumerate(todo, 1):
        ctx, retrieval = await retrieve(rag, p)
        low = ctx.casefold()
        rec = dict(
            key=p["key"], citing=p["citing"], cited=p["cited"], doc=p["doc"], sentence=p["sentence"],
            kg_claims=p["claims"][:5], retrieval=retrieval,
        )
        user_a = USER_TMPL.format(citing=p["citing"], sentence=p["sentence"], cited=p["cited"],
                                  evidence="Evidence retrieved from the knowledge graph:\n" + ctx)
        user_b = USER_TMPL.format(citing=p["citing"], sentence=p["sentence"], cited=p["cited"], evidence="No evidence is provided.")
        text_a, lat_a, cached_a = await llm_json(model, SYSTEM_A, user_a, cache)
        text_b, lat_b, cached_b = await llm_json(model, SYSTEM_B, user_b, cache)
        la, lb = parse_label(text_a), parse_label(text_b)
        grounded = [e for e in la["evidence_ids"] if e.strip() and e.strip().casefold() in low]
        la["evidence_in_context"] = len(grounded)
        la["latency_s"], lb["latency_s"] = lat_a, lat_b
        la["from_cache"], lb["from_cache"] = cached_a, cached_b
        rec["A_kg"] = la
        rec["B_llm_only"] = lb
        results.append(rec)
        RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"[{i}/{len(todo)}] A={la['label']} B={lb['label']} ret={rec['retrieval']['entities']}e/{rec['retrieval']['relations']}r/{rec['retrieval']['chunks']}c "
              f"cited_in_ctx={rec['retrieval']['cited_in_context']} lat={ret_lat:.1f}s/{lat_a:.1f}s/{lat_b:.1f}s")
    await rag.finalize_storages()
    summarize(results)


def summarize(results: list[dict]) -> None:
    n = len(results)
    if not n:
        return
    la = [r["A_kg"]["label"] for r in results]
    lb = [r["B_llm_only"]["label"] for r in results]
    dist_a, dist_b = Counter(la), Counter(lb)
    agree = sum(1 for a, b in zip(la, lb) if a == b)
    kappa = cohen_kappa(la, lb)
    kappa_ci = bootstrap_kappa_ci(la, lb)
    ret = [r["retrieval"] for r in results]
    ev_total = sum(len(r["A_kg"]["evidence_ids"]) for r in results)
    ev_grounded = sum(r["A_kg"]["evidence_in_context"] for r in results)
    a_with_ev = sum(1 for r in results if r["A_kg"]["evidence_ids"])
    b_with_ev = sum(1 for r in results if r["B_llm_only"]["evidence_ids"])
    lat = dict(retrieval=[x["latency_s"] for x in ret], A=[r["A_kg"]["latency_s"] for r in results], B=[r["B_llm_only"]["latency_s"] for r in results])
    S = dict(n=n, documents=len({r["doc"] for r in results}), labels_A=dict(dist_a), labels_B=dict(dist_b), agreement=agree, kappa=kappa,
             retrieval=dict(entities_mean=float(np.mean([x["entities"] for x in ret])), relations_mean=float(np.mean([x["relations"] for x in ret])),
                            chunks_mean=float(np.mean([x["chunks"] for x in ret])), chunks_zero=sum(1 for x in ret if x["chunks"] == 0),
                            cited_in_context=sum(1 for x in ret if x["cited_in_context"]), claim_in_context=sum(1 for x in ret if x["claim_in_context"]),
                            sentence_in_chunks=sum(1 for x in ret if x["sentence_in_chunks"]), context_chars_mean=float(np.mean([x["context_chars"] for x in ret]))),
             evidence=dict(A_answers_with_ids=a_with_ev, B_answers_with_ids=b_with_ev, A_ids_total=ev_total, A_ids_in_context=ev_grounded),
             latency=dict(retrieval_median=float(np.median(lat["retrieval"])), A_median=float(np.median(lat["A"])), B_median=float(np.median(lat["B"]))))
    # 95% uncertainty: Wilson intervals for proportions over the n pairs (or the
    # identifiers), percentile bootstrap for kappa. One run per condition only.
    S["ci95"] = dict(
        method="Wilson score interval for proportions; percentile bootstrap "
               f"({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}) for kappa",
        agreement=wilson_ci(agree, n), kappa=kappa_ci,
        labels_A={lab: wilson_ci(dist_a.get(lab, 0), n) for lab in LABELS},
        labels_B={lab: wilson_ci(dist_b.get(lab, 0), n) for lab in LABELS},
        claim_in_context=wilson_ci(S["retrieval"]["claim_in_context"], n),
        sentence_in_chunks=wilson_ci(S["retrieval"]["sentence_in_chunks"], n),
        ids_in_context=wilson_ci(ev_grounded, ev_total) if ev_total else (0.0, 0.0),
    )
    (DATA / "pilot_summary.json").write_text(json.dumps(S, indent=2))
    def ci_tab(k: int, m: int) -> str:  # compact form for the table column
        lo, hi = wilson_ci(k, m)
        return f"[{100 * lo:.0f}--{100 * hi:.0f}]"
    rows = [["Pairs (documents)", f"{n} ({S['documents']})", ""]]
    rows.append([r"\emph{Label distribution}", "KG-grounded (A)", "LLM-only (B)"])
    for lab in LABELS + ["unparseable"]:
        if dist_a.get(lab, 0) or dist_b.get(lab, 0):
            ci_a = ci_tab(dist_a.get(lab, 0), n)
            ci_b = ci_tab(dist_b.get(lab, 0), n)
            rows.append([r"\hspace{2pt}\texttt{" + lab.replace("_", r"\_") + "}", f"{dist_a.get(lab, 0)} ({fmt_pct(dist_a.get(lab, 0) / n)}) {ci_a}", f"{dist_b.get(lab, 0)} ({fmt_pct(dist_b.get(lab, 0) / n)}) {ci_b}"])
    rows.append(["Agreement A = B", f"{agree} ({fmt_pct(agree / n)}) {ci_tab(agree, n)}", ""])
    rows.append(["Cohen's $\\kappa$ [bootstrap CI]", f"{kappa:.2f} [{kappa_ci[0]:.2f}, {kappa_ci[1]:.2f}]", ""])
    rows.append([r"\emph{Retrieval (condition A)}", "", ""])
    rr = S["retrieval"]
    rows.append(["Entities / relations / chunks per query (mean)", f"{fmt_float(rr['entities_mean'])} / {fmt_float(rr['relations_mean'])} / {fmt_float(rr['chunks_mean'])}", ""])
    rows.append(["Queries with zero chunks", fmt_int(rr["chunks_zero"]), ""])
    rows.append(["Cited work present in the context", f"{rr['cited_in_context']} ({fmt_pct(rr['cited_in_context'] / n)})", ""])
    rows.append(["A KG claim of the cited work present", f"{rr['claim_in_context']} ({fmt_pct(rr['claim_in_context'] / n)}) {ci_tab(rr['claim_in_context'], n)}", ""])
    rows.append(["Citing sentence found in a retrieved chunk", f"{rr['sentence_in_chunks']} ({fmt_pct(rr['sentence_in_chunks'] / n)}) {ci_tab(rr['sentence_in_chunks'], n)}", ""])
    rows.append([r"\emph{Grounding of the answer}", "A", "B"])
    rows.append(["Answers listing evidence identifiers", f"{a_with_ev} ({fmt_pct(a_with_ev / n)})", f"{b_with_ev} ({fmt_pct(b_with_ev / n)})"])
    rows.append(["Identifiers that occur verbatim in the context", f"{ev_grounded} / {ev_total} ({fmt_pct(ev_grounded / ev_total) if ev_total else '--'}) {ci_tab(ev_grounded, ev_total) if ev_total else ''}", "n/a"])
    rows.append([r"\emph{Latency (median)}", "", ""])
    rows.append(["Retrieval / judgment A / judgment B", f"{fmt_float(S['latency']['retrieval_median'])}~s / {fmt_float(S['latency']['A_median'])}~s", f"{fmt_float(S['latency']['B_median'])}~s"])
    write_table(TABLES / "tab_pilot.tex", caption=f"Pilot on {n} citation contexts: KG-grounded (A) versus LLM-only (B) claim-support judgments by the same model. No gold labels exist, so the table reports distributions, agreement, retrieval coverage and verifiable grounding, not accuracy. Brackets give 95\\% confidence intervals in percent.",
                label="tab:pilot", colspec="@{}p{3.3cm} r r@{}", header=["Measure", "A: KG-grounded", "B: LLM-only"], rows=rows, size=r"\scriptsize",
                note=f"Intervals: Wilson score for proportions over the {n} pairs (or over the identifiers); percentile bootstrap ({BOOTSTRAP_RESAMPLES:,} resamples, fixed seed) for $\\kappa$. Each condition was run once, so the intervals reflect sampling of pairs, not the run-to-run variation of the model.")
    M = dict(pilotN=fmt_int(n), pilotDocs=fmt_int(S["documents"]), pilotAgree=fmt_int(agree), pilotAgreePct=fmt_pct(agree / n), pilotKappa=f"{kappa:.2f}",
             pilotAgreeCI=fmt_ci_pct(*wilson_ci(agree, n)), pilotKappaCI=f"[{kappa_ci[0]:.2f}, {kappa_ci[1]:.2f}]",
             pilotClaimInCtxCI=fmt_ci_pct(*wilson_ci(S["retrieval"]["claim_in_context"], n)),
             pilotSentenceInChunksCI=fmt_ci_pct(*wilson_ci(S["retrieval"]["sentence_in_chunks"], n)),
             pilotIdsGroundedCI=fmt_ci_pct(*wilson_ci(ev_grounded, ev_total)) if ev_total else "--",
             pilotBootstrapResamples=f"{BOOTSTRAP_RESAMPLES:,}",
             pilotChunksMean=fmt_float(rr["chunks_mean"]), pilotEntitiesMean=fmt_float(rr["entities_mean"]), pilotRelationsMean=fmt_float(rr["relations_mean"]),
             pilotChunksZero=fmt_int(rr["chunks_zero"]), pilotCitedInCtx=fmt_int(rr["cited_in_context"]), pilotCitedInCtxPct=fmt_pct(rr["cited_in_context"] / n),
             pilotClaimInCtx=fmt_int(rr["claim_in_context"]), pilotClaimInCtxPct=fmt_pct(rr["claim_in_context"] / n), pilotSentenceInChunks=fmt_int(rr["sentence_in_chunks"]),
             pilotSentenceInChunksPct=fmt_pct(rr["sentence_in_chunks"] / n), pilotAEvidence=fmt_int(a_with_ev), pilotAEvidencePct=fmt_pct(a_with_ev / n),
             pilotBEvidence=fmt_int(b_with_ev), pilotBEvidencePct=fmt_pct(b_with_ev / n), pilotIdsTotal=fmt_int(ev_total), pilotIdsGrounded=fmt_int(ev_grounded),
             pilotIdsGroundedPct=fmt_pct(ev_grounded / ev_total) if ev_total else "--", pilotRetMedian=fmt_float(S["latency"]["retrieval_median"]),
             pilotAMedian=fmt_float(S["latency"]["A_median"]), pilotBMedian=fmt_float(S["latency"]["B_median"]))
    marker = [r for r in results if re.search(r"\[\d+\]", r["cited"])]
    titled = [r for r in results if not re.search(r"\[\d+\]", r["cited"])]
    INS = "insufficient_evidence"
    M.update(
        pilotBAbstainACommit=fmt_int(sum(1 for r in results if r["B_llm_only"]["label"] == INS and r["A_kg"]["label"] != INS)),
        pilotAAbstainBCommit=fmt_int(sum(1 for r in results if r["A_kg"]["label"] == INS and r["B_llm_only"]["label"] != INS)),
        pilotBothAbstain=fmt_int(sum(1 for r in results if r["A_kg"]["label"] == INS and r["B_llm_only"]["label"] == INS)),
        pilotDisagree=fmt_int(n - agree),
        pilotMarkerPairs=fmt_int(len(marker)), pilotTitledPairs=fmt_int(len(titled)),
        pilotMarkerBAbstain=fmt_int(sum(1 for r in marker if r["B_llm_only"]["label"] == INS)),
        pilotMarkerAAbstain=fmt_int(sum(1 for r in marker if r["A_kg"]["label"] == INS)),
        pilotMarkerACommit=fmt_int(sum(1 for r in marker if r["A_kg"]["label"] != INS)),
        pilotMarkerBothAbstain=fmt_int(sum(1 for r in marker if r["A_kg"]["label"] == INS and r["B_llm_only"]["label"] == INS)),
        pilotTitledAAbstain=fmt_int(sum(1 for r in titled if r["A_kg"]["label"] == INS)),
        pilotTitledBAbstain=fmt_int(sum(1 for r in titled if r["B_llm_only"]["label"] == INS)),
    )
    for lab in LABELS + ["unparseable"]:
        key = "".join(w.capitalize() for w in lab.split("_"))
        M[f"pilotA{key}"] = fmt_int(dist_a.get(lab, 0))
        M[f"pilotB{key}"] = fmt_int(dist_b.get(lab, 0))
        M[f"pilotA{key}Pct"] = fmt_pct(dist_a.get(lab, 0) / n)
        M[f"pilotB{key}Pct"] = fmt_pct(dist_b.get(lab, 0) / n)
        M[f"pilotA{key}CI"] = fmt_ci_pct(*wilson_ci(dist_a.get(lab, 0), n))
        M[f"pilotB{key}CI"] = fmt_ci_pct(*wilson_ci(dist_b.get(lab, 0), n))
    write_macros(DATA / "numbers_pilot.tex", M)
    with (DATA / "annotation_sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "document", "citing", "cited_work", "citing_sentence", "kg_claims_of_cited_work", "label_A_kg", "rationale_A", "label_B_llm_only", "rationale_B", "human_label", "human_comment"])
        for i, r in enumerate(results, 1):
            w.writerow([i, r["doc"], r["citing"], r["cited"], r["sentence"], " | ".join(r["kg_claims"]), r["A_kg"]["label"], r["A_kg"]["rationale"], r["B_llm_only"]["label"], r["B_llm_only"]["rationale"], "", ""])
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--per-doc", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="process at most this many pending pairs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--recount", action="store_true", help="re-run retrieval for stored results (no judgment calls)")
    a = ap.parse_args()
    if a.summarize_only:
        summarize(json.loads(RESULTS.read_text()))
    elif a.recount:
        asyncio.run(recount(a))
    else:
        asyncio.run(run(a))
