"""Schema-adherence audit of the raw extraction outputs (offline, no LLM).

The LLM response cache keeps every extraction response verbatim. The corpus
was ingested through the API server, which does not run the strict schema
validator, so this script replays ``make_scientific_validator()`` over the
cached outputs to measure how far prompt guidance alone kept the model inside
the 9 entity types and the 14-keyword relation vocabulary.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import Counter, defaultdict

from common import (
    DATA, ENTITY_TYPES, TABLES, VOCAB, fmt_float, fmt_int, fmt_pct,
    latex_escape, load_kv, load_validator_module, split_keywords,
    write_macros, write_table,
)

CONTINUE_MARK = "Based on the last extraction task"


def norm_type(raw) -> str | None:
    """Mirror ``_normalize_and_validate_entity_type`` in lightrag/operate.py."""
    t = str(raw or "").strip()
    if not t or any(ch in t for ch in ["'", "(", ")", "<", ">", "|", "/", "\\"]):
        return None
    if "," in t:
        toks = [x.strip() for x in t.split(",") if x.strip()]
        if not toks:
            return None
        t = toks[0]
    return t.replace(" ", "").lower()


def norm_name(raw) -> str:
    return str(raw or "").strip().strip('"').strip()


def main() -> None:
    logging.getLogger("lightrag").setLevel(logging.WARNING)
    try:
        import json_repair
    except ImportError:  # pragma: no cover
        json_repair = None
    validate = load_validator_module().make_scientific_validator()
    cache = load_kv("llm_response_cache")

    P = {p: defaultdict(int) for p in ("initial", "gleaning")}
    type_counter = {p: Counter() for p in P}
    offtype_counter = {p: Counter() for p in P}
    first_kw_counter = {p: Counter() for p in P}
    novocab_counter = {p: Counter() for p in P}
    chunks_seen = {p: set() for p in P}
    chunks_union: dict[str, dict] = {}

    for rec in cache.values():
        if rec.get("cache_type") != "extract":
            continue
        p = "gleaning" if CONTINUE_MARK in str(rec.get("original_prompt", "")) else "initial"
        st = P[p]
        st["records"] += 1
        chunks_seen[p].add(rec.get("chunk_id"))
        raw = rec.get("return", "")
        obj = None
        try:
            obj = json.loads(raw)
        except Exception:
            st["json_repaired"] += 1
            if json_repair is not None:
                try:
                    obj = json_repair.loads(raw)
                except Exception:
                    obj = None
        if not isinstance(obj, dict):
            st["unparseable"] += 1
            continue
        ents = obj.get("entities") or []
        rels = obj.get("relationships") or []
        st["entities"] += len(ents)
        st["relations"] += len(rels)

        maybe_nodes: dict[str, list[dict]] = {}
        key_by_fold: dict[str, str] = {}
        for e in ents:
            name = norm_name(e.get("name"))
            t = norm_type(e.get("type"))
            if not name:
                st["entities_no_name"] += 1
                continue
            if t is None:
                st["entities_invalid_type"] += 1
                continue
            type_counter[p][t] += 1
            if t not in ENTITY_TYPES:
                st["entity_records_off_schema"] += 1
                offtype_counter[p][t] += 1
            k = key_by_fold.setdefault(name.casefold(), name)
            maybe_nodes.setdefault(k, []).append(
                {"entity_name": k, "entity_type": t, "description": e.get("description", ""), "source_id": rec.get("chunk_id")}
            )
        st["entity_names"] += len(maybe_nodes)
        st["entity_names_off_schema"] += sum(
            1 for recs in maybe_nodes.values() if all(r["entity_type"] not in ENTITY_TYPES for r in recs)
        )

        maybe_edges: dict[tuple[str, str], list[dict]] = {}
        for r in rels:
            s, t_ = norm_name(r.get("source")), norm_name(r.get("target"))
            if not s or not t_:
                st["relations_no_endpoint_name"] += 1
                continue
            ks, kt = key_by_fold.get(s.casefold(), s), key_by_fold.get(t_.casefold(), t_)
            kws = str(r.get("keywords", ""))
            toks = split_keywords(kws)
            if toks and toks[0] in VOCAB:
                st["relations_first_keyword_vocab"] += 1
            if toks:
                first_kw_counter[p][toks[0]] += 1
            if not any(tk in VOCAB for tk in toks):
                st["relations_no_vocab"] += 1
                novocab_counter[p][toks[0] if toks else "(empty)"] += 1
            if any(x not in maybe_nodes for x in (ks, kt)):
                st["relations_endpoint_missing_same_response"] += 1
            maybe_edges.setdefault((ks, kt), []).append(
                {"src_id": ks, "tgt_id": kt, "keywords": kws, "description": r.get("description", ""), "source_id": rec.get("chunk_id")}
            )
        st["relation_pairs"] += len(maybe_edges)
        per_chunk = chunks_union.setdefault(rec.get("chunk_id"), {"nodes": {}, "edges": {}})
        for k, recs in maybe_nodes.items():
            per_chunk["nodes"].setdefault(k, []).extend(recs)
        for k, recs in maybe_edges.items():
            per_chunk["edges"].setdefault(k, []).extend(recs)

    # ---- validator replayed per chunk on the union of both passes --------
    global_names = {k.casefold() for ch in chunks_union.values() for k in ch["nodes"]}
    U = defaultdict(int)
    missing_names = Counter()
    for chunk_id, ch in chunks_union.items():
        U["chunks"] += 1
        U["entity_records"] += sum(len(v) for v in ch["nodes"].values())
        U["entity_names"] += len(ch["nodes"])
        U["relation_records"] += sum(len(v) for v in ch["edges"].values())
        U["relation_pairs"] += len(ch["edges"])
        for (a, b), recs in ch["edges"].items():
            miss = [x for x in (a, b) if x not in ch["nodes"]]
            if miss:
                U["relation_records_endpoint_missing"] += len(recs)
                U["relation_pairs_endpoint_missing"] += 1
                if all(x.casefold() in global_names for x in miss):
                    U["relation_pairs_resolvable_globally"] += 1
                for x in miss:
                    missing_names[x] += 1
        nodes_out, edges_out = validate(chunk_id, "", copy.deepcopy(ch["nodes"]), copy.deepcopy(ch["edges"]))
        U["entity_names_kept"] += len(nodes_out)
        U["entity_records_kept"] += sum(len(v) for v in nodes_out.values())
        U["relation_pairs_kept"] += len(edges_out)
        U["relation_records_kept"] += sum(len(v) for v in edges_out.values())
    unknown_names = missing_names

    tot = defaultdict(int)
    for st in P.values():
        for k, v in st.items():
            tot[k] += v
    for p in P:
        P[p]["chunks"] = len(chunks_seen[p])
    tot["chunks"] = len(chunks_seen["initial"] | chunks_seen["gleaning"])

    def share(num, den):
        return num / den if den else 0.0

    out = dict(
        per_pass={p: dict(P[p]) for p in P}, total=dict(tot), per_chunk_union=dict(U),
        types_initial=dict(type_counter["initial"].most_common()),
        types_gleaning=dict(type_counter["gleaning"].most_common()),
        off_schema_types={p: dict(offtype_counter[p].most_common(15)) for p in P},
        first_keywords={p: dict(first_kw_counter[p].most_common(25)) for p in P},
        no_vocab_first_tokens={p: dict(novocab_counter[p].most_common(15)) for p in P},
        unknown_endpoint_names=dict(unknown_names.most_common(20)),
        vocabulary_usage={k: sum(first_kw_counter[p][k] for p in P) for k in VOCAB},
    )
    (DATA / "schema_audit.json").write_text(json.dumps(out, indent=2))

    rows = []
    def row(label, key, pct_of=None, fmt=fmt_int):
        cells = [label]
        for p in ("initial", "gleaning"):
            v = P[p].get(key, 0)
            cell = fmt(v)
            if pct_of:
                cell += f" ({fmt_pct(share(v, P[p].get(pct_of, 0)))})"
            cells.append(cell)
        v = tot.get(key, 0)
        cell = fmt(v)
        if pct_of:
            cell += f" ({fmt_pct(share(v, tot.get(pct_of, 0)))})"
        cells.append(cell)
        rows.append(cells)

    row("Cached responses", "records")
    row("Responses needing JSON repair", "json_repaired")
    row("Unparseable responses", "unparseable")
    row("Entity records emitted", "entities")
    row(r"\quad with a type outside the schema", "entity_records_off_schema", "entities")
    row(r"\quad with an invalid type string", "entities_invalid_type", "entities")
    row("Distinct entity names per response (sum)", "entity_names")
    row(r"\quad dropped by the validator (all records off-schema)", "entity_names_off_schema", "entity_names")
    row("Relation records emitted", "relations")
    row(r"\quad keyword list starting with a vocabulary verb", "relations_first_keyword_vocab", "relations")
    row(r"\quad with no vocabulary verb at all", "relations_no_vocab", "relations")
    row(r"\quad whose endpoint is not listed in the same response", "relations_endpoint_missing_same_response", "relations")
    def urow(label, key, pct_of=None):
        v = U.get(key, 0)
        cell = fmt_int(v) + (f" ({fmt_pct(v / U[pct_of])})" if pct_of and U.get(pct_of) else "")
        rows.append([label, "", "", cell])
    rows.append([r"\emph{Per chunk, both passes merged (as the validator would see them)}", "", "", ""])
    urow("Relation records", "relation_records")
    urow(r"\quad with an endpoint absent from the chunk's entities", "relation_records_endpoint_missing", "relation_records")
    urow(r"\quad of which resolvable to an entity extracted in another chunk (pairs)", "relation_pairs_resolvable_globally")
    urow("Entity records kept by the strict validator", "entity_records_kept", "entity_records")
    urow("Relation records kept by the strict validator", "relation_records_kept", "relation_records")
    off_note = ", ".join(f"\\texttt{{{latex_escape(t)}}} ({n})" for t, n in Counter({**offtype_counter['initial'], **{}}).most_common(0))
    all_off = Counter()
    for p in P:
        all_off.update(offtype_counter[p])
    all_nov = Counter()
    for p in P:
        all_nov.update(novocab_counter[p])
    note = ("Most frequent off-schema types: " + ", ".join(f"\\texttt{{{latex_escape(t)}}} ({n})" for t, n in all_off.most_common(6)) + ". "
            if all_off else "No off-schema entity type was emitted. ")
    note += ("Most frequent first keywords of relations without any vocabulary verb: " + ", ".join(f"\\texttt{{{latex_escape(t)}}} ({n})" for t, n in all_nov.most_common(6)) + "."
             if all_nov else "Every relation carried at least one vocabulary verb.")
    write_table(TABLES / "tab_schema_audit.tex", wide=True, size=r"\scriptsize",
                caption="Schema adherence of the raw extraction outputs, measured by replaying the strict validator over the cached LLM responses (the ingestion run itself did not enforce the schema). Initial pass and gleaning pass are reported separately.",
                label="tab:schema_audit", colspec="p{7.2cm} r r r",
                header=["Measure", "Initial pass", "Gleaning pass", "Both"], rows=rows, note=note)
    M = dict(
        auditRecords=fmt_int(tot["records"]), auditChunks=fmt_int(tot["chunks"]), auditRepaired=fmt_int(tot["json_repaired"]), auditUnparseable=fmt_int(tot["unparseable"]),
        auditEntities=fmt_int(tot["entities"]), auditEntitiesOff=fmt_int(tot["entity_records_off_schema"]), pctEntitiesOff=fmt_pct(share(tot["entity_records_off_schema"], tot["entities"]), 2),
        auditEntitiesInvalid=fmt_int(tot["entities_invalid_type"]),
        auditRelations=fmt_int(tot["relations"]), auditRelFirstVocab=fmt_int(tot["relations_first_keyword_vocab"]), pctRelFirstVocab=fmt_pct(share(tot["relations_first_keyword_vocab"], tot["relations"])),
        auditRelNoVocab=fmt_int(tot["relations_no_vocab"]), pctRelNoVocab=fmt_pct(share(tot["relations_no_vocab"], tot["relations"]), 2),
        auditRelMissingSame=fmt_int(tot["relations_endpoint_missing_same_response"]), pctRelMissingSame=fmt_pct(share(tot["relations_endpoint_missing_same_response"], tot["relations"])),
        auditGleanMissingSame=fmt_int(P["gleaning"]["relations_endpoint_missing_same_response"]), pctGleanMissingSame=fmt_pct(share(P["gleaning"]["relations_endpoint_missing_same_response"], P["gleaning"]["relations"])),
        auditUnionRelations=fmt_int(U["relation_records"]), auditUnionRelMissing=fmt_int(U["relation_records_endpoint_missing"]), pctUnionRelMissing=fmt_pct(share(U["relation_records_endpoint_missing"], U["relation_records"]), 2),
        auditUnionPairsMissing=fmt_int(U["relation_pairs_endpoint_missing"]), auditUnionPairsResolvable=fmt_int(U["relation_pairs_resolvable_globally"]),
        auditEntKept=fmt_int(U["entity_records_kept"]), pctEntKept=fmt_pct(share(U["entity_records_kept"], U["entity_records"])),
        auditRelKept=fmt_int(U["relation_records_kept"]), pctRelKept=fmt_pct(share(U["relation_records_kept"], U["relation_records"]), 2),
        auditUnknownTop=", ".join(f"``{latex_escape(n)}'' ({c})" for n, c in missing_names.most_common(3)),
        auditEntPerInitial=fmt_float(share(P["initial"]["entities"], P["initial"]["records"])), auditEntPerGleaning=fmt_float(share(P["gleaning"]["entities"], P["gleaning"]["records"])),
        auditRelPerInitial=fmt_float(share(P["initial"]["relations"], P["initial"]["records"])), auditRelPerGleaning=fmt_float(share(P["gleaning"]["relations"], P["gleaning"]["records"])),
        auditOffTypes=", ".join(f"\\texttt{{{latex_escape(t)}}} ({n})" for t, n in all_off.most_common(5)) or "none",
        auditNoVocabTokens=", ".join(f"\\texttt{{{latex_escape(t)}}} ({n})" for t, n in all_nov.most_common(5)) or "none",
    )
    write_macros(DATA / "numbers_audit.tex", M)
    print(json.dumps(out["per_pass"], indent=1))
    print("total:", dict(tot))
    print("off-schema types:", all_off.most_common(10))
    print("no-vocab first tokens:", all_nov.most_common(10))
    print("per-chunk union:", dict(U))
    print("missing endpoints (union):", unknown_names.most_common(8))
    print("vocab usage (first keyword):", out["vocabulary_usage"])


if __name__ == "__main__":
    main()
