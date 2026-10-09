"""Can the label index resolve a cited work from what a verification query
carries (its title, a partial title, its first author)? No LLM involved.

Replays NetworkXStorage.search_labels scoring over the merged graph through
the real ``_search_entity_labels`` helper (keyword splitting + token fallback),
with the production top-k of 40.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys

from common import (
    DATA, HUB, REPO, TABLES, cited_endpoint, fmt_int, fmt_pct, is_placeholder,
    load_graph, split_keywords, write_macros, write_table,
)

sys.path.insert(0, str(REPO))
from lightrag.operate import _search_entity_labels  # noqa: E402

TOP_K = 40
KS = (1, 5, 10, 40)


class NxLabelIndex:
    """Same scoring as NetworkXStorage.search_labels (lightrag/kg/networkx_impl.py)."""

    def __init__(self, graph):
        self.names = [str(n) for n in graph.nodes()]

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        q = query.lower().strip()
        if not q:
            return []
        matches = []
        for node in self.names:
            low = node.lower()
            if q not in low:
                continue
            if low == q:
                score = 1000
            elif low.startswith(q):
                score = 500
            else:
                score = 100 - len(node)
                if f" {q}" in low or f"_{q}" in low:
                    score += 50
            matches.append((node, score))
        matches.sort(key=lambda x: (-x[1], x[0]))
        return [m[0] for m in matches[:limit]]


def partial_title(name: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']*", name) if len(t) >= 3]
    return " ".join(toks[:3])


async def main():
    g = load_graph()
    types = {n: str(d.get("entity_type", "")) for n, d in g.nodes(data=True)}
    index = NxLabelIndex(g)
    cited = set()
    for u, v, d in g.edges(data=True):
        if "cites" not in set(split_keywords(d.get("keywords", ""))):
            continue
        c = cited_endpoint(g, u, v)
        if types[c] == "paper" and c != HUB and not is_placeholder(c):
            cited.add(c)
    cited = sorted(cited)

    def first_author(p):
        for q in sorted(g.neighbors(p)):
            if types[q] == "author" and "authored_by" in set(split_keywords(g.edges[p, q].get("keywords", ""))):
                return q
        return None

    forms = {
        "full title": lambda p: p,
        "partial title": partial_title,
        "first author": first_author,
    }
    results = {}
    for form, fn in forms.items():
        ranks = []
        n_app = 0
        query_node_hit1 = 0
        neighbour_reach = 0
        for p in cited:
            q = fn(p)
            if not q:
                continue
            n_app += 1
            hits = await _search_entity_labels(q, index, TOP_K)
            ranks.append(hits.index(p) + 1 if p in hits else None)
            if hits and hits[0] == q:
                query_node_hit1 += 1
            if any(h == p or g.has_edge(h, p) for h in hits):
                neighbour_reach += 1
        results[form] = dict(
            applicable=n_app,
            **{f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) for k in KS},
            query_node_hit1=query_node_hit1, reach_via_edges=neighbour_reach,
            split_into_terms=sum(1 for p in cited if fn(p) and len(split_keywords(fn(p))) > 1),
        )
    out = dict(cited_works=len(cited), top_k=TOP_K, results=results)
    (DATA / "label_coverage.json").write_text(json.dumps(out, indent=2))
    rows = []
    for form, r in results.items():
        n = r["applicable"]
        rows.append([form, fmt_int(n)] + [f"{fmt_pct(r[f'hit@{k}'] / n)}" for k in KS] + [fmt_pct(r["reach_via_edges"] / n)])
    write_table(TABLES / "tab_label_coverage.tex",
                caption=f"Label-index resolution of the {len(cited):,} distinct cited works: share recovered within the first $k$ entities returned by the name search used in \\texttt{{LABEL}} lookup (top-$k$ = {TOP_K}), for three query formulations a verification query may carry.",
                label="tab:label_coverage", colspec="l r r r r r r", size=r"\scriptsize",
                header=["Query", "$n$", "hit@1", "hit@5", "hit@10", "hit@40", "reach"], rows=rows,
                note="``partial title'': the first three title tokens of at least three characters. ``reach'': the cited work is itself among the returned entities or is adjacent to one of them, i.e. it enters the context through the edge expansion that follows the lookup. ``first author'' applies only to cited works linked to an \\texttt{author} node; the search returns the author node, never the paper, so the work is reached through its \\texttt{authored\\_by} edge. Titles containing commas are split into several search terms by the keyword splitter.")
    M = dict(labelCitedWorks=fmt_int(len(cited)))
    for form, r in results.items():
        key = "".join(w.capitalize() for w in re.sub(r"[^a-z ]", "", form.lower()).split())
        n = r["applicable"]
        M["label" + key + "N"] = fmt_int(n)
        for k in KS:
            M[f"label{key}Hit{['One','Five','Ten','Forty'][KS.index(k)]}"] = fmt_pct(r[f"hit@{k}"] / n)
        M["label" + key + "Reach"] = fmt_pct(r["reach_via_edges"] / n)
        M["label" + key + "QueryNodeHitOne"] = fmt_pct(r["query_node_hit1"] / n)
    write_macros(DATA / "numbers_label.tex", M)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
