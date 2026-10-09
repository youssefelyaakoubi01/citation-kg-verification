"""Figures for the article, rendered from the merged GraphML (read-only).

Colours follow a single categorical order (blue, orange, aqua, yellow,
magenta, green, violet, red); types beyond the eighth fold into grey.
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import (  # noqa: E402
    DATA, FIGURES, HUB, VOCAB, cited_endpoint, has_citing_sentence, is_placeholder,
    load_graph, load_kv, node_file_paths, split_keywords,
)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY, INK, INK2, GRID, AXIS = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TYPE_SLOT = {"paper": 0, "author": 1, "concept": 2, "methodology": 3, "claim": 4, "identifier": 5, "venue": 6, "organization": 7}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)


def type_color(t):
    return PALETTE[TYPE_SLOT[t]] if t in TYPE_SLOT else GREY


def hbar(items, name, xlabel, highlight_last_grey=False):
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(3.45, 0.2 * len(items) + 0.7))
    y = np.arange(len(items))[::-1]
    colors = [PALETTE[0]] * len(items)
    if highlight_last_grey:
        colors = [GREY if k in ("other", "UNKNOWN") else PALETTE[0] for k in labels]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([("unknown" if k == "UNKNOWN" else k).replace("_", r"\_") if False else ("unknown" if k == "UNKNOWN" else k) for k in labels], color=INK)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, f"{v:,}", va="center", ha="left", fontsize=7, color=INK2)
    ax.set_xlim(0, max(values) * 1.14)
    save(fig, name)


def main():
    g = load_graph()
    types = {n: str(d.get("entity_type", "")) for n, d in g.nodes(data=True)}
    deg = dict(g.degree())

    # 1. entity types ------------------------------------------------------
    hbar(Counter(types.values()).most_common(), "fig_entity_types", "nodes", highlight_last_grey=True)

    # 2. relation keywords -------------------------------------------------
    contain = Counter()
    for _, _, d in g.edges(data=True):
        toks = set(split_keywords(d.get("keywords", "")))
        for k in VOCAB:
            if k in toks:
                contain[k] += 1
    hbar(contain.most_common(), "fig_relation_types", "edges containing the keyword")

    # 3. degree distribution ----------------------------------------------
    hist = Counter(deg.values())
    ks = sorted(k for k in hist if k > 0)
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ax.scatter(ks, [hist[k] for k in ks], s=9, color=PALETTE[0], alpha=0.85, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("degree $k$")
    ax.set_ylabel("number of nodes")
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    hub_k = deg.get(HUB, 0)
    ax.annotate(f"\u201c{HUB}\u201d hub\n$k$ = {hub_k:,}", xy=(hub_k, 1), xytext=(hub_k / 25, 6), fontsize=7, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    save(fig, "fig_degree_loglog")

    # 4. ego subgraph of one citing paper ----------------------------------
    def kws(u, v):
        return set(split_keywords(g.edges[u, v].get("keywords", "")))

    def claims_of(p):
        return [q for q in g.neighbors(p) if types[q] == "claim" and "supports" in kws(p, q)]

    best, best_targets = None, []
    for p in g.nodes:
        if types[p] != "paper" or p == HUB or is_placeholder(p):
            continue
        targets = []
        for q in g.neighbors(p):
            if types[q] != "paper" or q == HUB or is_placeholder(q) or "cites" not in kws(p, q):
                continue
            if cited_endpoint(g, p, q) != q:
                continue
            if claims_of(q) and has_citing_sentence(g.edges[p, q].get("description", ""), strict=True):
                targets.append(q)
        if len(targets) > len(best_targets):
            best, best_targets = p, targets
    targets = sorted(best_targets, key=lambda q: (-len(claims_of(q)), q))[:7]
    H = nx.DiGraph()
    H.add_node(best, kind="paper")
    n_t = len(targets)
    row_h = 1.7
    ys = [(n_t - 1) / 2 * row_h - i * row_h for i in range(n_t)]
    pos = {best: (0.0, 0.0)}
    claim_rows = []
    for q, y in zip(targets, ys):
        H.add_node(q, kind="paper")
        H.add_edge(best, q, label="cites")
        pos[q] = (1.0, y)
        cl = sorted(claims_of(q), key=lambda c: (-deg[c], c))[:2]
        offs = [0.0] if len(cl) == 1 else [0.42, -0.42]
        for c, o in zip(cl, offs):
            H.add_node(c, kind="claim")
            H.add_edge(q, c, label="supports")
            pos[c] = (2.05, y + o)
            claim_rows.append(c)
    fig, ax = plt.subplots(figsize=(7.1, max(3.4, 0.95 * n_t + 0.8)))
    col = {n: (PALETTE[0] if H.nodes[n]["kind"] == "paper" else PALETTE[1]) for n in H}
    cites_edges = [(u, v) for u, v, d in H.edges(data=True) if d["label"] == "cites"]
    sup_edges = [(u, v) for u, v, d in H.edges(data=True) if d["label"] == "supports"]
    nx.draw_networkx_edges(H, pos, ax=ax, edgelist=cites_edges, edge_color=PALETTE[0], width=0.9, arrows=True,
                           arrowstyle="-|>", arrowsize=8, node_size=200, min_source_margin=14, min_target_margin=10)
    nx.draw_networkx_edges(H, pos, ax=ax, edgelist=sup_edges, edge_color=PALETTE[1], width=0.9, arrows=True,
                           arrowstyle="-|>", arrowsize=8, node_size=200, min_source_margin=10, min_target_margin=10)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[col[n] for n in H], node_size=[380 if n == best else 170 for n in H], linewidths=0)
    for n, (x, y) in pos.items():
        if n == best:
            ax.text(x, y - 0.32, textwrap.fill(n, 26), ha="center", va="top", fontsize=6.5, color=INK, fontweight="bold")
        elif H.nodes[n]["kind"] == "paper":
            ax.text(x, y + 0.16, textwrap.fill(n, 30), ha="center", va="bottom", fontsize=5.6, color=INK)
        else:
            ax.text(x + 0.07, y, textwrap.fill(n, 44), ha="left", va="center", fontsize=5.4, color=INK)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(min(pos[c][1] for c in pos) - 0.9, max(pos[c][1] for c in pos) + 0.7)
    ax.axis("off")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[0], markersize=7, label="paper node"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[1], markersize=7, label="claim node"),
                       Line2D([0], [0], color=PALETTE[0], lw=1.2, label="cites (citing \u2192 cited)"),
                       Line2D([0], [0], color=PALETTE[1], lw=1.2, label="supports (cited \u2192 claim)")],
              loc="upper left", frameon=False, fontsize=6.5, bbox_to_anchor=(0.0, 1.02))
    save(fig, "fig_ego_subgraph")
    ego_meta = dict(citing=best, cited=targets, claims=claim_rows,
                    sentences={q: g.edges[best, q].get("description", "") for q in targets})

    # 5. overview of one document's subgraph ------------------------------
    ds = load_kv("doc_status")
    files = [d["file_path"] for d in ds.values() if d["status"] == "processed"]
    members = {f: [n for n, d in g.nodes(data=True) if f in node_file_paths(d)] for f in files}
    target_n = 450
    doc = min(files, key=lambda f: abs(len(members[f]) - target_n))
    S_full = g.subgraph(members[doc]).copy()
    lcc = max(nx.connected_components(S_full), key=len)
    S = S_full.subgraph(lcc).copy()
    pos = nx.spring_layout(S, seed=7, k=2.6 / np.sqrt(S.number_of_nodes()), iterations=200)
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    nx.draw_networkx_edges(S, pos, ax=ax, edge_color=INK2, alpha=0.18, width=0.5)
    sizes = [6 + 2.2 * min(S.degree(n), 60) for n in S]
    nx.draw_networkx_nodes(S, pos, ax=ax, node_color=[type_color(types[n]) for n in S], node_size=sizes, linewidths=0, alpha=0.95)
    top = sorted(S.nodes, key=lambda n: -S.degree(n))[:1]
    for n in top:
        ax.text(pos[n][0], pos[n][1] + 0.04, textwrap.fill(n, 34), fontsize=6, ha="center", va="bottom", color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=AXIS, lw=0.4, alpha=0.9))
    cnt = Counter(types[n] for n in S)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[s], markersize=6, label=f"{t} ({cnt.get(t, 0)})")
               for t, s in TYPE_SLOT.items() if cnt.get(t, 0)]
    other_n = sum(v for t, v in cnt.items() if t not in TYPE_SLOT)
    if other_n:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=6, label=f"other ({other_n})"))
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.5, ncol=2)
    ax.axis("off")
    save(fig, "fig_overview")
    meta = dict(ego=ego_meta, overview=dict(document=doc, nodes=S.number_of_nodes(), edges=S.number_of_edges(),
                                              nodes_all=S_full.number_of_nodes(), edges_all=S_full.number_of_edges(), components_all=nx.number_connected_components(S_full),
                                              types=dict(cnt.most_common()), labelled=top))
    (DATA / "figures_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1)[:1800])


if __name__ == "__main__":
    main()
