"""Shared helpers for the article's reproducible analyses.

Every script in this folder reads the LightRAG working directory
(``rag_storage/``) and the server log READ-ONLY and writes its outputs next to
the article (``data/``, ``tables/``, ``figures/``). Numbers quoted in the
article come from ``data/numbers_*.tex`` macros written here, never from hand.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

# Paths are resolved relative to this file so the scripts run from any clone
# of the published repository (https://github.com/youssefelyaakoubi01/citation-kg-verification).
ANALYSIS = Path(__file__).resolve().parent
PAPER = ANALYSIS.parent
REPO = PAPER / "LightRAG"
# The live server working directory when present (as on the author's machine),
# otherwise the frozen copy shipped with the article.
STORAGE = REPO / "rag_storage" if (REPO / "rag_storage").is_dir() else ANALYSIS / "rag_storage_pilot"
LOG = REPO / "lightrag.log"
# The server .env is never published (it holds credentials for unused back
# ends); .env.article carries only the configuration keys these scripts read.
ENV = REPO / ".env" if (REPO / ".env").is_file() else REPO / ".env.article"
DATA = PAPER / "data"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

for d in (DATA, TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

ENTITY_TYPES = [
    "paper", "author", "organization", "venue", "identifier",
    "methodology", "dataset", "concept", "claim", "other",
]
VOCAB = [
    "cites", "supports", "contradicts", "authored_by", "affiliated_with",
    "published_in", "has_identifier", "proposes", "uses_method",
    "uses_dataset", "extends", "compares_with", "addresses", "related_to",
]
HUB = "This Paper"
PLACEHOLDER_RE = re.compile(r"^(reference|ref\.?)\s*\[?\d+\]?$|^\[\d+\]$", re.I)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.I)
SEP = "<SEP>"


def load_kv(name: str) -> dict:
    return json.loads((STORAGE / f"kv_store_{name}.json").read_text())


def load_graph():
    import networkx as nx

    return nx.read_graphml(STORAGE / "graph_chunk_entity_relation.graphml")


def canonical_keyword(keyword: str) -> str:
    return "_".join(keyword.strip().casefold().split())


def split_keywords(keywords: str) -> list[str]:
    return [
        canonical_keyword(p)
        for p in str(keywords or "").replace("，", ",").split(",")
        if p.strip()
    ]


def is_placeholder(name: str) -> bool:
    return bool(PLACEHOLDER_RE.match(str(name).strip()))


def has_citing_sentence(description: str, strict: bool = False) -> bool:
    """Whether a ``cites`` edge description carries a citation context.

    strict: the description quotes a bracketed marker ``[n]`` or contains
    quotation marks (the profile asks for the citing sentence verbatim).
    lenient (default): strict, or at least eight words (a sentence-length
    paraphrase rather than a bare label).
    """
    d = str(description or "")
    if re.search(r"\[\d+\]", d):
        return True
    if '"' in d or "\u201c" in d or "\u201d" in d:
        return True
    return (not strict) and len(d.split()) >= 8


def read_env(keys: list[str]) -> dict[str, str]:
    """Values of uncommented ``KEY=value`` lines of the runtime .env."""
    values: dict[str, str] = {}
    for line in ENV.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line.strip())
        if m and m.group(1) in keys:
            values[m.group(1)] = m.group(2).strip()
    return values


def load_validator_module():
    """Import examples/scientific_citation_kg.py without a package install."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(
        "scientific_citation_kg", REPO / "examples" / "scientific_citation_kg.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------- LaTeX ----

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def fmt_int(n) -> str:
    return f"{int(round(n)):,}"


def fmt_float(x, digits=1) -> str:
    return f"{x:,.{digits}f}"


def fmt_pct(x, digits=1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval (95% by default) for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci_pct(lo: float, hi: float, digits=0) -> str:
    """``[lo, hi]`` as percentages, e.g. ``[33\\%, 60\\%]``."""
    return f"[{100 * lo:.{digits}f}\\%, {100 * hi:.{digits}f}\\%]"


def write_macros(path: Path, macros: dict[str, str]) -> None:
    lines = ["% Generated by analysis/*.py -- do not edit by hand."]
    for name, value in macros.items():
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"macro name must be letters only: {name}")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    path.write_text("\n".join(lines) + "\n")


def write_table(
    path: Path,
    *,
    caption: str,
    label: str,
    colspec: str,
    header: list[str],
    rows: list[list[str]],
    wide: bool = False,
    size: str = r"\footnotesize",
    note: str | None = None,
    placement: str = "!t",
) -> None:
    env = "table*" if wide else "table"
    out = [
        f"% Generated by analysis/*.py -- do not edit by hand.",
        f"\\begin{{{env}}}[{placement}]",
        r"\centering",
        size,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    out += [" & ".join(r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out.append(r"\par\smallskip\begin{minipage}{" + ("0.95\\textwidth" if wide else "0.95\\columnwidth") + "}" + size + " " + note + r"\end{minipage}")
    out.append(f"\\end{{{env}}}")
    path.write_text("\n".join(out) + "\n")


def node_file_paths(data: dict) -> set[str]:
    return {p for p in str(data.get("file_path", "")).split(SEP) if p}


def cited_endpoint(graph, u: str, v: str) -> str:
    """Which endpoint of an undirected ``cites`` edge is the cited paper.

    The graph stores edges undirected, so direction is recovered by
    convention: the generic hub ``This Paper`` is always the citing side;
    otherwise the lower-degree endpoint is taken as the cited work (citing
    articles accumulate hundreds of edges, cited entries a handful).
    """
    if u == HUB:
        return v
    if v == HUB:
        return u
    return v if graph.degree(v) <= graph.degree(u) else u
