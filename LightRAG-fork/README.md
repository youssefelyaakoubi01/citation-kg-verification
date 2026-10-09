# State of the LightRAG fork

`LightRAG/` in this repository is the working tree of a fork of [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG), branch `feat/claude-agent-sdk-scientific-kg`, published without its git history. This folder lets you rebuild it on a real clone of upstream.

- `UPSTREAM_BASE`: the upstream commit the fork starts from.
- `0001-…patch`, `0002-…patch`, `0003-…patch`: the three commits of the fork (`git format-patch`).
- `uncommitted.diff`: changes present in the working tree but not yet committed at the time of the article (LABEL lookup, chunk-selection fallback fix, no-op vector storage, tests). `uncommitted.stat` and `status.txt` summarise them. Untracked files in `status.txt` (`tests/test_operate_label_lookup.py`, `examples/lib/`) are included in `LightRAG/` directly.

```bash
git clone https://github.com/HKUDS/LightRAG.git && cd LightRAG
git checkout -b feat/claude-agent-sdk-scientific-kg "$(cat ../LightRAG-fork/UPSTREAM_BASE)"
git am ../LightRAG-fork/000*.patch
git apply ../LightRAG-fork/uncommitted.diff
```
