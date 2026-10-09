#!/usr/bin/env bash
# Synchronise the article working folder into the published repository clone.
# Usage: ./publish.sh [/path/to/citation-kg-verification]
# Everything that is generated, private or too heavy for git is excluded below.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="${1:-$SRC/../citation-kg-verification}"
mkdir -p "$DST"
rsync -a --delete \
  --exclude='.git/' \
  --exclude='LightRAG-fork/' \
  --exclude='*.aux' --exclude='*.fls' --exclude='*.fdb_latexmk' --exclude='*.out' --exclude='*.blg' \
  --exclude='/*.log' --exclude='/*.synctex.gz' \
  --exclude='/Screenshot*' --exclude='/archive/' --exclude='/RECAP_modifications.md' --exclude='/QuestionTests/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.mypy_cache/' \
  --exclude='/LightRAG/.git/' --exclude='/LightRAG/.venv/' --exclude='/LightRAG/venv/' \
  --exclude='/LightRAG/lightrag_webui/node_modules/' \
  --exclude='/LightRAG/rag_storage/' --exclude='/LightRAG/inputs/' --exclude='/LightRAG/output/' \
  --exclude='/LightRAG/.env' --exclude='/LightRAG/.env.backup.*' \
  --exclude='/LightRAG/examples/knowledge_graph.html' \
  "$SRC/" "$DST/"
echo "synced $SRC -> $DST"
