# Local RAG

Question answering over local documents with a local LLM.

- LLM: Ollama `llama3.2:3b`
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: Chroma (embedded, on disk)

## Setup

```bash
brew install ollama
ollama serve
ollama pull llama3.2:3b

python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements-dev.txt
pip install --no-deps -e .

local-rag check
```

Put documents in `resources/`, one folder per document; all `.docx` files in a folder are
merged and the folder name is used as the title. Settings are `RAG_*` environment variables; see
`.env.example`.

## Usage

```bash
local-rag ingest                          # rebuild the index from resources/
local-rag search "boiled eggs"            # show retrieved chunks and their scores
local-rag ask "How long do I boil eggs?"  # answer with sources; --show-context to see chunks
local-rag ui                              # web UI on http://127.0.0.1:8501
```

Recipes added in the UI are saved to `resources/uploads/<name>/`.

Retrieval is hybrid by default: vector similarity and BM25 keyword scores, merged with
Reciprocal Rank Fusion. Compare with `--mode vector` or `--mode keyword`.

## Development

```bash
pytest
ruff check . && ruff format --check .
pip-audit -r requirements.txt
```

Update dependencies by editing `pyproject.toml`, then:

```bash
pip-compile --generate-hashes --strip-extras --allow-unsafe -o requirements.txt pyproject.toml
pip-compile --generate-hashes --strip-extras --allow-unsafe --extra dev -o requirements-dev.txt pyproject.toml
```

Chroma is used only as an embedded `PersistentClient`. `pip-audit` reports CVE-2026-45829,
-45830, -45831 and -45833 for chromadb; they affect Chroma's HTTP server mode, not embedded use.
