from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import ollama
from dotenv import load_dotenv

from local_rag import __version__
from local_rag.answer import NO_MATCH, relevant, stream_answer
from local_rag.checks import run_checks
from local_rag.chunking import chunk_recipe
from local_rag.config import ConfigError, Settings, apply_env_defaults
from local_rag.embeddings import Embedder, SentenceTransformerEmbedder
from local_rag.loaders import load_recipes
from local_rag.store import Hit, IndexMissingError, VectorStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    apply_env_defaults()

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "check":
            return _cmd_check(settings)
        if args.command == "ingest":
            return _cmd_ingest(settings)
        if args.command == "search":
            return _cmd_search(settings, args.query)
        return _cmd_ask(settings, args.question, args.show_context)
    except IndexMissingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-rag", description="Fully local RAG over your own documents."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="verify Python, the resources folder and Ollama")
    commands.add_parser("ingest", help="rebuild the index from the resources folder")
    search = commands.add_parser("search", help="show the chunks retrieved for a query")
    search.add_argument("query")
    ask = commands.add_parser("ask", help="answer a question from the indexed documents")
    ask.add_argument("question")
    ask.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    return parser


def _cmd_check(settings: Settings) -> int:
    results = run_checks(settings, lambda host: ollama.Client(host=host))
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{'ok' if r.ok else 'FAIL':>4}] {r.name:<{width}}  {r.detail}")
    return 0 if all(r.ok for r in results) else 1


def _cmd_ingest(settings: Settings) -> int:
    report = load_recipes(settings.resources_dir)
    for path, reason in report.skipped:
        print(f"skipped {path}: {reason}")
    chunks = [
        chunk
        for recipe in report.recipes
        for chunk in chunk_recipe(recipe, settings.chunk_size, settings.chunk_overlap)
    ]
    if not chunks:
        print(f"error: no documents with text under {settings.resources_dir}", file=sys.stderr)
        return 1

    embedder = make_embedder(settings)
    for chunk in chunks:
        tokens = embedder.count_tokens(chunk.text)
        if tokens > embedder.max_tokens:
            print(
                f"warning: {chunk.id} has {tokens} tokens; only the first "
                f"{embedder.max_tokens} are embedded"
            )
    vectors = embedder.embed([c.text for c in chunks])
    make_store(settings).rebuild(chunks, vectors)
    print(f"indexed {len(chunks)} chunks from {len(report.recipes)} recipes")
    return 0


def _cmd_search(settings: Settings, query: str) -> int:
    for i, hit in enumerate(_retrieve(settings, query), 1):
        _print_hit(i, hit)
    return 0


def _cmd_ask(settings: Settings, question: str, show_context: bool) -> int:
    hits = _retrieve(settings, question)
    if show_context:
        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit)
    hits = relevant(hits, settings.max_distance)
    if not hits:
        print(NO_MATCH)
        return 0

    client = ollama.Client(host=settings.ollama_host)
    for token in stream_answer(client, settings.llm_model, question, hits):
        print(token, end="", flush=True)
    print("\n\nSources:")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit.title} ({hit.source}, distance {hit.distance:.3f})")
    return 0


def _retrieve(settings: Settings, query: str) -> list[Hit]:
    vector = make_embedder(settings).embed([query])[0]
    return make_store(settings).query(vector, settings.top_k)


def _print_hit(i: int, hit: Hit) -> None:
    print(f"--- [{i}] {hit.title}  distance={hit.distance:.3f}  id={hit.id}")
    print(hit.text, end="\n\n")


def make_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(settings.embedding_model)


def make_store(settings: Settings) -> VectorStore:
    return VectorStore(settings.db_dir, settings.collection)
