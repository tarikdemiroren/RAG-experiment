from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import ollama

from local_rag import __version__, pipeline
from local_rag.answer import NO_MATCH, cited, stream_answer
from local_rag.checks import run_checks
from local_rag.config import ConfigError, Settings, load_settings
from local_rag.loaders import MAX_FILE_BYTES
from local_rag.pipeline import NothingToIngestError
from local_rag.retrieval import MODES
from local_rag.store import Hit, IndexMissingError


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "check":
            return _cmd_check(settings)
        if args.command == "ingest":
            return _cmd_ingest(settings)
        if args.command == "ui":
            return _cmd_ui()
        if args.command == "search":
            return _cmd_search(settings, args.query, args.mode)
        return _cmd_ask(settings, args.question, args.mode, args.show_context)
    except (IndexMissingError, NothingToIngestError) as exc:
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
    commands.add_parser("ui", help="start the web UI on http://127.0.0.1:8501")
    search = commands.add_parser("search", help="show the chunks retrieved for a query")
    search.add_argument("query")
    ask = commands.add_parser("ask", help="answer a question from the indexed documents")
    ask.add_argument("question")
    ask.add_argument("--show-context", action="store_true", help="print the retrieved chunks")
    for command in (search, ask):
        command.add_argument("--mode", choices=MODES, default="hybrid", help="retrieval method")
    return parser


def _cmd_check(settings: Settings) -> int:
    results = run_checks(settings, lambda host: ollama.Client(host=host))
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{'ok' if r.ok else 'FAIL':>4}] {r.name:<{width}}  {r.detail}")
    return 0 if all(r.ok for r in results) else 1


def _cmd_ingest(settings: Settings) -> int:
    embedder = pipeline.make_embedder(settings)
    result = pipeline.ingest(settings, embedder, pipeline.make_store(settings))
    for path, reason in result.skipped:
        print(f"skipped {path}: {reason}")
    for chunk_id, tokens in result.truncated:
        print(
            f"warning: {chunk_id} has {tokens} tokens; only the first "
            f"{embedder.max_tokens} are embedded"
        )
    print(f"indexed {result.chunks} chunks from {result.recipes} recipes")
    return 0


def _cmd_ui() -> int:
    script = Path(__file__).with_name("ui.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address=127.0.0.1",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
        f"--server.maxUploadSize={MAX_FILE_BYTES // 2**20}",
    ]
    try:
        return subprocess.call(command)  # noqa: S603
    except KeyboardInterrupt:
        return 0


def _cmd_search(settings: Settings, query: str, mode: str) -> int:
    retriever = pipeline.make_retriever(settings)
    for i, hit in enumerate(retriever.search(query, settings.top_k, mode), 1):
        _print_hit(i, hit)
    return 0


def _cmd_ask(settings: Settings, question: str, mode: str, show_context: bool) -> int:
    hits = pipeline.make_retriever(settings).search(
        question, settings.top_k, mode, max_distance=settings.max_distance
    )
    if not hits:
        print(NO_MATCH)
        return 0
    if show_context:
        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit)

    client = ollama.Client(host=settings.ollama_host)
    parts = []
    for token in stream_answer(client, settings.llm_model, question, hits):
        parts.append(token)
        print(token, end="", flush=True)
    print()
    sources = cited("".join(parts), hits)
    if sources:
        print("\nSources:")
        for n, hit in sources:
            print(f"[{n}] {hit.title}")
    return 0


def _print_hit(i: int, hit: Hit) -> None:
    print(
        f"--- [{i}] {hit.title}  id={hit.id}\n"
        f"    distance={hit.distance:.3f}  keyword={hit.keyword_score:.2f} "
        f"({hit.keyword_coverage:.0%} of terms)  fused={hit.score:.4f}"
    )
    print(hit.text, end="\n\n")
