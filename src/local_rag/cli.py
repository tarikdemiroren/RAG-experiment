from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from local_rag import __version__
from local_rag.checks import run_checks
from local_rag.config import ConfigError, Settings, apply_privacy_defaults


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    apply_privacy_defaults()

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        return _cmd_check(settings)
    parser.error(f"unknown command {args.command!r}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-rag", description="Fully local RAG over your own documents."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="verify Python, the resources folder and Ollama")
    return parser


def _cmd_check(settings: Settings) -> int:
    import ollama

    results = run_checks(settings, lambda host: ollama.Client(host=host))
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{'ok' if r.ok else 'FAIL':>4}] {r.name:<{width}}  {r.detail}")
    return 0 if all(r.ok for r in results) else 1
