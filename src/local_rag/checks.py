from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_rag.config import Settings

MIN_PYTHON = (3, 10)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_python(version: tuple[int, ...] = tuple(sys.version_info[:3])) -> CheckResult:
    shown = ".".join(map(str, version))
    if version[:2] >= MIN_PYTHON:
        return CheckResult("python", True, shown)
    needed = ".".join(map(str, MIN_PYTHON))
    return CheckResult("python", False, f"{shown} found, {needed}+ required")


def check_resources(resources_dir: Path) -> CheckResult:
    if not resources_dir.is_dir():
        return CheckResult("resources", False, f"{resources_dir} is not a directory")
    docs = [p for p in resources_dir.rglob("*.docx") if _is_document(p, resources_dir)]
    if not docs:
        return CheckResult("resources", False, f"no .docx files under {resources_dir}")
    folders = {p.parent for p in docs}
    return CheckResult(
        "resources",
        True,
        f"{len(docs)} .docx files in {len(folders)} folders under {resources_dir}",
    )


def check_ollama(client: Any, host: str, model: str) -> list[CheckResult]:
    try:
        listing = client.list()
    except Exception as exc:
        return [
            CheckResult("ollama", False, f"cannot reach {host}: {exc}"),
            CheckResult("llm model", False, f"{model} not checked (server unreachable)"),
        ]
    available = {m.model for m in listing.models if m.model}
    wanted = normalize_model_tag(model)
    server = CheckResult("ollama", True, f"reachable at {host}")
    if wanted in available:
        return [server, CheckResult("llm model", True, f"{wanted} is pulled")]
    return [server, CheckResult("llm model", False, f"{wanted} missing; run: ollama pull {wanted}")]


def normalize_model_tag(model: str) -> str:
    name = model.rsplit("/", 1)[-1]
    return model if ":" in name else f"{model}:latest"


def run_checks(settings: Settings, client_factory: Callable[[str], Any]) -> list[CheckResult]:
    return [
        check_python(),
        check_resources(settings.resources_dir),
        *check_ollama(
            client_factory(settings.ollama_host), settings.ollama_host, settings.llm_model
        ),
    ]


def _is_document(path: Path, root: Path) -> bool:
    # Skip hidden files and Word ~$ lock files.
    parts = path.relative_to(root).parts
    return path.is_file() and not any(part.startswith((".", "~$")) for part in parts)
