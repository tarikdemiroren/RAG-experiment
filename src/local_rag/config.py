from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

DEFAULTS: dict[str, str] = {
    "RAG_RESOURCES_DIR": "resources",
    "RAG_DB_DIR": "chroma_db",
    "RAG_COLLECTION": "recipes",
    "RAG_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "RAG_LLM_MODEL": "llama3.2:3b",
    "RAG_OLLAMA_HOST": "http://127.0.0.1:11434",
    "RAG_ALLOW_REMOTE_OLLAMA": "false",
    "RAG_CHUNK_SIZE": "700",
    "RAG_CHUNK_OVERLAP": "100",
    "RAG_TOP_K": "4",
    "RAG_MAX_DISTANCE": "0.6",
}

ENV_DEFAULTS: dict[str, str] = {
    "ANONYMIZED_TELEMETRY": "False",  # Chroma
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}

_COLLECTION_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,61}[a-zA-Z0-9]$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    resources_dir: Path
    db_dir: Path
    collection: str
    embedding_model: str
    llm_model: str
    ollama_host: str
    allow_remote_ollama: bool
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_distance: float

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env

        def get(key: str) -> str:
            value = source.get(key, "").strip()
            return value or DEFAULTS[key]

        allow_remote = _parse_bool("RAG_ALLOW_REMOTE_OLLAMA", get("RAG_ALLOW_REMOTE_OLLAMA"))
        chunk_size = _parse_int("RAG_CHUNK_SIZE", get("RAG_CHUNK_SIZE"), minimum=100)
        chunk_overlap = _parse_int("RAG_CHUNK_OVERLAP", get("RAG_CHUNK_OVERLAP"), minimum=0)
        if chunk_overlap >= chunk_size:
            raise ConfigError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")

        collection = get("RAG_COLLECTION")
        if not _COLLECTION_RE.match(collection):
            raise ConfigError(
                "RAG_COLLECTION must be 3-63 characters of letters, digits, '.', '_' or '-', "
                "starting and ending with a letter or digit"
            )

        return cls(
            resources_dir=Path(get("RAG_RESOURCES_DIR")).expanduser(),
            db_dir=Path(get("RAG_DB_DIR")).expanduser(),
            collection=collection,
            embedding_model=get("RAG_EMBEDDING_MODEL"),
            llm_model=get("RAG_LLM_MODEL"),
            ollama_host=validate_ollama_host(get("RAG_OLLAMA_HOST"), allow_remote=allow_remote),
            allow_remote_ollama=allow_remote,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=_parse_int("RAG_TOP_K", get("RAG_TOP_K"), minimum=1, maximum=50),
            max_distance=_parse_float("RAG_MAX_DISTANCE", get("RAG_MAX_DISTANCE"), 0.0, 2.0),
        )


def load_settings() -> Settings:
    """Settings from the environment and the .env in the current directory."""
    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    apply_env_defaults()
    return Settings.from_env()


def validate_ollama_host(url: str, *, allow_remote: bool = False) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"RAG_OLLAMA_HOST must be an http(s) URL, got {url!r}")
    if parsed.username or parsed.password:
        raise ConfigError("RAG_OLLAMA_HOST must not contain credentials")
    if not allow_remote and not _is_loopback(parsed.hostname):
        raise ConfigError(
            f"RAG_OLLAMA_HOST {parsed.hostname!r} is not local; "
            "set RAG_ALLOW_REMOTE_OLLAMA=true to allow it"
        )
    return url.rstrip("/")


def apply_env_defaults(env: MutableMapping[str, str] | None = None) -> None:
    target = os.environ if env is None else env
    for key, value in ENV_DEFAULTS.items():
        target.setdefault(key, value)


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _parse_bool(key: str, raw: str) -> bool:
    value = raw.lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{key} must be true or false, got {raw!r}")


def _parse_int(key: str, raw: str, *, minimum: int, maximum: int | None = None) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from None
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ConfigError(f"{key} must be {bounds}, got {value}")
    return value


def _parse_float(key: str, raw: str, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from None
    if not minimum <= value <= maximum:
        raise ConfigError(f"{key} must be between {minimum} and {maximum}, got {value}")
    return value
