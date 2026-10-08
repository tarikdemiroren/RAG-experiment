from pathlib import Path

import pytest

from local_rag.config import (
    DEFAULTS,
    ENV_DEFAULTS,
    ConfigError,
    Settings,
    apply_env_defaults,
    validate_ollama_host,
)


def test_defaults():
    s = Settings.from_env({})
    assert s.resources_dir == Path("resources")
    assert s.db_dir == Path("chroma_db")
    assert s.collection == "recipes"
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.llm_model == "llama3.2:3b"
    assert s.ollama_host == "http://127.0.0.1:11434"
    assert s.allow_remote_ollama is False
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (700, 100, 4)
    assert s.max_distance == 0.6


def test_env_overrides_defaults():
    s = Settings.from_env(
        {
            "RAG_RESOURCES_DIR": "/data/docs",
            "RAG_COLLECTION": "my-docs",
            "RAG_LLM_MODEL": "llama3.2:1b",
            "RAG_CHUNK_SIZE": "500",
            "RAG_CHUNK_OVERLAP": "0",
            "RAG_TOP_K": "8",
            "RAG_MAX_DISTANCE": "0.45",
        }
    )
    assert s.resources_dir == Path("/data/docs")
    assert s.collection == "my-docs"
    assert s.llm_model == "llama3.2:1b"
    assert (s.chunk_size, s.chunk_overlap, s.top_k) == (500, 0, 8)
    assert s.max_distance == 0.45


def test_blank_values_fall_back_to_defaults():
    s = Settings.from_env({"RAG_LLM_MODEL": "   ", "RAG_TOP_K": ""})
    assert s.llm_model == DEFAULTS["RAG_LLM_MODEL"]
    assert s.top_k == 4


def test_home_is_expanded():
    s = Settings.from_env({"RAG_DB_DIR": "~/rag-db"})
    assert s.db_dir == Path.home() / "rag-db"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"RAG_CHUNK_SIZE": "big"}, "must be an integer"),
        ({"RAG_CHUNK_SIZE": "50"}, ">= 100"),
        ({"RAG_CHUNK_OVERLAP": "-1"}, ">= 0"),
        ({"RAG_CHUNK_SIZE": "200", "RAG_CHUNK_OVERLAP": "200"}, "smaller than"),
        ({"RAG_TOP_K": "0"}, "between 1 and 50"),
        ({"RAG_TOP_K": "51"}, "between 1 and 50"),
        ({"RAG_MAX_DISTANCE": "far"}, "must be a number"),
        ({"RAG_MAX_DISTANCE": "2.5"}, "between 0.0 and 2.0"),
        ({"RAG_ALLOW_REMOTE_OLLAMA": "maybe"}, "true or false"),
        ({"RAG_COLLECTION": "ab"}, "RAG_COLLECTION"),
        ({"RAG_COLLECTION": "-recipes"}, "RAG_COLLECTION"),
        ({"RAG_COLLECTION": "my recipes"}, "RAG_COLLECTION"),
    ],
)
def test_invalid_values_are_rejected(env, message):
    with pytest.raises(ConfigError, match=message):
        Settings.from_env(env)


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_values(raw):
    env = {"RAG_ALLOW_REMOTE_OLLAMA": raw, "RAG_OLLAMA_HOST": "http://gpu-box:11434"}
    assert Settings.from_env(env).allow_remote_ollama is True


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434",
        "http://127.8.9.10:11434",
        "http://[::1]:11434",
        "https://localhost",
    ],
)
def test_loopback_hosts_are_accepted(url):
    assert validate_ollama_host(url) == url


def test_trailing_slash_is_stripped():
    assert validate_ollama_host("http://localhost:11434/") == "http://localhost:11434"


@pytest.mark.parametrize(
    "url", ["http://192.168.1.20:11434", "http://0.0.0.0:11434", "http://example.com"]
)
def test_remote_hosts_need_opt_in(url):
    with pytest.raises(ConfigError, match="RAG_ALLOW_REMOTE_OLLAMA"):
        validate_ollama_host(url)
    assert validate_ollama_host(url, allow_remote=True) == url


def test_remote_host_rejected_through_settings():
    with pytest.raises(ConfigError, match="not local"):
        Settings.from_env({"RAG_OLLAMA_HOST": "http://10.0.0.5:11434"})


@pytest.mark.parametrize(
    "url", ["127.0.0.1:11434", "ftp://localhost", "file:///etc/passwd", "http://", ""]
)
def test_non_http_urls_are_rejected(url):
    with pytest.raises(ConfigError, match="http"):
        validate_ollama_host(url, allow_remote=True)


def test_credentials_in_url_are_rejected():
    with pytest.raises(ConfigError, match="credentials"):
        validate_ollama_host("http://user:secret@localhost:11434")


def test_env_defaults_are_set():
    env = {}
    apply_env_defaults(env)
    assert env == ENV_DEFAULTS
    assert env["ANONYMIZED_TELEMETRY"] == "False"


def test_env_defaults_do_not_override_user_choice():
    env = {"ANONYMIZED_TELEMETRY": "True"}
    apply_env_defaults(env)
    assert env["ANONYMIZED_TELEMETRY"] == "True"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
