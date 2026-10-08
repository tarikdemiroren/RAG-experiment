from conftest import FakeOllamaClient

from local_rag.checks import (
    check_ollama,
    check_python,
    check_resources,
    normalize_model_tag,
    run_checks,
)
from local_rag.config import Settings

HOST = "http://127.0.0.1:11434"


def test_python_version_supported():
    assert check_python((3, 13, 1)).ok
    assert check_python((3, 10, 0)).ok


def test_python_version_too_old():
    result = check_python((3, 9, 6))
    assert not result.ok
    assert "3.10+" in result.detail


def test_resources_missing_dir(tmp_path):
    result = check_resources(tmp_path / "nope")
    assert not result.ok
    assert "not a directory" in result.detail


def test_resources_without_documents(tmp_path):
    (tmp_path / "Empty folder").mkdir()
    (tmp_path / "notes.txt").write_text("hi")
    result = check_resources(tmp_path)
    assert not result.ok
    assert "no .docx" in result.detail


def test_resources_counts_documents_and_folders(tmp_path, make_docx):
    make_docx(tmp_path / "recipes" / "Pancake" / "Ingredients.docx", "flour")
    make_docx(tmp_path / "recipes" / "Pancake" / "Recipe.docx", "mix")
    make_docx(tmp_path / "recipes" / "Hummus" / "Untitled.docx", "chickpeas")
    result = check_resources(tmp_path)
    assert result.ok
    assert result.detail.startswith("3 .docx files in 2 folders")


def test_resources_skips_hidden_and_lock_files(tmp_path, make_docx):
    make_docx(tmp_path / "Pancake" / "Recipe.docx", "mix")
    make_docx(tmp_path / "Pancake" / "~$Recipe.docx", "lock")
    make_docx(tmp_path / ".trash" / "Old.docx", "old")
    result = check_resources(tmp_path)
    assert result.detail.startswith("1 .docx files in 1 folders")


def test_ollama_ready():
    results = check_ollama(FakeOllamaClient(["llama3.2:3b", "other:7b"]), HOST, "llama3.2:3b")
    assert [r.ok for r in results] == [True, True]
    assert HOST in results[0].detail


def test_ollama_model_missing():
    server, model = check_ollama(FakeOllamaClient(["llama3.2:1b"]), HOST, "llama3.2:3b")
    assert server.ok
    assert not model.ok
    assert "ollama pull llama3.2:3b" in model.detail


def test_ollama_unreachable():
    client = FakeOllamaClient(error=ConnectionError("Failed to connect"))
    server, model = check_ollama(client, HOST, "llama3.2:3b")
    assert not server.ok
    assert "Failed to connect" in server.detail
    assert not model.ok


def test_untagged_model_matches_latest():
    _, model = check_ollama(FakeOllamaClient(["mistral:latest"]), HOST, "mistral")
    assert model.ok


def test_normalize_model_tag():
    assert normalize_model_tag("llama3.2") == "llama3.2:latest"
    assert normalize_model_tag("llama3.2:3b") == "llama3.2:3b"
    assert (
        normalize_model_tag("registry.local:5000/llama3.2") == "registry.local:5000/llama3.2:latest"
    )


def test_run_checks_passes_configured_host(tmp_path, make_docx):
    make_docx(tmp_path / "Pancake" / "Recipe.docx", "mix")
    settings = Settings.from_env({"RAG_RESOURCES_DIR": str(tmp_path)})
    seen = []

    def factory(host):
        seen.append(host)
        return FakeOllamaClient(["llama3.2:3b"])

    results = run_checks(settings, factory)
    assert seen == [HOST]
    assert [r.name for r in results] == ["python", "resources", "ollama", "llm model"]
    assert all(r.ok for r in results)
