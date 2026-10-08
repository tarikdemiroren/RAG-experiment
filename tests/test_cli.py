import os

import pytest
from conftest import FakeOllamaClient

from local_rag import cli


@pytest.fixture
def fake_ollama(monkeypatch):
    hosts = []
    state = {"models": ["llama3.2:3b"], "error": None}

    def factory(host):
        hosts.append(host)
        return FakeOllamaClient(state["models"], state["error"])

    monkeypatch.setattr("ollama.Client", factory)
    state["hosts"] = hosts
    return state


def test_check_all_ok(isolated_env, make_docx, fake_ollama, capsys):
    make_docx(isolated_env / "resources" / "Pancake" / "Recipe.docx", "mix")
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "[  ok] resources" in out
    assert "[  ok] llm model" in out


def test_check_reports_failures(isolated_env, fake_ollama, capsys):
    fake_ollama["models"] = []
    assert cli.main(["check"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] resources" in out
    assert "[FAIL] llm model" in out


def test_dotenv_in_cwd_is_loaded(isolated_env, fake_ollama):
    (isolated_env / ".env").write_text("RAG_OLLAMA_HOST=http://localhost:9999\n")
    cli.main(["check"])
    assert fake_ollama["hosts"] == ["http://localhost:9999"]


def test_dotenv_does_not_override_real_env(isolated_env, fake_ollama):
    (isolated_env / ".env").write_text("RAG_OLLAMA_HOST=http://localhost:9999\n")
    os.environ["RAG_OLLAMA_HOST"] = "http://localhost:7777"
    cli.main(["check"])
    assert fake_ollama["hosts"] == ["http://localhost:7777"]


def test_dotenv_in_parent_dir_is_ignored(isolated_env, fake_ollama, monkeypatch):
    (isolated_env / ".env").write_text("RAG_OLLAMA_HOST=http://localhost:9999\n")
    child = isolated_env / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    cli.main(["check"])
    assert fake_ollama["hosts"] == ["http://127.0.0.1:11434"]


def test_config_error_exits_2(isolated_env, fake_ollama, capsys):
    os.environ["RAG_OLLAMA_HOST"] = "http://203.0.113.7:11434"
    assert cli.main(["check"]) == 2
    assert "config error" in capsys.readouterr().err
    assert fake_ollama["hosts"] == []


def test_privacy_env_applied(isolated_env, fake_ollama):
    os.environ.pop("ANONYMIZED_TELEMETRY", None)
    cli.main(["check"])
    assert os.environ["ANONYMIZED_TELEMETRY"] == "False"


def test_command_required(isolated_env):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
